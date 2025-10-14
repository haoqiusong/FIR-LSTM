import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data preprocessing

df = pd.read_csv('simulated_timeseries_dataset.csv')

patient_ids = df['patient_id'].values

X = df.drop(['label', 'patient_id'], axis=1).values
y = df['label'].values
feature_names = df.drop(['label', 'patient_id'], axis=1).columns.tolist()

unique_patients = np.unique(patient_ids)
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

def create_sequences(X, y, patient_ids, selected_patients, time_steps=6):
    Xs, ys = [], []
    for patient in selected_patients:
        patient_indices = np.where(patient_ids == patient)[0]
        patient_X = X[patient_indices]
        patient_y = y[patient_indices]

        if len(patient_X) <= time_steps:
            continue

        for i in range(len(patient_X) - time_steps):
            Xs.append(patient_X[i:(i + time_steps), :])
            ys.append(patient_y[i + time_steps])
    
    return np.array(Xs), np.array(ys)

X_train_sequence, y_train_sequence = create_sequences(X, y, patient_ids, train_patients)
X_test_sequence, y_test_sequence = create_sequences(X, y, patient_ids, test_patients)

X_train, X_val, y_train, y_val = train_test_split(X_train_sequence, y_train_sequence, test_size=0.3, random_state=42, stratify=y_train_sequence)

num_samples_train, seq_len_train, num_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, num_features)

# Fit the scaler on the training data only
scaler = MinMaxScaler()
scaler.fit(X_train_reshaped)

# Scale the training data
X_train_scaled = scaler.transform(X_train_reshaped)
X_train_scaled = X_train_scaled.reshape(num_samples_train, seq_len_train, num_features)

# Scale the validation data
num_samples_val = X_val.shape[0]
X_val_reshaped = X_val.reshape(-1, num_features)
X_val_scaled = scaler.transform(X_val_reshaped)
X_val_scaled = X_val_scaled.reshape(num_samples_val, seq_len_train, num_features)

# Scale the test data
num_samples_test = X_test_sequence.shape[0]
seq_len_test = X_test_sequence.shape[1]
X_test_reshaped = X_test_sequence.reshape(-1, num_features)
X_test_scaled = scaler.transform(X_test_reshaped)
X_test_scaled = X_test_scaled.reshape(num_samples_test, seq_len_test, num_features)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_sequence, dtype=torch.float32).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# Model

# LRP
def lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps=1e-6, bias_factor=0.0):
    """
    LRP for a linear layer with input dimension D and output dimension M.
    Args:
    - hin:        forward pass input, shape (batch_size, D)
    - w:          weights, shape (D, M)
    - b:          biases, shape (M,)
    - hout:       forward pass output, shape (batch_size, M)
    - Rout:       relevance at layer output, shape (batch_size, M)
    - bias_nb_units: total number of units in the lower layer
    - eps:        stabilizer for numerical stability
    - bias_factor: factor for bias redistribution
    Returns:
    - Rin:        relevance at layer input, shape (batch_size, D)
    """

    # Compute the Sign of Output Activations
    # Determines the sign of each output activation in hout to handle positive and negative activations differently, which is important for numerical stability.
    sign_out = torch.where(
        hout >= 0,
        torch.tensor(1.0, device=hout.device),
        torch.tensor(-1.0, device=hout.device)
    )  # Shape: (batch_size, M)

    # Reshape tensors for broadcasting
    # Prepares the input activations hin and weights w for element-wise multiplication using broadcasting.
    hin_ = hin.unsqueeze(2)  # Shape: (batch_size, D, 1)
    w_ = w.unsqueeze(0)      # Shape: (1, D, M)

    # Compute numerator
    # Calculates the contribution of each input activation and weight to each output activation.
    # Element-wise Multiplication: Each input activation hin is multiplied by each weight w connecting it to the outputs.
    numer = hin_ * w_  # Shape: (batch_size, D, M)

    # Compute bias term and expand to match numer shape
    # Distributes the relevance associated with the bias term to the inputs.
    # Divided by bias_nb_units: Normalizes the bias term distribution across the number of input units.
    bias_term = (bias_factor * (b * 1.0 + eps * sign_out) / bias_nb_units)
    bias_term = bias_term.unsqueeze(1).expand(-1, hin.size(1), -1)  # Shape: (batch_size, D, M)

    numer += bias_term  # Incorporates the bias contributions into the numerator. Shape: (batch_size, D, M)

    # Compute denominator and expand to match numer shape
    # Provides the normalization factor for the relevance redistribution, ensuring relevance conservation.
    denom = hout + (eps * sign_out)  # Shape: (batch_size, M)
    denom = denom.unsqueeze(1).expand(-1, hin.size(1), -1)  # Shape: (batch_size, D, M)

    # Compute message
    # Purpose: Calculates the amount of relevance to be passed from each output unit back to each input unit.
    # message contains the relevance messages from each output unit to each input unit.
    message = (numer / denom) * Rout.unsqueeze(1)  # Shape: (batch_size, D, M)

    # Sum over output dimension M to get Rin of shape (batch_size, D)
    # Aggregates the relevance messages from all output units for each input unit.
    # Rin contains the total relevance for each input unit, summing over all contributions from the outputs.
    Rin = message.sum(dim=2)

    return Rin

# LSTM model with activations tracking for LRP
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.batch_norm = nn.BatchNorm1d(hidden_dim)

        # LSTM parameters
        self.W_ih = nn.Parameter(torch.Tensor(4 * hidden_dim, input_dim))
        self.W_hh = nn.Parameter(torch.Tensor(4 * hidden_dim, hidden_dim))
        self.b_ih = nn.Parameter(torch.Tensor(4 * hidden_dim))
        self.b_hh = nn.Parameter(torch.Tensor(4 * hidden_dim))

        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Initialize parameters
        self.reset_parameters()

        # For storing intermediate values
        self.gates = []
        self.hiddens = []
        self.cells = []

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.hidden_dim)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x):
        self.x = x  # Shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        self.gates = []
        self.hiddens = [h_t]
        self.cells = [c_t]

        for t in range(seq_len):
            x_t = x[:, t, :]
            gates = (torch.mm(x_t, self.W_ih.t()) + self.b_ih) + (torch.mm(h_t, self.W_hh.t()) + self.b_hh)
            i_t, f_t, g_t, o_t = gates.chunk(4, 1)

            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            self.gates.append({'i': i_t, 'f': f_t, 'g': g_t, 'o': o_t})
            self.hiddens.append(h_t)
            self.cells.append(c_t)

        # Take the last hidden state
        final_out = self.batch_norm(h_t)
        final_out = self.dropout_layer(final_out)
        logits = self.fc(final_out)
        return logits

    def lrp(self, R_output, eps=1e-6, bias_factor=0.0):
        """
        Perform Layer-wise Relevance Propagation (LRP).
        Args:
        - R_output: Relevance at the output layer, shape (batch_size, output_dim)
        Returns:
        - R_input: Relevance at the input layer, shape (batch_size, seq_len, input_dim)
        """
        batch_size = R_output.size(0)
        seq_len = len(self.gates)
        hidden_dim = self.hidden_dim
        input_dim = self.input_dim

        # Relevance scores for hidden states and cell states
        R_h = [torch.zeros(batch_size, hidden_dim, device=R_output.device) for _ in range(seq_len + 1)]
        R_c = [torch.zeros(batch_size, hidden_dim, device=R_output.device) for _ in range(seq_len + 1)]

        # Start from the output relevance
        # Apply LRP to the output layer (fc layer)
        h_T = self.hiddens[-1]  # Shape: (batch_size, hidden_dim)
        w = self.fc.weight  # Shape: (output_dim, hidden_dim)
        b = self.fc.bias    # Shape: (output_dim,)

        # Compute relevance for the last hidden state
        fc_output = self.fc(h_T)  # Shape: (batch_size, output_dim)
        R_h_T = lrp_linear(h_T, w.t(), b, fc_output, R_output, bias_nb_units=hidden_dim, eps=eps, bias_factor=bias_factor)
        R_h[-1] = R_h_T  # Assign relevance to the last hidden state

        # Initialize relevance for inputs
        R_x = []

        # Backward relevance propagation through time
        for t in reversed(range(seq_len)):
            gates_t = self.gates[t]
            i_t = gates_t['i']
            f_t = gates_t['f']
            g_t = gates_t['g']
            o_t = gates_t['o']
            c_t = self.cells[t + 1]
            c_prev = self.cells[t]
            h_prev = self.hiddens[t]

            # Relevance from the output gate to the cell state
            R_c_t = R_h[t + 1] * o_t * (1 - torch.tanh(c_t) ** 2)
            R_c[t + 1] += R_c_t

            # Relevance through the cell state
            R_c_t_total = R_c[t + 1]

            # Relevance to input gate and candidate gate
            z_i = i_t * g_t
            z_f = f_t * c_prev
            denom = z_i + z_f + eps

            # Split relevance proportionally
            R_i = R_c_t_total * (z_i / denom)
            R_f = R_c_t_total * (z_f / denom)

            # Relevance to candidate gate g_t
            #R_g = R_i  # Since g_t is multiplied by i_t

            R_it_gt = R_i
            abs_i_t = i_t.abs()
            abs_g_t = g_t.abs()
            sum_abs = abs_i_t + abs_g_t + eps  # eps to prevent division by zero

            R_i = R_it_gt * (abs_i_t / sum_abs)
            R_g = R_it_gt * (abs_g_t / sum_abs)

            # Relevance to input x_t and previous hidden state h_prev
            # For g_t
            # Prepare inputs for lrp_linear
            hin = torch.cat([h_prev, self.x[:, t, :]], dim=1)  # Shape: (batch_size, hidden_dim + input_dim)
            w_combined = torch.cat([self.W_hh[2 * hidden_dim:3 * hidden_dim, :], self.W_ih[2 * hidden_dim:3 * hidden_dim, :]], dim=1)  # Shape: (hidden_dim, hidden_dim + input_dim)
            b_combined = self.b_hh[2 * hidden_dim:3 * hidden_dim] + self.b_ih[2 * hidden_dim:3 * hidden_dim]  # Shape: (hidden_dim,)

            hout = g_t  # Shape: (batch_size, hidden_dim)
            Rout = R_g  # Shape: (batch_size, hidden_dim)

            # Compute relevance for inputs and previous hidden state
            Rin_g = lrp_linear(hin, w_combined.t(), b_combined, hout, Rout, bias_nb_units=hidden_dim + input_dim, eps=eps, bias_factor=bias_factor)
            R_h[t] += Rin_g[:, :hidden_dim]
            R_x_t = Rin_g[:, hidden_dim:]

            # Similarly compute Rin for f_t and c_prev if needed
            # For simplicity, we'll assume that R_f is fully assigned to c_prev
            R_c[t] += R_f

            # Aggregate the relevance for x_t
            R_x.insert(0, R_x_t)  # Prepend to maintain time order

        # Stack relevance scores for inputs
        R_input = torch.stack(R_x, dim=1)  # Shape: (batch_size, seq_len, input_dim)

        return R_input


# Training, testing, calibration

input_dim = X_train.shape[2]
hidden_dim = 128
output_dim = 1
num_layers = 1
dropout = 0.5
num_epochs = 100
vis_prefix = 'test'

model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
class_weights = torch.tensor([1.0, np.sum(y_train == 0) / np.sum(y_train == 1)], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return accuracy, precision, recall, specificity, f1

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
        train_correct += (torch.sigmoid(outputs).squeeze().round() == y_batch).sum().item()
        train_total += y_batch.size(0)
    
    train_losses.append(train_loss / train_total)
    train_accuracies.append(train_correct / train_total)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            val_loss += loss.item() * X_batch.size(0)
            val_correct += (torch.sigmoid(outputs).squeeze().round() == y_batch).sum().item()
            val_total += y_batch.size(0)
    
    val_losses.append(val_loss / val_total)
    val_accuracies.append(val_correct / val_total)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, '
          f'Train Accuracy: {train_accuracies[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}')

# Final evaluation and calibration
model.eval()
test_loss, test_correct, test_total = 0.0, 0, 0
all_test_outputs, all_test_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        test_loss += criterion(outputs.squeeze(), y_batch).item() * X_batch.size(0)
        test_correct += (torch.sigmoid(outputs).squeeze().round() == y_batch).sum().item()
        test_total += y_batch.size(0)
        all_test_outputs.append(outputs.cpu())
        all_test_targets.append(y_batch.cpu())

all_test_outputs = torch.cat(all_test_outputs)
all_test_targets = torch.cat(all_test_targets)
y_test_np = all_test_targets.numpy()

# Calibration
val_probs, val_targets = [], []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        outputs = model(X_batch)
        probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
        val_probs.extend(probs)
        val_targets.extend(y_batch.cpu().numpy())

val_probs = np.array(val_probs)
val_targets = np.array(val_targets)
calibrator = LogisticRegression()
calibrator.fit(val_probs.reshape(-1, 1), val_targets)

joblib.dump(calibrator, f"{vis_prefix}_Calibrator.pkl")

end_time = time.time()
runtime = end_time - start_time
print(runtime)

test_preds = torch.sigmoid(all_test_outputs).squeeze().detach().numpy()
test_preds_calibrated = calibrator.predict_proba(test_preds.reshape(-1, 1))[:, 1]

test_preds_binary = (test_preds_calibrated > 0.15).astype(int)

test_accuracy, test_precision, test_recall, test_specificity, test_f1 = calculate_metrics(y_test_np, test_preds_binary)
auroc = roc_auc_score(y_test_np, test_preds_calibrated)
auprc = average_precision_score(y_test_np, test_preds_calibrated)

tn, fp, fn, tp = confusion_matrix(y_test_np, test_preds_binary).ravel()
npv = tn / (tn + fn)

print(f'Test AUROC: {auroc:.4f}')
print(f'Test AUPRC: {auprc:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision: {test_precision:.4f}')
print(f'Test Recall: {test_recall:.4f}')
print(f'Test Specificity: {test_specificity:.4f}')
print(f'Test F1: {test_f1:.4f}')
print(f'Test NPV: {npv:.4f}')

# AUROC, AUPRC visualizations
# Calculate the ROC curve points and the AUC
fpr, tpr, thresholds = roc_curve(y_test_np, test_preds_calibrated)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.savefig(f'{vis_prefix}_Auroc.png')

# Calculate the Precision-Recall curve points and AUC
precision, recall, thresholds = precision_recall_curve(y_test_np, test_preds_calibrated)
prc_auc = auc(recall, precision)

# Plot the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {prc_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.savefig(f'{vis_prefix}_Auprc.png')

# Training and validation loss plot
plt.figure(figsize=(10, 8))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(f'{vis_prefix}_Loss.png')

# Calibration plot
prob_true, prob_pred = calibration_curve(y_test_np, test_preds_calibrated, n_bins=100)
reg = LinearRegression().fit(prob_pred.reshape(-1, 1), prob_true)
slope = reg.coef_[0]
intercept = reg.intercept_
r2 = reg.score(prob_pred.reshape(-1, 1), prob_true)

plt.figure(figsize=(10, 8))
plt.scatter(prob_pred, prob_true, label="Calibration", edgecolors='k', alpha=0.6, s=30)
plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
plt.plot(prob_pred, reg.predict(prob_pred.reshape(-1, 1)), 'r-', label=f"Linear fit: y={intercept:.2f} + {slope:.2f}x, R²={r2:.2f}")
plt.xlabel("Predicted risk")
plt.ylabel("Observed risk")
plt.title("Calibration Plot")
plt.legend()
plt.savefig(f'{vis_prefix}_Calibration.png')

# Feature importance
model.eval()

# Collect all inputs and outputs
all_inputs = []
all_outputs = []
all_R_inputs = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        probs = torch.sigmoid(logits).squeeze()

        # For samples where the prediction is positive
        R_output = probs  # Shape: (batch_size,)

        # Compute relevance scores
        R_input = model.lrp(R_output.unsqueeze(1))  # Shape: (batch_size, seq_len, input_dim)

        all_inputs.append(X_batch.cpu())
        all_outputs.append(y_batch.cpu())
        all_R_inputs.append(R_input.cpu())

# Concatenate all relevance scores
all_R_inputs = torch.cat(all_R_inputs, dim=0)  # Shape: (total_samples, seq_len, input_dim)

# Compute absolute relevance scores for each sample
all_R_inputs_abs = torch.abs(all_R_inputs)  # Shape: (total_samples, seq_len, input_dim)

# Compute average relevance over all samples
average_relevance = all_R_inputs_abs.mean(dim=0)  # Shape: (seq_len, input_dim)

# Sum over time steps to get overall feature relevance
overall_feature_relevance = average_relevance.sum()  # Shape: (input_dim,)

normalized_relevance = average_relevance / overall_feature_relevance

plt.figure(figsize=(12, 10))
sns.heatmap(
    normalized_relevance.numpy().T,
    cmap='Blues',
    annot=False,
    yticklabels=feature_names
)
plt.title('Feature Importance using LRP')
plt.xlabel('Time Epoch')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig(f'{vis_prefix}_Feature_importance.png')

# Normalize relevance scores by total relevance
total_relevance = overall_feature_relevance.sum()
normalized_overall_relevance = overall_feature_relevance / total_relevance

# Sum over time steps to get overall feature relevance
feature_relevance = average_relevance.sum(dim=0)

# Plot bar chart of normalized overall feature importance
sorted_indices = np.argsort(feature_relevance.cpu().numpy())[::-1]
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_relevance = feature_relevance.cpu().numpy()[sorted_indices]

plt.figure(figsize=(12, 12))
plt.barh(sorted_features, sorted_relevance)
plt.xlabel('Normalized Relevance Score')
plt.title('Overall Feature Importance (LRP) - Sorted')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(f'{vis_prefix}_Overall_feature_importance.png')

# Save the model and optimizer state dictionaries
torch.save(model.state_dict(), f'{vis_prefix}_Model.pth')
torch.save(optimizer.state_dict(), f'{vis_prefix}_Optimizer.pth')
print("Model and optimizer have been saved.")