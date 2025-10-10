import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
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

df = pd.read_csv('dataset.csv')

patient_ids = df['Patient'].values

X = df.drop(['Label', 'Max WAT score'], axis=1).values
y = df['Label'].values

unique_patients = np.unique(patient_ids)
train_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

def create_sequences(X, y, patient_ids, max_wat_scores, selected_patients, time_steps=6):
    Xs, ys = [], []
    for patient in selected_patients:
        patient_indices = np.where(patient_ids == patient)[0]
        patient_X = X[patient_indices]
        patient_y = y[patient_indices]
        patient_max_wat_scores = max_wat_scores[patient_indices]

        if len(patient_X) <= time_steps:
            continue

        for i in range(len(patient_X) - time_steps):
            if patient_max_wat_scores[i + time_steps] != -1:
                Xs.append(patient_X[i:(i + time_steps), :])
                ys.append(patient_y[i + time_steps])
    
    return np.array(Xs), np.array(ys)

X_train_sequence, y_train_sequence = create_sequences(X, y, patient_ids, max_wat_scores, train_patients)
X_test_sequence, y_test_sequence = create_sequences(X, y, patient_ids, max_wat_scores, test_patients)

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
            R_g = R_i  # Since g_t is multiplied by i_t

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

# Load pre-trained model
input_dim = X_train.shape[2]
hidden_dim = 66
output_dim = 1
num_layers = 1
dropout = 0.5

model = LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
class_weights = torch.tensor([1.0, np.sum(y_train == 0) / np.sum(y_train == 1)], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

model.load_state_dict(torch.load('a_Model.pth'))
optimizer.load_state_dict(torch.load('a_Optimizer.pth'))


# Feature importance validation

def compute_model_performance(model, data_loader):
    model.eval()
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            all_outputs.extend(probs)
            all_targets.extend(y_batch.cpu().numpy())
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    auroc = roc_auc_score(all_targets, all_outputs)
    auprc = average_precision_score(all_targets, all_outputs)
    
    # Convert probabilities to binary predictions
    threshold = 0.5
    y_pred = (all_outputs >= threshold).astype(int)
    
    accuracy = accuracy_score(all_targets, y_pred)
    precision = precision_score(all_targets, y_pred)
    recall = recall_score(all_targets, y_pred)  # Sensitivity
    f1 = f1_score(all_targets, y_pred)
    tn, fp, fn, tp = confusion_matrix(all_targets, y_pred).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    
    performance_metrics = {
        'AUROC': auroc,
        'AUPRC': auprc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall (Sensitivity)': recall,
        'Specificity': specificity,
        'F1-score': f1,
        'NPV': npv
    }
    return performance_metrics

# Permute some features to see influence
def permutation_importance(model, test_loader, feature_indices, time_step, n_repeats=5):
    # Compute baseline performance
    baseline_performance = compute_model_performance(model, test_loader)
    print('Baseline Performance:')
    for metric, value in baseline_performance.items():
        print(f'{metric}: {value:.4f}')
    
    importance_scores = {}

    for feature_idx in feature_indices:
        print(f'\nEvaluating feature at index {feature_idx}')
        metrics_list = []
        for repeat in range(n_repeats):
            # Collect permuted Xs and ys
            permuted_Xs = []
            permuted_ys = []
            for X_batch, y_batch in test_loader:
                X_permuted = X_batch.clone()
                # Permute the feature values at the specified time step
                perm = torch.randperm(X_permuted.size(0))
                X_permuted[:, time_step, feature_idx] = X_permuted[perm, time_step, feature_idx]
                permuted_Xs.append(X_permuted)
                permuted_ys.append(y_batch)
            # Concatenate permuted Xs and ys
            permuted_Xs = torch.cat(permuted_Xs, dim=0)
            permuted_ys = torch.cat(permuted_ys, dim=0)
            # Create TensorDataset and DataLoader
            permuted_dataset = TensorDataset(permuted_Xs, permuted_ys)
            permuted_loader = DataLoader(dataset=permuted_dataset, batch_size=test_loader.batch_size, shuffle=False)
            # Compute performance with permuted feature
            performance = compute_model_performance(model, permuted_loader)
            metrics_list.append(performance)
            print(f'Permutation {repeat + 1}: AUROC = {performance["AUROC"]:.4f}')
        
        # Calculate the mean decrease in performance for each metric
        mean_metrics = {metric: np.mean([m[metric] for m in metrics_list]) for metric in baseline_performance.keys()}
        importance = {metric: baseline_performance[metric] - mean_metrics[metric] for metric in baseline_performance.keys()}
        importance_scores[feature_idx] = importance
        print('Mean Performance after permutation:')
        for metric, value in mean_metrics.items():
            print(f'{metric}: {value:.4f}')
        #print('Importance scores:')
        #for metric, value in importance.items():
            #print(f'{metric}: {value:.4f}')
    
    return importance_scores

feature_names = df.drop(['Label', 'Max WAT score'], axis=1).columns.tolist()
# Map feature names to indices
feature_to_index = {feature: idx for idx, feature in enumerate(feature_names)}

features_of_interest = [
    'Previous_Withdrawal_Epochs',
    'Heart Rate (Maximum value)',
    'Temperature (Maximum value)',
    'Duration of methadone'
]

# Get their indices
feature_indices = [feature_to_index[feature] for feature in features_of_interest]
print('Feature Indices:', feature_indices)

importance_scores = permutation_importance(
    model,
    test_loader,
    feature_indices=feature_indices,
    time_step=5,  # 6th epoch (indexing starts from 0)
    n_repeats=5
)

# Display the importance scores for each feature
for feature, idx in zip(features_of_interest, feature_indices):
    print(f'\nFeature: {feature}')
    print('Importance Scores:')
    for metric, value in importance_scores[idx].items():
        print(f'{metric}: {value:.4f}')

# Feature masking (mask some features to 0 to see influence)
def feature_masking(model, test_loader, feature_indices, time_step):
    # Compute baseline performance
    baseline_performance = compute_model_performance(model, test_loader)
    print('Baseline Performance:')
    for metric, value in baseline_performance.items():
        print(f'{metric}: {value:.4f}')
    
    importance_scores = {}

    for feature_idx in feature_indices:
        print(f'\nMasking feature at index {feature_idx}')
        # Collect masked Xs and ys
        masked_Xs = []
        masked_ys = []
        for X_batch, y_batch in test_loader:
            X_masked = X_batch.clone()
            # Mask the feature values at the specified time step
            X_masked[:, time_step, feature_idx] = 0.0  # Or use the mean value if more appropriate
            masked_Xs.append(X_masked)
            masked_ys.append(y_batch)
        # Concatenate masked Xs and ys
        masked_Xs = torch.cat(masked_Xs, dim=0)
        masked_ys = torch.cat(masked_ys, dim=0)
        # Create TensorDataset and DataLoader
        masked_dataset = TensorDataset(masked_Xs, masked_ys)
        masked_loader = DataLoader(dataset=masked_dataset, batch_size=test_loader.batch_size, shuffle=False)
        # Compute performance with masked feature
        performance = compute_model_performance(model, masked_loader)
        importance = {metric: baseline_performance[metric] - performance[metric] for metric in baseline_performance.keys()}
        importance_scores[feature_idx] = importance
        print('Performance after masking:')
        for metric, value in performance.items():
            print(f'{metric}: {value:.4f}')
        print('Importance scores:')
        for metric, value in importance.items():
            print(f'{metric}: {value:.4f}')
    
    return importance_scores

importance_scores_masking = feature_masking(
    model,
    test_loader,
    feature_indices=feature_indices,
    time_step=5  # 6th epoch (indexing starts from 0)
)

# Display the importance scores for each feature
for feature, idx in zip(features_of_interest, feature_indices):
    print(f'\nFeature: {feature}')
    print('Importance Scores:')
    for metric, value in importance_scores_masking[idx].items():
        print(f'{metric}: {value:.4f}')
