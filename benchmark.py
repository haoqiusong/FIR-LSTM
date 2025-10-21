import math
import json
import time
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
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, average_precision_score)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Data Preprocessing
# =========================

df = pd.read_csv('dataset.csv')

patient_ids = df['Patient'].values

# -----------------------------
# Remaining data preprocessing steps are omitted for brevity.
# -----------------------------

X = df.drop(['Label'], axis=1).values
y = df['Label'].values

unique_patients = np.unique(patient_ids)
train_val_patients, test_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)
# Within the remaining 80%, take 25% as validation (~20% total of all patients)
train_patients, val_patients = train_test_split(train_val_patients, test_size=0.25, random_state=42, stratify=None)
# This yields: 60% train, 20% val, 20% test (all patient-level distinct)

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
X_val_sequence, y_val_sequence = create_sequences(X, y, patient_ids, max_wat_scores, val_patients)
X_test_sequence, y_test_sequence = create_sequences(X, y, patient_ids, max_wat_scores, test_patients)

X_train = X_train_sequence
y_train = y_train_sequence
X_val = X_val_sequence
y_val = y_val_sequence

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

# =========================
# Models
# =========================

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=1, num_layers=1, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]
        h_n = self.batch_norm(h_n)
        h_n = self.dropout_layer(h_n)
        logits = self.fc(h_n)
        return logits

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, output_dim=1, dropout=0.5):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, h_n = self.gru(x)
        h_n = h_n[-1]
        logits = self.fc(h_n)
        return logits

class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_filters=64, kernel_size=3, dropout=0.5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.mean(dim=2)
        logits = self.fc(x)
        return logits

class MLPModel(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim=128, output_dim=1, dropout=0.5):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim*seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        logits = self.fc3(x)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.5, output_dim=1):
        super(TimeSeriesTransformer, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        last_timestep = encoded[-1, :, :]
        logits = self.fc(last_timestep)
        return logits

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu2(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_dim, num_channels=[64,64], kernel_size=3, dropout=0.5, output_dim=1):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=input_dim, num_channels=num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_dim)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.tcn(x)
        y = y.mean(dim=2)
        logits = self.fc(y)
        return logits

def calculate_metrics(y_true, y_pred_binary):
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    specificity = tn / (tn + fp)
    npv = tn / (tn + fn)
    return accuracy, precision, recall, specificity, f1, npv

def train_and_evaluate(model, train_loader, val_loader, test_loader, class_weights, num_epochs=50, lr=1e-4, threshold=0.5, save_prefix='model'):
    start_time = time.time()

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

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
        val_probs, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = torch.sigmoid(outputs).squeeze()
                val_probs.extend(preds.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
                val_correct += (preds.round() == y_batch).sum().item()
                val_total += y_batch.size(0)
        
        val_losses.append(val_loss / val_total)
        val_accuracies.append(val_correct / val_total)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, '
              f'Train Acc: {train_accuracies[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}')

    # Calibrate with validation set
    calibrator = LogisticRegression()
    val_probs = np.array(val_probs)
    val_targets = np.array(val_targets)
    calibrator.fit(val_probs.reshape(-1, 1), val_targets)

    # Test evaluation
    model.eval()
    test_probs, test_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            preds = torch.sigmoid(outputs).squeeze().cpu().numpy()
            test_targets.extend(y_batch.cpu().numpy())
            test_probs.extend(preds)

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)

    # Calibrate test predictions
    test_preds_calibrated = calibrator.predict_proba(test_probs.reshape(-1, 1))[:, 1]
    test_preds_binary = (test_preds_calibrated > threshold).astype(int)

    test_accuracy, test_precision, test_recall, test_specificity, test_f1, test_npv = calculate_metrics(test_targets, test_preds_binary)
    auroc = roc_auc_score(test_targets, test_preds_calibrated)
    auprc = average_precision_score(test_targets, test_preds_calibrated)

    # Save plots
    fpr, tpr, _ = roc_curve(test_targets, test_preds_calibrated)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.savefig(f'{save_prefix}_Auroc.png')
    plt.close()

    precision, recall, _ = precision_recall_curve(test_targets, test_preds_calibrated)
    prc_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {prc_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.savefig(f'{save_prefix}_Auprc.png')
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(f'{save_prefix}_Loss.png')
    plt.close()

    # Calibration plot
    prob_true, prob_pred = calibration_curve(test_targets, test_preds_calibrated, n_bins=10)
    reg = LinearRegression().fit(prob_pred.reshape(-1, 1), prob_true)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    r2 = reg.score(prob_pred.reshape(-1, 1), prob_true)
    plt.figure(figsize=(10, 8))
    plt.scatter(prob_pred, prob_true, label="Calibration", edgecolors='k', alpha=0.6, s=30)
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
    plt.plot(prob_pred, reg.predict(prob_pred.reshape(-1, 1)), 'r-', label=f"y={intercept:.2f}+{slope:.2f}x, R²={r2:.2f}")
    plt.xlabel("Predicted risk")
    plt.ylabel("Observed risk")
    plt.title("Calibration Plot")
    plt.legend()
    plt.savefig(f'{save_prefix}_Calibration.png')
    plt.close()

    # Save model and optimizer
    torch.save(model.state_dict(), f'{save_prefix}_Model.pth')
    torch.save(optimizer.state_dict(), f'{save_prefix}_Optimizer.pth')
    print(f"Model and optimizer states saved as {save_prefix}_Model.pth and {save_prefix}_Optimizer.pth")

    end_time = time.time()
    runtime = end_time - start_time

    return {
        'test_targets': test_targets,
        'test_preds_calibrated': test_preds_calibrated,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Accuracy': test_accuracy,
        'Precision': test_precision,
        'Recall': test_recall,
        'Specificity': test_specificity,
        'F1': test_f1,
        'NPV': test_npv,
        'Runtime': runtime
    }

class_weights = torch.tensor([1.0, np.sum(y_train == 0) / np.sum(y_train == 1)], dtype=torch.float32).to(device)

models_to_test = {
    'LSTM': LSTMModel(input_dim=num_features, hidden_dim=128, output_dim=1, num_layers=1, dropout=0.5).to(device),
    'GRU': GRUModel(input_dim=num_features, hidden_dim=128, num_layers=1, output_dim=1, dropout=0.5).to(device),
    'CNN': CNNModel(input_dim=num_features, output_dim=1, num_filters=64, kernel_size=3, dropout=0.5).to(device),
    'MLP': MLPModel(input_dim=num_features, seq_len=seq_len_train, hidden_dim=128, output_dim=1, dropout=0.5).to(device),
    'Transformer': TimeSeriesTransformer(input_dim=num_features, d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.5, output_dim=1).to(device),
    'TCN': TCNModel(input_dim=num_features, num_channels=[64,64], kernel_size=3, dropout=0.5, output_dim=1).to(device)
}

results = {}

# Train and evaluate each model
for model_name, model_obj in models_to_test.items():
    print(f"===== Training and Evaluating: {model_name} =====")
    res = train_and_evaluate(model_obj, train_loader, val_loader, test_loader, class_weights, num_epochs=500, lr=1e-4, threshold=0.15, save_prefix=model_name)
    results[model_name] = res
    print()

# After all models are evaluated, find thresholds for ~0.8 sensitivity and store in CSV
target_recall = 0.8
final_sensitivity_results = []
for model_name, res in results.items():
    test_targets = res['test_targets']
    test_preds_calibrated = res['test_preds_calibrated']
    runtime = res['Runtime']  # Extract runtime

    thresholds = np.linspace(0, 1, 1001)
    best_threshold = None
    best_diff = float('inf')
    for thr in thresholds:
        y_pred_binary = (test_preds_calibrated >= thr).astype(int)
        current_recall = recall_score(test_targets, y_pred_binary)
        diff = abs(current_recall - target_recall)
        if diff < best_diff:
            best_diff = diff
            best_threshold = thr

    y_pred_binary = (test_preds_calibrated >= best_threshold).astype(int)
    accuracy, precision, recall, specificity, f1, npv = calculate_metrics(test_targets, y_pred_binary)
    auroc = roc_auc_score(test_targets, test_preds_calibrated)
    auprc = average_precision_score(test_targets, test_preds_calibrated)

    final_sensitivity_results.append({
        'Model': model_name,
        'Threshold': best_threshold,
        'Recall': recall,
        'Precision': precision,
        'Accuracy': accuracy,
        'Specificity': specificity,
        'F1': f1,
        'NPV': npv,
        'AUROC': auroc,
        'AUPRC': auprc,
        'Runtime': runtime
    })

df_final = pd.DataFrame(final_sensitivity_results)
df_final.to_csv('sensitivity_0.8_results.csv', index=False)
