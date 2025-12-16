"""
Neural network training (CNN/LSTM) for depression/stress detection.
Uses aggregated features (not raw audio) for baseline comparison.
Uses PyTorch instead of TensorFlow.
"""
import sys
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
from collections import Counter

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set random seeds for reproducibility
RANDOM_SEED = 42
if TORCH_AVAILABLE:
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path="configs/neural.yaml"):
    """Load configuration."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Return default config
        return {
            'model_type': 'simple_dnn',
            'learning_rate': 0.001,
            'epochs': 50,
            'batch_size': 16,
            'training': {
                'cv': {
                    'n_splits': 5,
                    'shuffle': True,
                    'random_state': 42
                }
            }
        }

def load_features(features_path="data/features/aggregated_features.csv"):
    """Load features (same as classical ML)."""
    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    df = df.dropna(subset=['label'])
    
    X = df.drop(['participant_id', 'label'], axis=1).values
    y = df['label'].values.astype(int)
    participant_ids = df['participant_id'].values
    
    # Handle NaN/Inf
    X = np.where(np.isinf(X), np.nan, X)
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        if np.isnan(col).any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            col = np.where(np.isnan(col), median_val, col)
            X[:, col_idx] = col
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Loaded {len(X)} samples with {X.shape[1]} features")
    
    try:
        metadata_path = "data/processed/metadata.csv"
        if os.path.exists(metadata_path):
            metadata_df = pd.read_csv(metadata_path)
            split_map = dict(zip(metadata_df['participant_id'], metadata_df.get('split', 'unknown')))
            splits = [split_map.get(pid, 'unknown') for pid in participant_ids]
        else:
            splits = None
    except:
        splits = None
    
    return X, y, participant_ids, splits

# PyTorch Dataset class
class FeatureDataset(Dataset):
    """PyTorch Dataset for features."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Neural Network Models
class SimpleDNN(nn.Module):
    """Simple Dense Neural Network with strong regularization."""
    def __init__(self, input_dim, dropout_rate=0.7):
        super(SimpleDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # Reduced from 256
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Increased from 0.5
            
            nn.Linear(128, 64),  # Reduced from 256->128
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(64, 32),  # Reduced from 128->64
            nn.ReLU(),
            nn.Dropout(0.6),  # Increased from 0.4
            
            nn.Linear(32, 1)
            # No sigmoid - will use BCEWithLogitsLoss which applies sigmoid internally
        )
    
    def forward(self, x):
        return self.model(x).squeeze()

class CNNLSTM(nn.Module):
    """1D CNN + LSTM model."""
    def __init__(self, input_dim):
        super(CNNLSTM, self).__init__()
        # Reshape input to (batch, channels, length)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.lstm1 = nn.LSTM(64, 64, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, dropout=0.3)
        
        self.fc1 = nn.Linear(32, 32)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # CNN layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = nn.functional.dropout(x, 0.3, training=self.training)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = nn.functional.dropout(x, 0.3, training=self.training)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = nn.functional.dropout(x, 0.3, training=self.training)
        
        # LSTM expects (batch, seq, features)
        x = x.transpose(1, 2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        
        # Take last output
        x = x[:, -1, :]
        
        # Dense layers (no sigmoid - BCEWithLogitsLoss handles it)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze()

class LSTMModel(nn.Module):
    """LSTM-only model."""
    def __init__(self, input_dim):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(1, 128, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # Reshape: (batch, features) -> (batch, features, 1)
        x = x.unsqueeze(2)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Take last output
        
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # No sigmoid - BCEWithLogitsLoss handles it
        
        return x.squeeze()

def build_model(input_dim, model_type='simple_dnn', dropout_rate=0.7):
    """Build PyTorch model with configurable dropout."""
    if model_type == 'simple_dnn':
        return SimpleDNN(input_dim, dropout_rate=dropout_rate)
    elif model_type == 'cnn_lstm':
        return CNNLSTM(input_dim)
    elif model_type == 'lstm_only':
        return LSTMModel(input_dim)
    else:
        return SimpleDNN(input_dim, dropout_rate=dropout_rate)

def train_model(model, train_loader, val_loader, config, device):
    """Train PyTorch model with strong regularization."""
    # Use class weights if available to handle imbalance
    # Note: BCEWithLogitsLoss supports pos_weight (unlike BCELoss)
    pos_weight = config.get('pos_weight', None)
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor([pos_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Add weight decay (L2 regularization)
    weight_decay = config.get('weight_decay', 0.01)  # Strong L2 regularization
    learning_rate = config.get('learning_rate', 0.0005)  # Lower default LR for stability
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    epochs = config.get('epochs', 50)
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.float().to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    return model

def evaluate_model_pytorch(model, X, y, device):
    """Evaluate PyTorch model."""
    model.eval()
    dataset = FeatureDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_preds.extend(preds)
            all_probs.extend(probs)
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    accuracy = accuracy_score(y, all_preds)
    precision = precision_score(y, all_preds, average='binary', zero_division=0)
    recall = recall_score(y, all_preds, average='binary', zero_division=0)
    f1 = f1_score(y, all_preds, average='binary', zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y, all_probs)
    except:
        roc_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': all_preds,
        'probabilities': all_probs
    }

def train_neural_network(X, y, config, save_model=True):
    """Train neural network with cross-validation."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available. Install with: pip install torch")
    
    print("\n Training Neural Network with PyTorch...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]
    
    # Build model with strong regularization
    model_type = config.get('model_type', 'simple_dnn')
    dropout_rate = config.get('dropout_rate', 0.7)  # Higher dropout for small datasets
    model = build_model(input_dim, model_type, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Calculate class weights for imbalance handling
    from collections import Counter
    class_counts = Counter(y)
    if len(class_counts) == 2:
        pos_weight = class_counts[0] / class_counts[1]  # negative / positive
        config['pos_weight'] = pos_weight
        print(f"  Class weights: {pos_weight:.2f} (pos_weight for positive class)")
    
    print(f"  Model architecture: {model_type}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Weight decay: {config.get('weight_decay', 0.01)}")
    
    # Cross-validation
    cv_config = config['training']['cv']
    cv = StratifiedKFold(
        n_splits=cv_config['n_splits'],
        shuffle=cv_config['shuffle'],
        random_state=cv_config['random_state']
    )
    
    print(f"\n  Performing {cv_config['n_splits']}-fold cross-validation...")
    
    cv_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y), 1):
        print(f"\n  Fold {fold}/{cv_config['n_splits']}...")
        
        X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create datasets and loaders
        train_dataset = FeatureDataset(X_train_fold, y_train_fold)
        val_dataset = FeatureDataset(X_val_fold, y_val_fold)
        
        batch_size = config.get('batch_size', 16)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Build fresh model for each fold with regularization
        fold_model = build_model(input_dim, model_type, dropout_rate=dropout_rate)
        fold_model = fold_model.to(device)
        
        # Train
        fold_model = train_model(fold_model, train_loader, val_loader, config, device)
        
        # Evaluate
        results = evaluate_model_pytorch(fold_model, X_val_fold, y_val_fold, device)
        
        cv_scores['accuracy'].append(results['accuracy'])
        cv_scores['precision'].append(results['precision'])
        cv_scores['recall'].append(results['recall'])
        cv_scores['f1'].append(results['f1'])
        cv_scores['roc_auc'].append(results['roc_auc'])
    
    # Print CV results
    print("\n  Cross-Validation Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        scores = cv_scores[metric]
        print(f"    {metric.upper()}: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    # Train on full dataset
    print("\n  Training on full dataset...")
    full_dataset = FeatureDataset(X_scaled, y)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    # Create validation loader (use 20% for validation during training)
    from sklearn.model_selection import train_test_split
    X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_full_dataset = FeatureDataset(X_train_full, y_train_full)
    val_full_dataset = FeatureDataset(X_val_full, y_val_full)
    train_full_loader = DataLoader(train_full_dataset, batch_size=batch_size, shuffle=True)
    val_full_loader = DataLoader(val_full_dataset, batch_size=batch_size, shuffle=False)
    
    final_model = build_model(input_dim, model_type, dropout_rate=dropout_rate)
    final_model = final_model.to(device)
    final_model = train_model(final_model, train_full_loader, val_full_loader, config, device)
    
    # Save model and scaler
    if save_model:
        os.makedirs("results/models", exist_ok=True)
        model_path = "results/models/neural_model_pytorch.pth"
        scaler_path = "results/models/neural_scaler.pkl"
        
        torch.save(final_model.state_dict(), model_path)
        joblib.dump(scaler, scaler_path)
        print(f"  Model saved to {model_path}")
        print(f"  Scaler saved to {scaler_path}")
    
    # Create wrapper for sklearn-like interface
    class PyTorchWrapper:
        def __init__(self, model, scaler, device, model_type, input_dim):
            self.model = model
            self.scaler = scaler
            self.device = device
            self.model_type = model_type
            self.input_dim = input_dim
        
        def predict(self, X):
            X_scaled = self.scaler.transform(X)
            results = evaluate_model_pytorch(self.model, X_scaled, np.zeros(len(X_scaled)), self.device)
            return results['predictions']
        
        def predict_proba(self, X):
            X_scaled = self.scaler.transform(X)
            results = evaluate_model_pytorch(self.model, X_scaled, np.zeros(len(X_scaled)), self.device)
            probs = results['probabilities']
            return np.column_stack([1 - probs, probs])
    
    wrapped_model = PyTorchWrapper(final_model, scaler, device, model_type, input_dim)
    
    # Create cv_results format
    cv_results = {f'test_{k}': v for k, v in cv_scores.items()}
    
    return wrapped_model, cv_results, scaler

def evaluate_model(model, X, y):
    """Evaluate model (sklearn-like interface)."""
    print("\n Evaluation:")
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary', zero_division=0)
    recall = recall_score(y, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y, y_pred, average='binary', zero_division=0)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y, y_pred_proba)
            print(f"  ROC-AUC: {roc_auc:.4f}")
        except:
            pass
    
    print(f"\n  Confusion Matrix:\n{confusion_matrix(y, y_pred)}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def save_results(results, config):
    """Save results."""
    os.makedirs("results/tables", exist_ok=True)
    
    summary = {
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'cv_mean_accuracy': np.mean(results['cv_results']['test_accuracy']),
        'cv_std_accuracy': np.std(results['cv_results']['test_accuracy']),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = "results/tables/neural_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n Results saved to {summary_path}")

def main():
    """Main neural network training pipeline."""
    print("=" * 70)
    print("Neural Network (CNN/LSTM) Training Pipeline - PyTorch")
    print("=" * 70)
    
    if not TORCH_AVAILABLE:
        print("ERROR: PyTorch not available. Install with: pip install torch")
        return
    
    # Load or create config
    config_path = "configs/neural.yaml"
    config = load_config(config_path)
    
    X, y, participant_ids, splits = load_features()
    
    # Simple train/test split for neural networks
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train
    model, cv_results, scaler = train_neural_network(X_train, y_train, config)
    
    # Evaluate on test set
    test_results = evaluate_model(model, X_test, y_test)
    test_results['cv_results'] = cv_results
    
    # Evaluate on train (for overfitting check)
    train_results = evaluate_model(model, X_train, y_train)
    
    gap = train_results['accuracy'] - test_results['accuracy']
    print(f"\n Overfitting Check: Gap = {gap:.4f}")
    
    save_results(test_results, config)
    
    print("\n" + "=" * 70)
    print(" Neural Network Training Complete!")
    print(f"   CV Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()


