"""
XGBoost training script for depression/stress detection.
Similar structure to Random Forest but uses XGBoost.
"""
import sys
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
import joblib
from collections import Counter

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Error: xgboost not available. Install with: pip install xgboost")

# Try to import SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Note: imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import shared functions - define inline since they're not in a separate module
# (These functions are duplicated from 03_train_classical.py for independence)
def load_config(config_path="configs/xgboost.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_features(features_path="data/features/aggregated_features.csv"):
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

def split_train_test(X, y, participant_ids, splits, test_split='test', test_size=0.2):
    from sklearn.model_selection import train_test_split
    
    if splits is None or len(splits) == 0:
        print("  Warning: No split information. Using 80/20 random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        train_ids = participant_ids[:len(X_train)]
        test_ids = participant_ids[len(X_train):]
        return X_train, X_test, y_train, y_test, train_ids, test_ids
    
    test_mask = np.array([s == test_split for s in splits])
    train_mask = ~test_mask
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_ids, test_ids = participant_ids[train_mask], participant_ids[test_mask]
    
    if len(X_test) == 0:
        print(f"  Warning: No '{test_split}' split found. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        train_ids = participant_ids[:len(X_train)]
        test_ids = participant_ids[len(X_train):]
    
    print(f"\n  Train/Test Split: Train={len(X_train)}, Test={len(X_test)}")
    return X_train, X_test, y_train, y_test, train_ids, test_ids

def apply_smote(X, y, random_state=42, sampling_strategy=0.8, use_smote=True):
    if not SMOTE_AVAILABLE or not use_smote:
        return X, y
    
    print("\n  Applying SMOTE...")
    print(f"  Before: {Counter(y)}")
    k_neighbors = min(3, min(Counter(y).values()) - 1)
    if k_neighbors < 1:
        k_neighbors = 1
    
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors, sampling_strategy=sampling_strategy)
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"  After: {Counter(y_resampled)}")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"  SMOTE failed: {e}. Using original data.")
        return X, y

def apply_feature_selection(X, y, config, aggressive=True, return_selectors=False):
    print("\n Applying feature selection...")
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
    
    variance_threshold_val = 0.05 if aggressive else 0.01
    variance_threshold = VarianceThreshold(threshold=variance_threshold_val)
    X_selected = variance_threshold.fit_transform(X)
    removed_low_var = X.shape[1] - X_selected.shape[1]
    print(f"  Removed {removed_low_var} low-variance features")
    
    corr_threshold = 0.90 if aggressive else 0.95
    df_selected = pd.DataFrame(X_selected)
    corr_matrix = df_selected.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    
    if to_drop:
        X_selected = df_selected.drop(columns=to_drop).values
        columns_to_keep = [col for col in df_selected.columns if col not in to_drop]
        removed_corr = len(to_drop)
        print(f"  Removed {removed_corr} highly correlated features")
    else:
        columns_to_keep = list(df_selected.columns)
    
    print(f"  Final features: {X_selected.shape[1]}")
    
    if return_selectors:
        return X_selected, (variance_threshold, columns_to_keep, corr_threshold)
    return X_selected

def evaluate_model(model, X, y, cv_results=None):
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
        'f1': f1,
        'confusion_matrix': confusion_matrix(y, y_pred)
    }

def train_xgboost(X, y, config, use_smote=False, save_model=True):
    """Train XGBoost model with cross-validation."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available. Install with: pip install xgboost")
    
    print("\n Training XGBoost...")
    
    hyperparams = config['model']['hyperparameters']
    print(f"  Hyperparameters:")
    print(f"    n_estimators: {hyperparams['n_estimators']}")
    print(f"    max_depth: {hyperparams['max_depth']}")
    print(f"    learning_rate: {hyperparams['learning_rate']}")
    
    # Calculate scale_pos_weight for class imbalance
    if hyperparams.get('scale_pos_weight') is None:
        counter = Counter(y)
        scale_pos_weight = counter[0] / counter[1] if counter[1] > 0 else 1.0
        print(f"    scale_pos_weight: {scale_pos_weight:.2f} (auto-calculated)")
    else:
        scale_pos_weight = hyperparams['scale_pos_weight']
    
    # Adjust for small datasets
    max_depth = hyperparams['max_depth']
    n_estimators = hyperparams['n_estimators']
    
    if len(X) < 200:
        max_depth = min(max_depth, 6)
        n_estimators = min(n_estimators, 250)
        print(f"   Regularization applied for small dataset")
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=hyperparams['learning_rate'],
        min_child_weight=hyperparams['min_child_weight'],
        subsample=hyperparams['subsample'],
        colsample_bytree=hyperparams['colsample_bytree'],
        reg_alpha=hyperparams['reg_alpha'],
        reg_lambda=hyperparams['reg_lambda'],
        gamma=hyperparams['gamma'],
        scale_pos_weight=scale_pos_weight,
        random_state=hyperparams['random_state'],
        n_jobs=hyperparams['n_jobs'],
        eval_metric='logloss'
    )
    
    cv_config = config['training']['cv']
    cv = StratifiedKFold(
        n_splits=cv_config['n_splits'],
        shuffle=cv_config['shuffle'],
        random_state=cv_config['random_state']
    )
    
    print(f"\n  Performing {cv_config['n_splits']}-fold stratified cross-validation...")
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='binary', zero_division=0),
        'recall': make_scorer(recall_score, average='binary', zero_division=0),
        'f1': make_scorer(f1_score, average='binary', zero_division=0),
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    print("\n  Cross-Validation Results:")
    for metric_key in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        test_key = f'test_{metric_key}'
        train_key = f'train_{metric_key}'
        
        if test_key in cv_results:
            test_mean = np.mean(cv_results[test_key])
            test_std = np.std(cv_results[test_key])
            
            if train_key in cv_results:
                train_mean = np.mean(cv_results[train_key])
                gap = train_mean - test_mean
                print(f"    {metric_key.upper()}: Train={train_mean:.4f}, Test={test_mean:.4f} (+/-{test_std:.4f}), Gap={gap:.4f}")
            else:
                print(f"    {metric_key.upper()}: {test_mean:.4f} (+/-{test_std:.4f})")
    
    print("\n  Training on full dataset...")
    model.fit(X, y)
    
    if save_model:
        os.makedirs("results/models", exist_ok=True)
        model_path = "results/models/xgboost_model.pkl"
        joblib.dump(model, model_path)
        print(f"  Model saved to {model_path}")
    
    return model, cv_results

def save_results(results, config):
    """Save evaluation results."""
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
    summary_path = "results/tables/xgboost_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n Results saved to {summary_path}")

def main():
    """Main XGBoost training pipeline."""
    print("=" * 70)
    print("XGBoost Training Pipeline")
    print("=" * 70)
    
    if not XGBOOST_AVAILABLE:
        print("ERROR: XGBoost not available. Install with: pip install xgboost")
        return
    
    config = load_config("configs/xgboost.yaml")
    X, y, participant_ids, splits = load_features()
    
    X_train, X_test, y_train, y_test, train_ids, test_ids = split_train_test(
        X, y, participant_ids, splits, test_split='test'
    )
    
    # Feature selection
    aggressive = config.get('feature_selection', {}).get('aggressive', True)
    X_train_selected, (variance_selector, columns_to_keep, _) = apply_feature_selection(
        X_train, y_train, config, aggressive=aggressive, return_selectors=True
    )
    
    # Apply to test set
    X_test_var = variance_selector.transform(X_test)
    df_test = pd.DataFrame(X_test_var, columns=range(X_test_var.shape[1]))
    X_test_selected = df_test[columns_to_keep].values
    
    # SMOTE
    use_smote = config.get('training', {}).get('use_smote', True)
    if use_smote and SMOTE_AVAILABLE:
        X_train_resampled, y_train_resampled = apply_smote(
            X_train_selected, y_train, sampling_strategy=0.8, use_smote=True
        )
    else:
        X_train_resampled, y_train_resampled = X_train_selected, y_train
    
    # Train
    model, cv_results = train_xgboost(
        X_train_resampled, y_train_resampled, config, use_smote=use_smote
    )
    
    # Evaluate
    print("\n" + "="*70)
    print(" Test Set Evaluation")
    print("="*70)
    test_results = evaluate_model(model, X_test_selected, y_test, cv_results)
    test_results['cv_results'] = cv_results
    
    print("\n" + "="*70)
    print(" Training Set Evaluation")
    print("="*70)
    train_results = evaluate_model(model, X_train_selected, y_train, cv_results)
    
    # Overfitting check
    gap = train_results['accuracy'] - test_results['accuracy']
    print(f"\n Overfitting Check: Gap = {gap:.4f}")
    
    save_results(test_results, config)
    
    print("\n" + "=" * 70)
    print(" XGBoost Training Complete!")
    print(f"   CV Accuracy: {np.mean(cv_results['test_accuracy']):.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()


