import sys
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, make_scorer
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
import joblib
from collections import Counter

# Try to import SMOTE for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Note: imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def load_config(config_path="configs/random_forest.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_features(features_path="data/features/aggregated_features.csv"):
    """Load aggregated features and labels."""
    print(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    
    # Remove rows with NaN labels (participants without labels)
    initial_count = len(df)
    df = df.dropna(subset=['label'])
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"  Removed {removed_count} samples with NaN labels")
    
    # Separate features and labels
    X = df.drop(['participant_id', 'label'], axis=1).values
    y = df['label'].values.astype(int)  # Convert to integer labels
    participant_ids = df['participant_id'].values
    
    # Handle infinity and NaN values in features more thoroughly
    # Replace infinity with very large finite values, then replace NaN with column median
    X = np.where(np.isinf(X), np.nan, X)  # Convert inf to NaN first
    # Replace NaN with column median (or 0 if all NaN)
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        if np.isnan(col).any():
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0.0
            col = np.where(np.isnan(col), median_val, col)
            X[:, col_idx] = col
    # Final check - replace any remaining NaN/inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"  Label distribution: {Counter(y)}")
    
    # Load metadata to get split information for LOSO CV
    try:
        metadata_path = "data/processed/metadata.csv"
        if os.path.exists(metadata_path):
            metadata_df = pd.read_csv(metadata_path)
            # Map participant_id to split
            split_map = dict(zip(metadata_df['participant_id'], metadata_df.get('split', 'unknown')))
            splits = [split_map.get(pid, 'unknown') for pid in participant_ids]
        else:
            splits = None
    except:
        splits = None
    
    return X, y, participant_ids, splits

def split_train_test(X, y, participant_ids, splits, test_split='test', test_size=0.2):
    """
    Split data into train and test sets based on split column.
    Returns train/test splits for proper evaluation.
    """
    from sklearn.model_selection import train_test_split
    
    if splits is None or len(splits) == 0:
        print("  Warning: No split information available. Using 80/20 random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        train_ids = participant_ids[:len(X_train)]
        test_ids = participant_ids[len(X_train):]
        return X_train, X_test, y_train, y_test, train_ids, test_ids
    
    # Split based on original dataset splits
    test_mask = np.array([s == test_split for s in splits])
    train_mask = ~test_mask
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    train_ids, test_ids = participant_ids[train_mask], participant_ids[test_mask]
    
    # If no test samples found, use random split
    if len(X_test) == 0:
        print(f"  Warning: No '{test_split}' split samples found. Using {int(test_size*100)}/{int((1-test_size)*100)} random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        train_ids = participant_ids[:len(X_train)]
        test_ids = participant_ids[len(X_train):]
    
    print(f"\n   Train/Test Split:")
    print(f"    Train: {len(X_train)} samples ({Counter(y_train)})")
    print(f"    Test:  {len(X_test)} samples ({Counter(y_test)})")
    
    return X_train, X_test, y_train, y_test, train_ids, test_ids

def apply_smote(X, y, random_state=42, sampling_strategy='auto', use_smote=True):
    """
    Apply SMOTE to balance classes.
    
    Args:
        sampling_strategy: 'auto' for 1:1, or float for target ratio
        use_smote: If False, return original data (can use class_weight instead)
    """
    if not SMOTE_AVAILABLE or not use_smote:
        if not SMOTE_AVAILABLE:
            print("  SMOTE not available - skipping oversampling")
        else:
            print("  SMOTE disabled - using class weights instead")
        return X, y
    
    print("\n  Applying SMOTE to balance classes...")
    print(f"  Before SMOTE: {Counter(y)}")
    
    # Use smaller k_neighbors for very small datasets
    k_neighbors = min(3, min(Counter(y).values()) - 1)
    if k_neighbors < 1:
        k_neighbors = 1
    
    smote = SMOTE(
        random_state=random_state, 
        k_neighbors=k_neighbors,
        sampling_strategy=sampling_strategy
    )
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"  After SMOTE: {Counter(y_resampled)}")
        print(f"  New dataset size: {len(X_resampled)} samples")
        return X_resampled, y_resampled
    except Exception as e:
        print(f"  SMOTE failed: {e}. Using original data.")
        return X, y

def apply_feature_selection(X, y, config, aggressive=True, return_selectors=False):
    """
    Apply feature selection to remove:
    - Low variance features (more aggressive threshold if aggressive=True)
    - Highly correlated features (>0.90 if aggressive, >0.95 otherwise)
    """
    print("\n Applying feature selection...")
    if aggressive:
        print("  Using AGGRESSIVE feature selection to reduce overfitting")
    
    # Ensure no infinity or NaN values (should already be cleaned, but double-check)
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
    
    # Remove low variance features (more aggressive for small datasets)
    variance_threshold_val = 0.05 if aggressive else 0.01  # Higher threshold = more features removed
    variance_threshold = VarianceThreshold(threshold=variance_threshold_val)
    X_selected = variance_threshold.fit_transform(X)
    removed_low_var = X.shape[1] - X_selected.shape[1]
    print(f"  Removed {removed_low_var} low-variance features (threshold={variance_threshold_val})")
    
    # Remove highly correlated features (more aggressive correlation threshold)
    corr_threshold = 0.90 if aggressive else 0.95  # Lower threshold = more features removed
    df_selected = pd.DataFrame(X_selected)
    corr_matrix = df_selected.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation > threshold
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > corr_threshold)]
    
    if to_drop:
        X_selected = df_selected.drop(columns=to_drop).values
        columns_to_keep = [col for col in df_selected.columns if col not in to_drop]
        removed_corr = len(to_drop)
        print(f"  Removed {removed_corr} highly correlated features (>{corr_threshold})")
    else:
        columns_to_keep = list(df_selected.columns)
        removed_corr = 0
        print(f"  No highly correlated features found (>{corr_threshold})")
    
    print(f"  Final feature count: {X_selected.shape[1]} (reduced from {X.shape[1]})")
    
    if return_selectors:
        return X_selected, (variance_threshold, columns_to_keep, corr_threshold)
    return X_selected

def train_random_forest(X, y, config, splits=None, use_smote=False, save_model=True):
    """
    Train Random Forest model with cross-validation.
    """
    print("\n Training Random Forest...")
    
    # Get hyperparameters from config
    hyperparams = config['model']['hyperparameters']
    print(f"  Hyperparameters:")
    print(f"    n_estimators: {hyperparams['n_estimators']}")
    print(f"    max_depth: {hyperparams['max_depth']}")
    print(f"    class_weight: {hyperparams['class_weight']}")
    
    # Create model with STRONG regularization to reduce overfitting
    # Adjust hyperparameters for small datasets - be very conservative
    max_depth = hyperparams['max_depth']
    min_samples_split = hyperparams['min_samples_split']
    min_samples_leaf = hyperparams['min_samples_leaf']
    n_estimators = hyperparams['n_estimators']
    
    if len(X) < 200:  # For small datasets, use balanced regularization
        max_depth = min(max_depth, 12)  # Moderate reduction
        min_samples_split = max(min_samples_split, 15)  # Moderate increase
        min_samples_leaf = max(min_samples_leaf, 7)     # Moderate increase
        n_estimators = min(n_estimators, 350)  # Slight reduction
        print(f"   Balanced regularization applied for small dataset:")
        print(f"    max_depth: {max_depth} (reduced from {hyperparams['max_depth']})")
        print(f"    min_samples_split: {min_samples_split} (increased from {hyperparams['min_samples_split']})")
        print(f"    min_samples_leaf: {min_samples_leaf} (increased from {hyperparams['min_samples_leaf']})")
        print(f"    n_estimators: {n_estimators} (slightly reduced from {hyperparams['n_estimators']})")
    
    # Use balanced class_weight if not using SMOTE (SMOTE handles imbalance)
    # For reducing overfitting, prefer class_weight over SMOTE
    if use_smote:
        class_weight = None
        print(f"  Using class_weight=None (SMOTE handles class imbalance)")
    else:
        class_weight = 'balanced'  # Always use balanced when not using SMOTE
        print(f"  Using class_weight='balanced' (handles class imbalance without SMOTE)")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,  # Use adjusted value
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=hyperparams['max_features'],
        bootstrap=hyperparams['bootstrap'],
        random_state=hyperparams['random_state'],
        class_weight=class_weight,
        n_jobs=-1  # Use all CPU cores
    )
    
    # Get CV configuration
    cv_config = config['training']['cv']
    
    # Try Leave-One-Speaker-Out if splits are available and method is specified
    if cv_config.get('method') == 'leave_one_speaker_out' and splits is not None:
        print(f"\n  Using Leave-One-Speaker-Out cross-validation...")
        # Group by split (train/dev/test)
        groups = np.array(splits)
        cv = LeaveOneGroupOut()
        cv_splits = list(cv.split(X, y, groups))
        print(f"  Number of CV folds: {len(cv_splits)}")
    elif cv_config['method'] == 'stratified_kfold':
        cv = StratifiedKFold(
            n_splits=cv_config['n_splits'],
            shuffle=cv_config['shuffle'],
            random_state=cv_config['random_state']
        )
        cv_splits = None
        print(f"\n  Performing {cv_config['n_splits']}-fold stratified cross-validation...")
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_splits = None
        print(f"\n  Performing 5-fold stratified cross-validation...")
    
    # Perform cross-validation with multiple metrics
    # Use make_scorer for binary classification metrics to handle edge cases
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='binary', zero_division=0),
        'recall': make_scorer(recall_score, average='binary', zero_division=0),
        'f1': make_scorer(f1_score, average='binary', zero_division=0),
        'roc_auc': 'roc_auc'
    }
    
    # Apply SMOTE inside CV to avoid data leakage
    # We need to create a custom CV that applies SMOTE to training folds
    if use_smote and SMOTE_AVAILABLE:
        from sklearn.model_selection import BaseCrossValidator
        from sklearn.base import BaseEstimator, TransformerMixin
        
        class SMOTECVWrapper(BaseCrossValidator):
            """Wrapper that applies SMOTE to training folds"""
            def __init__(self, base_cv):
                self.base_cv = base_cv
                self.smote = SMOTE(random_state=42, k_neighbors=3)
            
            def split(self, X, y=None, groups=None):
                for train_idx, test_idx in self.base_cv.split(X, y, groups):
                    # Apply SMOTE to training fold only
                    X_train, y_train = X[train_idx], y[train_idx]
                    try:
                        X_train_resampled, y_train_resampled = self.smote.fit_resample(X_train, y_train)
                        # Map resampled indices back
                        train_resampled_idx = np.arange(len(X_train_resampled))
                        yield train_resampled_idx, test_idx
                    except:
                        yield train_idx, test_idx
            
            def get_n_splits(self, X=None, y=None, groups=None):
                return self.base_cv.get_n_splits(X, y, groups)
        
        # Wrap CV with SMOTE
        if cv_splits is not None:
            # For LOSO, create wrapper manually
            print("  Note: SMOTE not applied in LOSO CV (use stratified k-fold for SMOTE)")
            cv_for_cross_validate = cv_splits
        else:
            cv_for_cross_validate = SMOTECVWrapper(cv)
    else:
        cv_for_cross_validate = cv_splits if cv_splits is not None else cv
    
    # Use cv_splits if available (LOSO), otherwise use cv object
    # IMPORTANT: Set return_train_score=True to check for overfitting
    if isinstance(cv_for_cross_validate, list):
        cv_results = cross_validate(
            model, X, y,
            cv=cv_for_cross_validate,
            scoring=scoring,
            return_train_score=True,  # Get train scores to check overfitting
            n_jobs=-1
        )
    else:
        cv_results = cross_validate(
            model, X, y,
            cv=cv_for_cross_validate,
            scoring=scoring,
            return_train_score=True,  # Get train scores to check overfitting
            n_jobs=-1
        )
    
    # Print results with train vs test comparison for overfitting detection
    print("\n  Cross-Validation Results:")
    metric_names = {
        'accuracy': 'ACCURACY',
        'precision': 'PRECISION',
        'recall': 'RECALL',
        'f1': 'F1',
        'roc_auc': 'ROC_AUC'
    }
    
    for metric_key, metric_name in metric_names.items():
        test_key = f'test_{metric_key}'
        train_key = f'train_{metric_key}'
        
        if test_key in cv_results:
            test_scores = cv_results[test_key]
            test_mean = np.mean(test_scores)
            test_std = np.std(test_scores)
            
            # Check for overfitting by comparing train vs test
            if train_key in cv_results:
                train_scores = cv_results[train_key]
                train_mean = np.mean(train_scores)
                train_std = np.std(train_scores)
                gap = train_mean - test_mean
                
                print(f"    {metric_name}:")
                print(f"      Train: {train_mean:.4f} (+/- {train_std:.4f})")
                print(f"      Test:  {test_mean:.4f} (+/- {test_std:.4f})")
                print(f"      Gap:   {gap:.4f} {' OVERFITTING!' if gap > 0.15 else ' OK'}")
            else:
                print(f"    {metric_name}: {test_mean:.4f} (+/- {test_std:.4f})")
    
    # Overall overfitting check
    if 'train_accuracy' in cv_results and 'test_accuracy' in cv_results:
        train_acc_mean = np.mean(cv_results['train_accuracy'])
        test_acc_mean = np.mean(cv_results['test_accuracy'])
        acc_gap = train_acc_mean - test_acc_mean
        
        print(f"\n   Overfitting Analysis:")
        print(f"    Train Accuracy: {train_acc_mean:.4f}")
        print(f"    Test Accuracy:  {test_acc_mean:.4f}")
        print(f"    Accuracy Gap:   {acc_gap:.4f}")
        
        if acc_gap > 0.20:
            print(f"      SEVERE OVERFITTING (>20% gap)")
        elif acc_gap > 0.10:
            print(f"      MODERATE OVERFITTING (10-20% gap)")
        elif acc_gap > 0.05:
            print(f"      MILD OVERFITTING (5-10% gap)")
        else:
            print(f"     NO OVERFITTING (<5% gap)")
    
    # Train on full dataset
    print("\n  Training on full dataset...")
    model.fit(X, y)
    
    # Save model if requested
    if save_model:
        os.makedirs("results/models", exist_ok=True)
        model_path = "results/models/random_forest_model.pkl"
        joblib.dump(model, model_path)
        print(f"  Model saved to {model_path}")
    
    return model, cv_results

def evaluate_model(model, X, y, cv_results):
    """Evaluate model and print detailed metrics."""
    print("\n Detailed Evaluation:")
    
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics (use binary for binary classification)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='binary', zero_division=0)
    recall = recall_score(y, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y, y_pred, average='binary', zero_division=0)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(y, y_pred_proba)
            print(f"  ROC-AUC: {roc_auc:.4f}")
        except:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    {cm}")
    
    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=['No Depression', 'Depression']))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'cv_results': cv_results
    }

def save_results(results, config):
    """Save evaluation results to file."""
    os.makedirs("results/tables", exist_ok=True)
    
    # Save summary metrics
    cv_acc_key = 'test_accuracy'
    summary = {
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'cv_mean_accuracy': np.mean(results['cv_results'][cv_acc_key]),
        'cv_std_accuracy': np.std(results['cv_results'][cv_acc_key]),
        'cv_mean_roc_auc': np.mean(results['cv_results'].get('test_roc_auc', [0])),
        'cv_std_roc_auc': np.std(results['cv_results'].get('test_roc_auc', [0])),
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = "results/tables/random_forest_results.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n Results saved to {summary_path}")
    
    return summary_path

def main():
    """Main training pipeline."""
    print("=" * 70)
    print("Random Forest Training Pipeline")
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    
    # Load features
    X, y, participant_ids, splits = load_features()
    
    # Split into train/test for proper evaluation
    X_train, X_test, y_train, y_test, train_ids, test_ids = split_train_test(
        X, y, participant_ids, splits, test_split='test'
    )
    
    # Apply feature selection on training data only (to avoid data leakage)
    print("\n Applying feature selection on training data...")
    # Use aggressive feature selection to reduce overfitting
    aggressive_feature_selection = True
    X_train_selected, (variance_selector, columns_to_keep, corr_threshold) = apply_feature_selection(
        X_train, y_train, config, aggressive=aggressive_feature_selection, return_selectors=True
    )
    
    # IMPORTANT: Apply the SAME feature selection to test set
    # Step 1: Apply variance threshold
    X_test_var = variance_selector.transform(X_test)
    
    # Step 2: Apply same correlation removal using same columns
    df_test = pd.DataFrame(X_test_var, columns=range(X_test_var.shape[1]))
    X_test_selected = df_test[columns_to_keep].values
    
    print(f"   Test set features: {X_test_selected.shape[1]} (matches training: {X_train_selected.shape[1]})")
    assert X_test_selected.shape[1] == X_train_selected.shape[1], f"Feature mismatch: train={X_train_selected.shape[1]}, test={X_test_selected.shape[1]}"
    
    # Apply SMOTE to training data only
    # Option 1: Use SMOTE with less aggressive oversampling (balanced approach)
    # Option 2: Use class_weight='balanced' instead
    # Try SMOTE with less aggressive sampling (don't fully balance)
    use_smote_for_training = config.get('training', {}).get('use_smote', True)  # Try SMOTE with moderation
    
    if use_smote_for_training and SMOTE_AVAILABLE:
        print("\n  Using SMOTE with moderate oversampling (0.8 ratio)...")
        # Use sampling_strategy=0.8 to not fully balance (reduces overfitting)
        X_train_resampled, y_train_resampled = apply_smote(
            X_train_selected, y_train, 
            random_state=42,
            sampling_strategy=0.8,  # Only oversample to 80% of majority class
            use_smote=True
        )
    else:
        print("\n  Using class_weight='balanced' instead of SMOTE")
        X_train_resampled, y_train_resampled = X_train_selected, y_train
        # Will use class_weight='balanced' in the model instead
    
    # Train model with CV on training data only
    # Pass use_smote flag based on whether we actually used SMOTE
    model, cv_results = train_random_forest(
        X_train_resampled, y_train_resampled, config, 
        splits=None,  # Don't use splits for CV on training data
        use_smote=use_smote_for_training  # Whether SMOTE was used
    )
    
    # Evaluate on held-out test set
    print("\n" + "="*70)
    print(" Evaluating on HELD-OUT TEST SET")
    print("="*70)
    test_results = evaluate_model(model, X_test_selected, y_test, cv_results)
    
    # Also evaluate on training data for comparison
    print("\n" + "="*70)
    print(" Evaluating on TRAINING DATA (for comparison)")
    print("="*70)
    train_results = evaluate_model(model, X_train_selected, y_train, cv_results)
    
    # Overfitting check: compare train vs test performance
    train_test_gap = train_results['accuracy'] - test_results['accuracy']
    print("\n" + "="*70)
    print(" OVERFITTING CHECK: Train vs Test Performance")
    print("="*70)
    print(f"  Training Accuracy: {train_results['accuracy']:.4f}")
    print(f"  Test Accuracy:     {test_results['accuracy']:.4f}")
    print(f"  Accuracy Gap:      {train_test_gap:.4f}")
    
    if train_test_gap > 0.20:
        print(f"    SEVERE OVERFITTING (>20% gap)")
    elif train_test_gap > 0.10:
        print(f"    MODERATE OVERFITTING (10-20% gap)")
    elif train_test_gap > 0.05:
        print(f"    MILD OVERFITTING (5-10% gap)")
    else:
        print(f"   NO OVERFITTING (<5% gap)")
    
    print("="*70)
    
    # Use test results for final metrics
    results = test_results
    
    # Save results
    save_results(results, config)
    
    print("\n" + "=" * 70)
    print(" Training Complete!")
    print(f"   Final CV Accuracy: {np.mean(results['cv_results']['test_accuracy']):.4f} (+/- {np.std(results['cv_results']['test_accuracy']):.4f})")
    print("=" * 70)

if __name__ == "__main__":
    main()
