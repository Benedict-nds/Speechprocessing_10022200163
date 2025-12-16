"""
Biomarker identification using SHAP analysis and feature importance.
Identifies which features indicate stress/fatigue/depression.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import yaml

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def load_model_and_features(model_path="results/models/random_forest_model.pkl",
                           features_path="data/features/aggregated_features.csv"):
    """Load trained model and feature data."""
    print("Loading model and features...")
    
    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train a model first using scripts/03_train_classical.py")
    
    model = joblib.load(model_path)
    print(f"   Loaded model from {model_path}")
    
    # Load features
    df = pd.read_csv(features_path)
    df = df.dropna(subset=['label'])
    
    # Get feature names (exclude participant_id and label)
    all_feature_names = [col for col in df.columns if col not in ['participant_id', 'label']]
    
    # Prepare data
    X = df[all_feature_names].values
    y = df['label'].values.astype(int)
    participant_ids = df['participant_id'].values
    
    # Handle NaN/Inf (same as training)
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
    
    # Check if model was trained with feature selection
    # The model's n_features_ attribute tells us how many features it expects
    if hasattr(model, 'n_features_in_'):
        model_n_features = model.n_features_in_
    elif hasattr(model, 'n_features_'):
        model_n_features = model.n_features_
    else:
        # Try to infer from feature_importances_
        if hasattr(model, 'feature_importances_'):
            model_n_features = len(model.feature_importances_)
        else:
            model_n_features = X.shape[1]
    
    print(f"   Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"  Model expects {model_n_features} features")
    
    # If model has fewer features, it was trained with feature selection
    # We need to map feature importances to original feature names
    if model_n_features < X.shape[1]:
        print(f"    Model was trained with feature selection ({model_n_features} features vs {X.shape[1]} original)")
        print(f"  Creating feature names based on model's feature count")
        # Since we don't know which exact features were selected, we'll use generic names
        # For proper analysis, we'd need to save the feature selector during training
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            # Use generic names - user should check training script to map back
            feature_names = [f"selected_feature_{i}" for i in range(model_n_features)]
            print(f"    Warning: Exact feature names unknown. Using generic names.")
            print(f"     For accurate names, re-run training with feature selector saved.")
    else:
        feature_names = all_feature_names
    
    # For SHAP analysis, we need to apply feature selection if model was trained with it
    # But we don't have the selector saved, so we'll skip SHAP or use a workaround
    original_X = X.copy()
    if model_n_features < X.shape[1]:
        # We can't apply exact feature selection without the selector
        # For now, just take first N features (this is approximate!)
        print(f"    Note: Using first {model_n_features} features for analysis")
        print(f"     For accurate SHAP, feature selector should be saved during training")
        X = X[:, :model_n_features]  # Approximate - not perfect!
    
    print(f"  Label distribution: {np.bincount(y)}")
    
    return model, X, y, feature_names, participant_ids, original_X, all_feature_names

def tree_based_feature_importance(model, feature_names, top_n=30):
    """Extract tree-based feature importance from Random Forest or XGBoost."""
    print("\n Tree-based Feature Importance:")
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("  Model does not have feature_importances_ attribute")
        return None
    
    # Ensure feature_names and importances have the same length
    if len(feature_names) != len(importances):
        print(f"    Mismatch: {len(feature_names)} feature names vs {len(importances)} importances")
        print(f"  Using indices for feature names")
        # Use indices if names don't match
        if hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        else:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Ensure lengths match
    min_len = min(len(feature_names), len(importances))
    if len(feature_names) != len(importances):
        print(f"    Adjusting: using {min_len} features to match importances length")
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names[:min_len],
        'importance': importances[:min_len]
    }).sort_values('importance', ascending=False)
    
    # Show top features
    print(f"\n  Top {top_n} Most Important Features:")
    print(importance_df.head(top_n).to_string(index=False))
    
    # Save to file
    os.makedirs("results/biomarkers", exist_ok=True)
    importance_df.to_csv("results/biomarkers/tree_importance.csv", index=False)
    print(f"\n   Saved to results/biomarkers/tree_importance.csv")
    
    # Plot
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances (Tree-based)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/biomarkers/tree_importance.png", dpi=300, bbox_inches='tight')
    print(f"   Saved plot to results/biomarkers/tree_importance.png")
    plt.close()
    
    return importance_df

def shap_analysis(model, X, y, feature_names, sample_size=100):
    """Perform SHAP analysis to identify stress/fatigue indicators."""
    if not SHAP_AVAILABLE:
        print("\n  SHAP not available. Skipping SHAP analysis.")
        print("   Install with: pip install shap")
        return None
    
    print("\n SHAP Analysis:")
    print("   Computing SHAP values... (this may take a few minutes)")
    
    # Sample data if too large (SHAP can be slow)
    if len(X) > sample_size:
        print(f"   Sampling {sample_size} instances from {len(X)} for faster computation...")
        np.random.seed(42)
        sample_idx = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[sample_idx]
        y_sample = y[sample_idx]
    else:
        X_sample = X
        y_sample = y
    
    # Create SHAP explainer
    # For tree-based models, use TreeExplainer (fast and exact)
    if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification - SHAP can return:
        # - List of arrays (one per class)
        # - 3D array (samples, features, classes)
        # - 2D array (if output_type is specified)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class SHAP values
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3:
                # 3D array: (samples, features, classes) - use positive class
                shap_values = shap_values[:, :, 1]  # Second class (positive)
            elif shap_values.ndim == 2:
                # 2D array: already correct shape
                pass
            else:
                raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")
    else:
        # Fallback to KernelExplainer (slower but works for any model)
        print("   Using KernelExplainer (slower)...")
        explainer = shap.KernelExplainer(model.predict_proba, X_sample[:min(50, len(X_sample))])
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    
    # Ensure shap_values is 2D
    if shap_values.ndim != 2:
        raise ValueError(f"Expected 2D SHAP values, got shape: {shap_values.shape}")
    
    # Ensure feature names match
    if len(feature_names) != shap_values.shape[1]:
        print(f"     Mismatch: {len(feature_names)} feature names vs {shap_values.shape[1]} SHAP features")
        print(f"   Adjusting feature names to match SHAP values")
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
    
    # Create DataFrame with feature names
    shap_df = pd.DataFrame(shap_values, columns=feature_names[:shap_values.shape[1]])
    
    # Calculate mean absolute SHAP values (global importance)
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    
    print(f"\n  Top 30 Features by Mean |SHAP| Value:")
    print(mean_abs_shap.head(30).to_string())
    
    # Save SHAP values
    os.makedirs("results/biomarkers", exist_ok=True)
    shap_df.to_csv("results/biomarkers/shap_values.csv", index=False)
    mean_abs_shap.to_frame(name='mean_abs_shap').to_csv("results/biomarkers/shap_importance.csv")
    print(f"\n   Saved SHAP values to results/biomarkers/")
    
    # Create visualizations
    print("\n  Generating SHAP visualizations...")
    
    # 1. Summary plot (most important)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig("results/biomarkers/shap_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     SHAP summary plot saved")
    
    # 2. Bar plot (mean absolute SHAP)
    plt.figure(figsize=(12, 8))
    top_shap = mean_abs_shap.head(30)
    plt.barh(range(len(top_shap)), top_shap.values)
    plt.yticks(range(len(top_shap)), top_shap.index)
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 30 Features by SHAP Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("results/biomarkers/shap_bar.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     SHAP bar plot saved")
    
    # 3. Waterfall plot for a sample instance (showing stress indicators)
    # Find a positive case (depression/stress)
    positive_idx = np.where(y_sample == 1)[0]
    if len(positive_idx) > 0:
        sample_idx = positive_idx[0]
        try:
            plt.figure(figsize=(12, 8))
            # Get expected value for positive class
            if isinstance(explainer.expected_value, (list, np.ndarray)):
                if len(explainer.expected_value) > 1:
                    base_value = explainer.expected_value[1]
                else:
                    base_value = explainer.expected_value[0]
            else:
                base_value = explainer.expected_value
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=base_value,
                    data=X_sample[sample_idx],
                    feature_names=feature_names[:len(shap_values[sample_idx])]
                ),
                show=False,
                max_display=20
            )
            plt.tight_layout()
            plt.savefig("results/biomarkers/shap_waterfall_sample.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"     SHAP waterfall plot (sample positive case) saved")
        except Exception as e:
            print(f"      Could not create waterfall plot: {e}")
            # Skip waterfall plot if it fails
    
    return shap_df, mean_abs_shap

def identify_stress_fatigue_features(importance_df, shap_importance=None, top_n=20):
    """Identify key features indicating stress/fatigue/depression."""
    print("\n Identifying Stress/Fatigue Biomarkers:")
    
    biomarkers = {}
    
    # From tree-based importance
    if importance_df is not None:
        top_tree = importance_df.head(top_n)
        biomarkers['tree_based'] = top_tree['feature'].tolist()
        print(f"\n  Top {top_n} from Tree-based Importance:")
        for i, feat in enumerate(top_tree['feature'].head(10), 1):
            print(f"    {i}. {feat} (importance: {top_tree.loc[top_tree['feature']==feat, 'importance'].values[0]:.4f})")
    
    # From SHAP
    if shap_importance is not None:
        top_shap = shap_importance.head(top_n)
        biomarkers['shap_based'] = top_shap.index.tolist()
        print(f"\n  Top {top_n} from SHAP Analysis:")
        for i, (feat, val) in enumerate(top_shap.head(10).items(), 1):
            print(f"    {i}. {feat} (|SHAP|: {val:.4f})")
    
    # Combined (intersection or union)
    if importance_df is not None and shap_importance is not None:
        tree_set = set(importance_df.head(top_n)['feature'])
        shap_set = set(shap_importance.head(top_n).index)
        common = tree_set.intersection(shap_set)
        biomarkers['common'] = list(common)
        
        print(f"\n  Common Features (appear in both top {top_n}):")
        for i, feat in enumerate(list(common)[:10], 1):
            print(f"    {i}. {feat}")
    
    # Save biomarkers
    biomarkers_df = pd.DataFrame({
        'rank': range(1, top_n + 1),
        'tree_based': biomarkers.get('tree_based', [None] * top_n)[:top_n],
        'shap_based': biomarkers.get('shap_based', [None] * top_n)[:top_n]
    })
    
    os.makedirs("results/biomarkers", exist_ok=True)
    biomarkers_df.to_csv("results/biomarkers/significant_biomarkers.csv", index=False)
    print(f"\n   Saved biomarkers to results/biomarkers/significant_biomarkers.csv")
    
    return biomarkers

def statistical_analysis(X, y, feature_names):
    """Statistical tests to identify significant differences between groups."""
    print("\n Statistical Analysis (T-tests):")
    
    from scipy.stats import ttest_ind, mannwhitneyu
    
    # Separate by class
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    
    results = []
    
    for i, feat_name in enumerate(feature_names):
        feat_class0 = X_class0[:, i]
        feat_class1 = X_class1[:, i]
        
        # T-test (assumes normality)
        try:
            t_stat, p_value = ttest_ind(feat_class0, feat_class1)
            
            # Mann-Whitney U (non-parametric)
            try:
                u_stat, p_value_mw = mannwhitneyu(feat_class0, feat_class1, alternative='two-sided')
            except:
                p_value_mw = np.nan
            
            # Effect size (Cohen's d)
            mean_diff = np.mean(feat_class1) - np.mean(feat_class0)
            pooled_std = np.sqrt((np.var(feat_class0) + np.var(feat_class1)) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            results.append({
                'feature': feat_name,
                'mean_class0': np.mean(feat_class0),
                'mean_class1': np.mean(feat_class1),
                'mean_diff': mean_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'p_value_mannwhitney': p_value_mw,
                'cohens_d': cohens_d,
                'significant_ttest': p_value < 0.05,
                'significant_mw': p_value_mw < 0.05 if not np.isnan(p_value_mw) else False
            })
        except:
            continue
    
    stats_df = pd.DataFrame(results)
    stats_df = stats_df.sort_values('p_value')
    
    # Show significant features
    significant = stats_df[stats_df['significant_ttest']].head(20)
    print(f"\n  Top 20 Statistically Significant Features (p < 0.05):")
    print(significant[['feature', 'mean_class0', 'mean_class1', 'mean_diff', 'p_value', 'cohens_d']].to_string(index=False))
    
    # Save
    os.makedirs("results/biomarkers", exist_ok=True)
    stats_df.to_csv("results/biomarkers/statistical_tests.csv", index=False)
    print(f"\n   Saved to results/biomarkers/statistical_tests.csv")
    
    return stats_df

def main():
    """Main biomarker identification pipeline."""
    print("=" * 70)
    print("Biomarker Identification & Interpretability Analysis")
    print("=" * 70)
    
    # Load model and features
    result = load_model_and_features()
    if len(result) == 7:
        model, X, y, feature_names, participant_ids, original_X, all_feature_names = result
    else:
        # Backward compatibility
        model, X, y, feature_names, participant_ids = result
        original_X = X
        all_feature_names = feature_names
    
    # Apply same feature selection as training (if needed)
    # For now, assume we're using all features or model was trained with all
    
    # 1. Tree-based feature importance
    importance_df = tree_based_feature_importance(model, feature_names)
    
    # 2. SHAP analysis
    shap_df, shap_importance = shap_analysis(model, X, y, feature_names)
    
    # 3. Statistical analysis
    stats_df = statistical_analysis(X, y, feature_names)
    
    # 4. Identify key biomarkers
    biomarkers = identify_stress_fatigue_features(importance_df, shap_importance)
    
    print("\n" + "=" * 70)
    print(" Biomarker Analysis Complete!")
    print("=" * 70)
    print("\nResults saved to:")
    print("  - results/biomarkers/tree_importance.csv")
    if SHAP_AVAILABLE:
        print("  - results/biomarkers/shap_values.csv")
        print("  - results/biomarkers/shap_importance.csv")
    print("  - results/biomarkers/statistical_tests.csv")
    print("  - results/biomarkers/significant_biomarkers.csv")
    print("\nPlots saved to:")
    print("  - results/biomarkers/tree_importance.png")
    if SHAP_AVAILABLE:
        print("  - results/biomarkers/shap_summary.png")
        print("  - results/biomarkers/shap_bar.png")
        print("  - results/biomarkers/shap_waterfall_sample.png")

if __name__ == "__main__":
    main()
