"""
Comprehensive model evaluation and comparison script.
Compares Random Forest, XGBoost, and Neural Network models.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def load_model_results():
    """Load results from all trained models."""
    results_dir = "results/tables"
    models = {}
    
    # Random Forest
    rf_path = os.path.join(results_dir, "random_forest_results.csv")
    if os.path.exists(rf_path):
        models['Random Forest'] = pd.read_csv(rf_path).iloc[0].to_dict()
    
    # XGBoost
    xgb_path = os.path.join(results_dir, "xgboost_results.csv")
    if os.path.exists(xgb_path):
        models['XGBoost'] = pd.read_csv(xgb_path).iloc[0].to_dict()
    
    # Neural Network
    nn_path = os.path.join(results_dir, "neural_results.csv")
    if os.path.exists(nn_path):
        models['Neural Network (PyTorch)'] = pd.read_csv(nn_path).iloc[0].to_dict()
    
    return models

def create_comparison_table(models):
    """Create comparison table of all models."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison_data = []
    
    for model_name, metrics in models.items():
        comparison_data.append({
            'Model': model_name,
            'CV Accuracy': f"{metrics.get('cv_mean_accuracy', 0):.4f} (±{metrics.get('cv_std_accuracy', 0):.4f})",
            'Test Accuracy': f"{metrics.get('accuracy', 0):.4f}",
            'Precision': f"{metrics.get('precision', 0):.4f}",
            'Recall': f"{metrics.get('recall', 0):.4f}",
            'F1-Score': f"{metrics.get('f1', 0):.4f}",
            'ROC-AUC': f"{metrics.get('cv_mean_roc_auc', metrics.get('roc_auc', 0)):.4f}" if 'cv_mean_roc_auc' in metrics or 'roc_auc' in metrics else 'N/A'
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by CV Accuracy (descending)
    if 'CV Accuracy' in df.columns:
        df = df.sort_values('CV Accuracy', ascending=False, key=lambda x: x.str.extract(r'([\d.]+)')[0].astype(float))
    
    print("\n" + df.to_string(index=False))
    
    # Save to file
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/model_comparison.csv", index=False)
    print(f"\n Comparison saved to results/tables/model_comparison.csv")
    
    return df

def plot_model_comparison(models):
    """Create visualization comparing models."""
    if len(models) == 0:
        print("  No model results to plot")
        return
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        model_names = []
        metric_values = []
        
        for model_name, metrics in models.items():
            model_names.append(model_name.replace(' ', '\n'))  # Break long names
            value = metrics.get(metric, 0)
            # Handle CV metrics if available
            if metric == 'accuracy' and 'cv_mean_accuracy' in metrics:
                value = metrics['cv_mean_accuracy']
            metric_values.append(value)
        
        bars = ax.bar(model_names, metric_values, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Plot saved to results/plots/model_comparison.png")

def identify_best_model(models):
    """Identify the best performing model."""
    if len(models) == 0:
        print("  No models to compare")
        return None
    
    # Score each model (higher is better)
    scores = {}
    
    for model_name, metrics in models.items():
        # Use CV accuracy as primary metric
        cv_acc = metrics.get('cv_mean_accuracy', 0)
        test_acc = metrics.get('accuracy', 0)
        f1 = metrics.get('f1', 0)
        
        # Combined score (weighted)
        score = (cv_acc * 0.5) + (test_acc * 0.3) + (f1 * 0.2)
        scores[model_name] = score
    
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    
    print("\n" + "="*80)
    print("BEST MODEL")
    print("="*80)
    print(f"  Best Model: {best_model}")
    print(f"  Score: {best_score:.4f}")
    print(f"  CV Accuracy: {models[best_model].get('cv_mean_accuracy', 0):.4f}")
    print(f"  Test Accuracy: {models[best_model].get('accuracy', 0):.4f}")
    print("="*80)
    
    return best_model

def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("Model Evaluation & Comparison")
    print("="*80)
    
    # Load all model results
    models = load_model_results()
    
    if len(models) == 0:
        print("\n  No model results found!")
        print("  Please train models first:")
        print("    - python scripts/03_train_classical.py (Random Forest)")
        print("    - python scripts/03_train_xgboost.py (XGBoost)")
        print("    - python scripts/04_train_neural.py (Neural Network)")
        return
    
    print(f"\n  Found results for {len(models)} model(s)")
    for model_name in models.keys():
        print(f"    - {model_name}")
    
    # Create comparison
    comparison_df = create_comparison_table(models)
    
    # Plot comparison
    plot_model_comparison(models)
    
    # Identify best model
    best_model = identify_best_model(models)
    
    print("\n" + "="*80)
    print(" Model Comparison Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
