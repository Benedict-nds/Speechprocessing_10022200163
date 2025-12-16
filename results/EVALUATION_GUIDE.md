# Model Evaluation Results & Plots - Location Guide

## Model Performance Tables

### Main Results Files

1. **Random Forest Results**
   - **File**: `results/tables/random_forest_results.csv`
   - **Contains**: CV accuracy, test accuracy, precision, recall, F1, ROC-AUC
   - **Best Performance**: 86.52% CV accuracy, 97.05% ROC-AUC

2. **XGBoost Results**
   - **File**: `results/tables/xgboost_results.csv`
   - **Contains**: CV accuracy, test accuracy, precision, recall, F1

3. **Neural Network Results**
   - **File**: `results/tables/neural_results.csv` (if trained)
   - **Contains**: CV accuracy, test accuracy, precision, recall, F1

4. **Model Comparison Table**
   - **File**: `results/tables/model_comparison.csv`
   - **Contains**: Side-by-side comparison of all models
   - **Note**: Run `python scripts/05_evaluate.py` to generate/update

5. **Performance Comparison (Detailed)**
   - **File**: `results/tables/performance_comparison.csv`
   - **Contains**: Train/val splits with multiple metrics

---

## Model Comparison Plots

### 1. Model Comparison Visualization
- **File**: `results/plots/model_comparison.png`
- **Contents**: 4-panel plot comparing Accuracy, Precision, Recall, F1 across all models
- **Generate**: Run `python scripts/05_evaluate.py`

---

## Interpretability & Biomarker Plots

### Feature Importance

1. **Tree-based Feature Importance**
   - **Plot**: `results/biomarkers/tree_importance.png`
   - **Data**: `results/biomarkers/tree_importance.csv`
   - **Contents**: Top 30 most important features from Random Forest/XGBoost

2. **SHAP Analysis Plots**
   - **Summary Plot**: `results/biomarkers/shap_summary.png`
     - Shows feature impact on model output (beeswarm plot)
   - **Bar Plot**: `results/biomarkers/shap_bar.png`
     - Mean absolute SHAP values (feature importance)
   - **Waterfall Plot**: `results/biomarkers/shap_waterfall_sample.png`
     - Sample explanation for one instance (positive case)
   - **Data**: `results/biomarkers/shap_values.csv` and `shap_importance.csv`
   - **Generate**: Run `python scripts/06_identify_biomarkers.py`

3. **Biomarker Distributions**
   - **Plot**: `results/interpretability/biomarker_distributions.png`
   - **Contents**: Feature distributions by class (depression vs no depression)

---

## 📉 Temporal Stability Analysis

1. **Temporal Stability Results**
   - **File**: `results/biomarkers/temporal_stability.csv`
   - **Contents**: Coefficient of variation and stability scores for all features

2. **Cross-Session Stability** (if available)
   - **File**: `results/biomarkers/cross_session_stability.csv`

3. **Stress Indicator Stability**
   - **File**: `results/biomarkers/stress_indicator_stability.csv`
   - **Contents**: Stability of identified biomarkers

4. **Distribution Plots**
   - **Plot**: `results/plots/temporal_stability_distributions.png`
   - **Generate**: Run `python scripts/08_temporal_stability.py`

---

## 📋 Statistical Analysis

1. **Statistical Tests**
   - **File**: `results/biomarkers/statistical_tests.csv`
   - **Contents**: T-tests, Mann-Whitney U, Cohen's d, p-values
   - **Identifies**: Features with statistically significant differences

2. **Significant Biomarkers**
   - **File**: `results/biomarkers/significant_biomarkers.csv`
   - **Contents**: Combined list from tree importance + SHAP analysis
   - **Top biomarkers**: Features identified by both methods

---

## How to Generate Missing Plots

### 1. Model Comparison Plot
```bash
python scripts/05_evaluate.py
```
Generates: `results/plots/model_comparison.png` and `results/tables/model_comparison.csv`

### 2. SHAP Analysis & Biomarker Plots
```bash
python scripts/06_identify_biomarkers.py
```
Generates:
- `results/biomarkers/shap_summary.png`
- `results/biomarkers/shap_bar.png`
- `results/biomarkers/shap_waterfall_sample.png`
- `results/biomarkers/tree_importance.png`
- Various CSV files

### 3. Temporal Stability Plots
```bash
python scripts/08_temporal_stability.py
```
Generates: `results/plots/temporal_stability_distributions.png`

---

## 📁 Complete File Structure

```
results/
├── tables/                          # Performance metrics (CSV)
│   ├── random_forest_results.csv   # Random Forest CV & test metrics
│   ├── xgboost_results.csv          # XGBoost CV & test metrics
│   ├── neural_results.csv           # Neural Network CV & test metrics
│   ├── model_comparison.csv         # Side-by-side comparison
│   └── performance_comparison.csv   # Detailed train/val splits
│
├── plots/                           # Visualization plots (PNG)
│   ├── model_comparison.png         # 4-panel model comparison
│   └── temporal_stability_distributions.png
│
├── biomarkers/                      # Interpretability results
│   ├── tree_importance.csv          # Feature importance rankings
│   ├── tree_importance.png          # Bar plot of top features
│   ├── shap_values.csv              # Full SHAP values
│   ├── shap_importance.csv          # Mean |SHAP| rankings
│   ├── shap_summary.png             # SHAP summary plot
│   ├── shap_bar.png                 # SHAP importance bar plot
│   ├── shap_waterfall_sample.png    # Sample explanation
│   ├── statistical_tests.csv        # Statistical significance tests
│   ├── significant_biomarkers.csv   # Top identified biomarkers
│   └── temporal_stability.csv       # Feature stability scores
│
├── interpretability/                # Additional interpretability
│   └── biomarker_distributions.png
│
└── models/                          # Trained model files
    ├── random_forest_model.pkl
    ├── xgboost_model.pkl
    └── neural_model_pytorch.pth
```

---

## 🔑 Key Metrics Summary

### Random Forest (Best Performing)
- **CV Accuracy**: 86.52% (±4.92%)
- **CV ROC-AUC**: 97.05% (±2.18%)
- **Test Accuracy**: 55.56% (small test set - high variance)
- **Overfitting**: Mild (5.18% gap)

### XGBoost
- **CV Accuracy**: 66.88% (±8.23%)
- **Test Accuracy**: 55.56%

### Neural Network (PyTorch)
- **CV Accuracy**: 58.44% (±3.98%)
- **Test Accuracy**: 44.44%
- **Overfitting**: Moderate (24.42% gap)

---

## Notes

- **Test set size**: 27 samples (8 positive cases) - too small for reliable evaluation
- **Primary metric**: Use CV accuracy (more reliable with small datasets)
- **Best model**: Random Forest (86.52% CV accuracy)
- **All plots**: High-resolution PNG files (300 DPI) suitable for publication

---

**Last Updated**: Based on latest training runs
