# Biomarker Identification Methodology

## Overview

Biomarker identification uses a **multi-method approach** combining:
1. **Tree-based feature importance** (model-driven)
2. **SHAP values** (model interpretability)
3. **Statistical tests** (hypothesis-driven)
4. **Consensus identification** (intersection of methods)

---

## Method 1: Tree-based Feature Importance

### How it works:
- Extracts feature importance scores from trained Random Forest/XGBoost model
- Each tree in the ensemble calculates how much each feature contributes to splits
- Average importance across all trees gives global feature importance

### Implementation:
```python
# From RandomForestClassifier or XGBoostClassifier
importances = model.feature_importances_
```

### Metrics:
- **Importance score**: 0-1 scale (higher = more important)
- **Ranking**: Features sorted by importance (descending)

### Output:
- `results/biomarkers/tree_importance.csv` - All features with importance scores
- `results/biomarkers/tree_importance.png` - Bar plot of top 30 features

### Top Features Identified:
1. `selected_feature_13` (4.79% importance)
2. `selected_feature_43` (3.68%)
3. `selected_feature_31` (3.60%)
4. `selected_feature_12` (3.25%)
5. `selected_feature_62` (3.25%)

---

## Method 2: SHAP (SHapley Additive exPlanations) Analysis

### How it works:
- **SHAP values** quantify the contribution of each feature to each prediction
- Uses game theory (Shapley values) to fairly distribute "credit" among features
- Provides both **global** (overall importance) and **local** (per-instance) explanations

### Implementation:
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# Global importance: mean absolute SHAP value
mean_abs_shap = np.abs(shap_values).mean(axis=0)
```

### Metrics:
- **Mean |SHAP|**: Average absolute SHAP value across all samples (higher = more important)
- **SHAP values**: Can be positive (increases prediction) or negative (decreases prediction)

### Output:
- `results/biomarkers/shap_importance.csv` - Mean |SHAP| for all features
- `results/biomarkers/shap_values.csv` - Full SHAP values matrix
- `results/biomarkers/shap_summary.png` - Beeswarm plot showing feature impact
- `results/biomarkers/shap_bar.png` - Bar plot of mean |SHAP|
- `results/biomarkers/shap_waterfall_sample.png` - Sample explanation

### Top Features Identified:
1. `selected_feature_62` (mean |SHAP|: 0.0316)
2. `selected_feature_79` (0.0220)
3. `selected_feature_72` (0.0213)
4. `selected_feature_43` (0.0186)
5. `selected_feature_13` (0.0155)

---

## Method 3: Statistical Analysis

### How it works:
- Compares feature distributions between classes (depression vs. no depression)
- Uses hypothesis testing to identify statistically significant differences
- Calculates effect sizes to measure practical significance

### Statistical Tests Performed:

1. **T-test (Independent Samples)**
   - Tests if mean feature values differ between classes
   - Assumes normality of distributions
   - Reports: t-statistic, p-value

2. **Mann-Whitney U Test**
   - Non-parametric alternative (doesn't assume normality)
   - Tests if distributions differ (not just means)
   - Reports: U-statistic, p-value

3. **Effect Size (Cohen's d)**
   - Measures magnitude of difference between groups
   - Formula: `d = (mean1 - mean0) / pooled_std`
   - Interpretation:
     - |d| < 0.2: Small effect
     - |d| < 0.5: Medium effect
     - |d| > 0.5: Large effect

### Implementation:
```python
from scipy.stats import ttest_ind, mannwhitneyu

# For each feature:
t_stat, p_value = ttest_ind(feature_class0, feature_class1)
u_stat, p_value_mw = mannwhitneyu(feature_class0, feature_class1)
cohens_d = (mean_class1 - mean_class0) / pooled_std
```

### Output:
- `results/biomarkers/statistical_tests.csv` - All features with test results
- **Significant features**: p-value < 0.05 (for both t-test and Mann-Whitney)

### Statistically Significant Features (p < 0.05):
1. `selected_feature_22` (p=0.0126, Cohen's d=-0.49)
2. `selected_feature_4` (p=0.0198, Cohen's d=0.45)
3. `selected_feature_74` (p=0.0257, Cohen's d=0.42)
4. `selected_feature_46` (p=0.0453, Cohen's d=0.40)

---

## Method 4: Consensus Identification

### How it works:
- Identifies features that appear in **both** tree-based and SHAP top lists
- These consensus features are considered **most reliable biomarkers**
- Intersection method ensures agreement between different interpretability approaches

### Implementation:
```python
# Get top N from each method
tree_top = set(importance_df.head(20)['feature'])
shap_top = set(shap_importance.head(20).index)

# Find common features
common_biomarkers = tree_top.intersection(shap_top)
```

### Output:
- `results/biomarkers/significant_biomarkers.csv` - Combined rankings
- **Common biomarkers**: Features appearing in both top 20 lists

### Consensus Biomarkers (10-11 features):
Features identified by BOTH tree importance AND SHAP:
1. `selected_feature_12`
2. `selected_feature_13`
3. `selected_feature_31`
4. `selected_feature_43`
5. `selected_feature_52`
6. `selected_feature_55`
7. `selected_feature_62`
8. `selected_feature_65`
9. `selected_feature_72`
10. `selected_feature_79`
11. `selected_feature_81`

---

## Complete Pipeline

### Step 1: Load Model and Data
- Load trained Random Forest model
- Load aggregated features
- Handle feature selection (if model was trained with it)

### Step 2: Tree-based Importance
- Extract `feature_importances_` from model
- Sort by importance
- Select top 30 features

### Step 3: SHAP Analysis
- Create SHAP TreeExplainer
- Compute SHAP values for 100 random samples
- Calculate mean absolute SHAP values
- Generate visualizations

### Step 4: Statistical Analysis
- Separate features by class (depression vs. no depression)
- Run t-tests and Mann-Whitney U tests
- Calculate Cohen's d effect sizes
- Identify significant features (p < 0.05)

### Step 5: Consensus Identification
- Compare top features from tree importance and SHAP
- Find intersection (common features)
- Save combined results

---

## Validation & Interpretation

### Multi-Method Validation:
- **Agreement between methods**: Features identified by multiple methods are more reliable
- **Statistical significance**: Ensures differences are not due to chance
- **Effect sizes**: Quantifies practical significance

### Final Biomarker List:
The **consensus biomarkers** (10-11 features appearing in both tree-based and SHAP top 20) are considered the **most reliable indicators** of depression/stress/fatigue.

### Limitations:
- **Feature names**: Currently generic (`selected_feature_X`) due to feature selection
- **Need mapping**: To get actual feature names (e.g., `covarep_mean`, `formant_F1_std`), need to save feature selector during training

---

## Files Generated

1. **Importance Rankings**:
   - `tree_importance.csv` - Tree-based rankings
   - `shap_importance.csv` - SHAP-based rankings

2. **Full Data**:
   - `shap_values.csv` - Complete SHAP values matrix
   - `statistical_tests.csv` - All statistical test results

3. **Visualizations**:
   - `tree_importance.png` - Bar plot
   - `shap_summary.png` - Beeswarm plot
   - `shap_bar.png` - SHAP importance bar plot
   - `shap_waterfall_sample.png` - Sample explanation

4. **Final Results**:
   - `significant_biomarkers.csv` - Combined consensus list

---

## Usage

Run the complete biomarker identification pipeline:
```bash
python scripts/06_identify_biomarkers.py
```

This executes all four methods and generates all outputs listed above.

---

**Methodology**: Multi-method consensus approach combining model-driven (tree importance, SHAP) and hypothesis-driven (statistical tests) techniques for robust biomarker identification.
