# Speech2Health Implementation Guide

## Implementation Roadmap

This document provides a detailed implementation guide for the Speech2Health project, focusing on stress and fatigue detection from speech prosody.

## Phase 1: Data Pipeline & Feature Extraction

### 1.1 Audio Preprocessing

**Objective**: Prepare audio for feature extraction

**Tasks**:
- [ ] Audio conversion (`src/data/preprocessing.py`)
  - Mono conversion, 16 kHz sampling rate
  - Volume normalization
  - Quality checks

- [ ] Voice Activity Detection (VAD) (`src/data/vad.py`)
  - Energy-based VAD
  - WebRTC VAD integration
  - Silence removal
  - Segment boundary detection

- [ ] Segmentation (`src/data/segmentation.py`)
  - Fixed-length segmentation (3-5 seconds)
  - Sliding window option
  - Overlap handling
  - Padding for short segments

**Key Files**:
- `src/data/preprocessing.py`
- `src/data/vad.py`
- `src/data/segmentation.py`
- `scripts/preprocessing/prepare_data.py`

### 1.2 Acoustic-Prosodic Feature Extraction

**Objective**: Extract comprehensive features for stress/fatigue detection

**Tasks**:
- [ ] Pitch (F0) extraction (`src/feature_extraction/pitch.py`)
  - Fundamental frequency estimation
  - Pitch contour analysis
  - Pitch statistics (mean, std, range, jitter)

- [ ] Energy and Intensity (`src/feature_extraction/energy.py`)
  - RMS energy
  - Intensity contours
  - Energy statistics

- [ ] Voice Quality Features (`src/feature_extraction/voice_quality.py`)
  - Jitter (pitch period variation)
  - Shimmer (amplitude variation)
  - Harmonics-to-Noise Ratio (HNR)
  - Glottal-to-Noise Excitation (GNE)

- [ ] Spectral Features (`src/feature_extraction/spectral.py`)
  - MFCCs (13 coefficients + deltas)
  - Log-Mel spectrograms
  - Spectral centroid, rolloff, bandwidth

- [ ] Formant Features (`src/feature_extraction/formants.py`)
  - Formant frequencies (F1, F2, F3)
  - Formant bandwidths
  - Formant trajectories

- [ ] Prosodic Features (`src/feature_extraction/prosody.py`)
  - Speaking rate
  - Pause duration and frequency
  - Rhythm patterns
  - Stress patterns

- [ ] Feature Aggregation (`src/feature_extraction/aggregation.py`)
  - Statistical aggregation (mean, std, min, max, median)
  - Temporal features (slope, curvature)
  - Feature normalization

**Key Files**:
- `src/feature_extraction/pitch.py`
- `src/feature_extraction/energy.py`
- `src/feature_extraction/voice_quality.py`
- `src/feature_extraction/spectral.py`
- `src/feature_extraction/formants.py`
- `src/feature_extraction/prosody.py`
- `src/feature_extraction/aggregation.py`
- `scripts/feature_extraction/extract_features.py`

### 1.3 Feature Storage & Management

**Objective**: Efficient feature storage and retrieval

**Tasks**:
- [ ] Feature storage (`src/data/feature_storage.py`)
  - CSV format for classical ML
  - NumPy arrays for deep learning
  - HDF5 for large datasets
  - Metadata preservation

- [ ] Feature validation (`src/data/feature_validation.py`)
  - Missing value detection
  - Outlier detection
  - Feature distribution checks

**Key Files**:
- `src/data/feature_storage.py`
- `src/data/feature_validation.py`

## Phase 2: Classical Machine Learning Models

### 2.1 Model Implementation

**Objective**: Implement classical ML models for stress/fatigue detection

**Tasks**:
- [ ] Random Forest (`src/models/random_forest.py`)
  - Sklearn implementation
  - Hyperparameter tuning
  - Feature importance extraction

- [ ] XGBoost (`src/models/xgboost.py`)
  - Gradient boosting implementation
  - Hyperparameter optimization
  - Feature importance

- [ ] LightGBM (`src/models/lightgbm.py`)
  - Fast gradient boosting
  - Hyperparameter tuning

- [ ] Support Vector Machine (`src/models/svm.py`)
  - Linear and RBF kernels
  - Hyperparameter optimization

- [ ] Logistic Regression (`src/models/logistic_regression.py`)
  - Baseline model
  - Regularization options

**Key Files**:
- `src/models/random_forest.py`
- `src/models/xgboost.py`
- `src/models/lightgbm.py`
- `src/models/svm.py`
- `src/models/logistic_regression.py`
- `scripts/training/train_classical.py`

### 2.2 Feature Selection

**Objective**: Identify most important features

**Tasks**:
- [ ] Univariate feature selection (`src/models/feature_selection.py`)
  - Chi-square test
  - Mutual information
  - F-test

- [ ] Recursive feature elimination
  - RFE with cross-validation
  - Feature ranking

- [ ] Feature importance analysis
  - Tree-based importance
  - Permutation importance

**Key Files**:
- `src/models/feature_selection.py`
- `notebooks/feature_analysis/feature_importance.ipynb`

### 2.3 Model Training & Evaluation

**Objective**: Train and evaluate classical ML models

**Tasks**:
- [ ] Training pipeline (`src/training/classical_trainer.py`)
  - Cross-validation
  - Hyperparameter tuning (GridSearch, RandomizedSearch)
  - Model persistence

- [ ] Evaluation metrics (`src/evaluation/metrics.py`)
  - Accuracy, precision, recall, F1
  - ROC-AUC, PR-AUC
  - Confusion matrices
  - Per-class metrics

- [ ] Cross-validation (`src/evaluation/cross_validation.py`)
  - K-fold CV
  - Stratified K-fold
  - Leave-one-speaker-out (important for generalization)

**Key Files**:
- `src/training/classical_trainer.py`
- `src/evaluation/metrics.py`
- `src/evaluation/cross_validation.py`
- `scripts/training/train_classical.py`

## Phase 3: Neural Network Models

### 3.1 Neural Architecture Design

**Objective**: Design neural networks for stress/fatigue detection

**Tasks**:
- [ ] CNN-LSTM architecture (`src/models/cnn_lstm.py`)
  - 1D/2D CNN for spectral features
  - LSTM for temporal modeling
  - Attention mechanisms (optional)

- [ ] Transformer-based model (`src/models/transformer.py`)
  - Self-attention for temporal features
  - Lightweight transformer variant

- [ ] Hybrid architectures
  - Multi-modal fusion
  - Feature-level fusion

**Key Files**:
- `src/models/cnn_lstm.py`
- `src/models/transformer.py`
- `src/models/hybrid.py`

### 3.2 Neural Network Training

**Objective**: Train neural networks effectively

**Tasks**:
- [ ] Training pipeline (`src/training/neural_trainer.py`)
  - Data loaders
  - Training loops
  - Validation monitoring
  - Early stopping
  - Learning rate scheduling

- [ ] Loss functions
  - Binary cross-entropy (stress/fatigue)
  - Focal loss (class imbalance)
  - Regression losses (if continuous labels)

- [ ] Regularization
  - Dropout
  - Batch normalization
  - Weight decay

**Key Files**:
- `src/training/neural_trainer.py`
- `scripts/training/train_neural.py`

## Phase 4: Biomarker Identification

### 4.1 Feature Importance Analysis

**Objective**: Identify key vocal biomarkers

**Tasks**:
- [ ] Statistical analysis (`src/interpretability/statistical_analysis.py`)
  - T-tests, Mann-Whitney U tests
  - Effect sizes (Cohen's d)
  - Correlation analysis

- [ ] Model-based importance (`src/interpretability/feature_importance.py`)
  - SHAP values
  - Permutation importance
  - Integrated gradients

- [ ] Biomarker documentation (`docs/biomarker_analysis/biomarkers.md`)
  - Significant features
  - Effect directions
  - Clinical interpretation

**Key Files**:
- `src/interpretability/statistical_analysis.py`
- `src/interpretability/feature_importance.py`
- `notebooks/feature_analysis/biomarker_identification.ipynb`
- `docs/biomarker_analysis/biomarkers.md`

### 4.2 Visualization

**Objective**: Visualize biomarkers and their effects

**Tasks**:
- [ ] Feature distribution plots
  - Stress vs non-stress distributions
  - Fatigue vs non-fatigue distributions

- [ ] Importance plots
  - SHAP summary plots
  - Feature importance rankings

- [ ] Temporal analysis
  - Feature trajectories over time
  - Session-level changes

**Key Files**:
- `notebooks/feature_analysis/visualization.ipynb`
- `src/interpretability/visualization.py`

## Phase 5: Model Comparison & Generalization

### 5.1 Comprehensive Comparison

**Objective**: Compare all models systematically

**Tasks**:
- [ ] Performance comparison (`src/evaluation/comparison.py`)
  - Accuracy, F1, AUC metrics
  - Statistical significance tests
  - Effect sizes

- [ ] Efficiency comparison
  - Training time
  - Inference time
  - Model size

- [ ] Interpretability comparison
  - Feature importance consistency
  - Model explainability scores

**Key Files**:
- `src/evaluation/comparison.py`
- `notebooks/model_comparison/comprehensive_comparison.ipynb`
- `scripts/evaluation/compare_models.py`

### 5.2 Generalization Analysis

**Objective**: Evaluate model generalization

**Tasks**:
- [ ] Cross-speaker evaluation
  - Leave-one-speaker-out CV
  - Speaker-specific performance
  - Speaker clustering analysis

- [ ] Cross-session evaluation
  - Temporal generalization
  - Session drift analysis

- [ ] Dataset generalization
  - Cross-dataset evaluation (if multiple datasets)
  - Domain adaptation analysis

**Key Files**:
- `src/evaluation/generalization.py`
- `notebooks/model_comparison/generalization_analysis.ipynb`

## Phase 6: Interpretability & Explainability

### 6.1 Model Interpretability

**Objective**: Make models interpretable for clinical use

**Tasks**:
- [ ] SHAP analysis (`src/interpretability/shap_analysis.py`)
  - Global importance
  - Local explanations
  - Interaction effects

- [ ] LIME analysis (`src/interpretability/lime_analysis.py`)
  - Local interpretability
  - Feature perturbations

- [ ] Attention visualization (for neural models)
  - Attention weights
  - Temporal attention patterns

**Key Files**:
- `src/interpretability/shap_analysis.py`
- `src/interpretability/lime_analysis.py`
- `notebooks/exploration/interpretability_analysis.ipynb`

## Phase 7: Clinical Validation & Documentation

### 7.1 Clinical Validation

**Objective**: Validate models for clinical relevance

**Tasks**:
- [ ] Statistical validation (`src/evaluation/clinical_validation.py`)
  - Sensitivity, specificity
  - Positive/negative predictive values
  - Clinical thresholds

- [ ] Validation report (`docs/clinical_validation/validation_report.md`)
  - Performance metrics
  - Limitations
  - Recommendations

**Key Files**:
- `src/evaluation/clinical_validation.py`
- `docs/clinical_validation/validation_report.md`

### 7.2 Interactive Demo (Optional)

**Objective**: Create user-friendly demo

**Tasks**:
- [ ] Flask web app (`demos/flask/app.py`)
  - Audio upload
  - Real-time prediction
  - Results visualization

- [ ] Gradio interface (`demos/gradio/app.py`)
  - Simple interface
  - Audio input/output
  - Feature visualization

**Key Files**:
- `demos/flask/app.py`
- `demos/gradio/app.py`

### 7.3 Documentation & Technical Report

**Objective**: Document research and prepare technical report

**Tasks**:
- [ ] Technical report (`docs/report/`)
  - Include comprehensive results plots and visualizations
  - Biomarker analysis and interpretation
  - Model comparison results
  - Clinical validation findings
- [ ] Reproducibility guide
- [ ] API documentation
- [ ] Biomarker documentation

## Technical Implementation Details

### Feature Extraction Specifications

**Pitch (F0)**:
- Method: PyWorld or Parselmouth
- Statistics: mean, std, min, max, range, jitter

**Jitter**:
- Local jitter (cycle-to-cycle variation)
- Relative jitter (normalized)

**Shimmer**:
- Local shimmer (amplitude variation)
- Relative shimmer (normalized)

**Formants**:
- F1, F2, F3 frequencies
- Bandwidths
- Trajectories over time

**MFCCs**:
- 13 coefficients + 13 delta + 13 delta-delta
- Aggregated statistics

### Model Architectures

**CNN-LSTM**:
- Input: Log-Mel spectrogram or MFCCs
- CNN layers: 2-3 conv layers
- LSTM layers: 1-2 LSTM layers
- Output: Binary classification or regression

**Random Forest**:
- n_estimators: 100-500
- max_depth: 10-30
- min_samples_split: 2-10

**XGBoost**:
- n_estimators: 100-500
- max_depth: 3-10
- learning_rate: 0.01-0.1

### Evaluation Protocol

1. **Speaker-level splitting**: 70/15/15 by speaker (not by samples)
2. **Cross-validation**: Leave-one-speaker-out for generalization
3. **Metrics**: Accuracy, F1, AUC, sensitivity, specificity
4. **Statistical tests**: Paired t-tests, Wilcoxon signed-rank tests

## Success Metrics

- **Accuracy**: >75% for stress/fatigue detection
- **AUC**: >0.80
- **Generalization**: <10% drop in cross-speaker performance
- **Biomarkers**: Identify 5-10 significant vocal biomarkers
- **Interpretability**: Generate explainable predictions


