# Scripts Directory

This directory contains the main executable scripts.

## Quick Start: Run Entire Pipeline

To run the complete pipeline from start to finish, use the main pipeline runner:

```bash
# From project root
python run_pipeline.py --all
```

For more options, see the [Pipeline Runner](#pipeline-runner) section below.

## Individual Scripts (Run in Order)

For manual execution, run scripts in numerical order:

1. **Data Preparation**
   ```bash
   python scripts/01_prepare_data.py
   ```

2. **Feature Extraction**
   ```bash
   python scripts/02_extract_features.py
   ```

3. **Model Training**
   ```bash
   # Random Forest
   python scripts/03_train_classical.py
   
   # XGBoost
   python scripts/03_train_xgboost.py
   
   # Neural Networks (PyTorch)
   python scripts/04_train_neural.py
   ```

4. **Evaluation** (optional)
   ```bash
   python scripts/05_evaluate.py
   ```

5. **Biomarker Identification**
   ```bash
   python scripts/06_identify_biomarkers.py
   ```

6. **Raw Audio Features** (optional, if using custom audio)
   ```bash
   python scripts/07_extract_raw_audio_features.py [audio_directory]
   ```

7. **Temporal Stability Analysis**
   ```bash
   python scripts/08_temporal_stability.py
   ```

8. **Cross-Lingual Analysis** (optional, requires LMIC data)
   ```bash
   python scripts/09_cross_lingual_analysis.py
   ```

## Pipeline Runner

The `run_pipeline.py` script (in project root) orchestrates all pipeline steps.

### Basic Usage

```bash
# Run everything
python run_pipeline.py --all

# Check what's already completed (without running)
python run_pipeline.py --check-only

# Skip data download (if already done)
python run_pipeline.py --all --skip-download
```

### Run Specific Steps

```bash
# Run only data preparation
python run_pipeline.py --step prepare

# Run only feature extraction
python run_pipeline.py --step features

# Run only model training
python run_pipeline.py --step training

# Run only biomarker analysis
python run_pipeline.py --step biomarkers

# Run only temporal stability
python run_pipeline.py --step temporal

# Run only cross-lingual analysis
python run_pipeline.py --step crosslingual
```

### Select Models to Train

```bash
# Train only Random Forest
python run_pipeline.py --all --models random_forest

# Train Random Forest and XGBoost
python run_pipeline.py --all --models random_forest xgboost

# Train all models (default)
python run_pipeline.py --all --models random_forest xgboost neural
```

### Skip Optional Steps

```bash
# Skip temporal stability analysis
python run_pipeline.py --all --skip-temporal

# Skip cross-lingual analysis
python run_pipeline.py --all --skip-crosslingual

# Skip both optional analyses
python run_pipeline.py --all --skip-temporal --skip-crosslingual
```

### Full Pipeline Example

```bash
# Complete pipeline with all models, skipping optional steps
python run_pipeline.py --all \
    --skip-download \
    --models random_forest xgboost neural \
    --skip-temporal \
    --skip-crosslingual
```

## Script Descriptions

- **01_prepare_data.py**: Builds metadata mapping participant IDs to feature files
- **02_extract_features.py**: Aggregates frame-level features to speaker-level statistics
- **03_train_classical.py**: Trains Random Forest model
- **03_train_xgboost.py**: Trains XGBoost model
- **04_train_neural.py**: Trains PyTorch neural network models (DNN, CNN-LSTM, LSTM)
- **05_evaluate.py**: Model evaluation utilities
- **06_identify_biomarkers.py**: SHAP analysis and biomarker identification
- **07_extract_raw_audio_features.py**: Extract features from raw audio files (optional)
- **08_temporal_stability.py**: Temporal stability analysis across sessions
- **09_cross_lingual_analysis.py**: Cross-lingual adaptation framework (requires LMIC data)
