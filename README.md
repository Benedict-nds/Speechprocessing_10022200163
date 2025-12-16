# Speech2Health: Detecting Stress and Fatigue from Speech Prosody

## Project Overview

Speech2Health bridges the gap between speech processing and healthcare applications. Human speech carries rich physiological and emotional cues—including pitch, energy, and vocal stability—that change under stress or fatigue. While most speech emotion recognition models focus on categorical emotions (e.g., anger, joy), Speech2Health focuses on the detection of psychophysiological states such as stress and fatigue from acoustic-prosodic speech features.

This project develops interpretable machine learning models that can detect stress and fatigue non-invasively, enabling applications in mental health monitoring, occupational safety, and well-being assessment. By ensuring transparency, reproducibility, and real-world relevance, Speech2Health contributes to the growing field of audio-based health research.

## Key Innovations

- **Psychophysiological State Detection**: Focus on stress and fatigue rather than categorical emotions
- **Comprehensive Feature Engineering**: Extraction of acoustic-prosodic biomarkers (pitch, jitter, shimmer, formants)
- **Interpretable Models**: Classical ML and neural approaches with explainability
- **Biomarker Identification**: Key vocal indicators of stress and fatigue
- **Clinical Relevance**: Models designed for healthcare and monitoring applications

## Project Goals

1. Develop interpretable models for detecting stress and fatigue from acoustic features
2. Compare classical ML (Random Forest, XGBoost) and neural approaches (CNN-LSTM)
3. Identify key vocal biomarkers associated with stress and fatigue
4. Evaluate model generalization across speakers and sessions
5. Promote reproducibility and explainability in audio-based health research

## Project Structure

```
speech2health/
├── README.md              # This file
├── IMPLEMENTATION.md      # Detailed implementation guide
├── STRUCTURE.md           # Project structure explanation
├── requirements.txt       # Python dependencies
│
├── code/                  # Source code (organized by functionality)
│   ├── data/             # Data loading and preprocessing
│   ├── feature_extraction/ # Acoustic-prosodic feature extraction
│   ├── models/           # Model architectures (ML and neural)
│   ├── training/         # Training utilities
│   ├── evaluation/       # Evaluation metrics
│   └── interpretability/ # Biomarker analysis
│
├── scripts/               # Main executable scripts (run in order)
│   ├── 01_prepare_data.py
│   ├── 02_extract_features.py
│   ├── 03_train_classical.py
│   ├── 04_train_neural.py
│   ├── 05_evaluate.py
│   └── 06_identify_biomarkers.py
│
├── configs/               # Configuration files
│   ├── random_forest.yaml
│   ├── xgboost.yaml
│   └── cnn_lstm.yaml
│
├── data/                  # Dataset storage
│   ├── raw/              # Original datasets
│   ├── processed/        # Preprocessed audio
│   └── features/         # Extracted features
│
├── results/               # All outputs for technical report
│   ├── models/           # Trained model checkpoints
│   ├── plots/            # Visualizations
│   ├── tables/           # Performance tables
│   └── biomarkers/       # Biomarker analysis results
│
├── notebooks/             # Jupyter notebooks for analysis
│   ├── 01_feature_analysis.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_biomarker_visualization.ipynb
│
└── report/                # Technical report materials
    ├── figures/
    ├── tables/
    └── draft/
```

See `STRUCTURE.md` for detailed explanation of the project structure.

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

**Quick Start - Run Everything:**
```bash
# Run the complete pipeline (download → training → analysis)
python run_pipeline.py --all

# Skip data download if already done
python run_pipeline.py --all --skip-download

# Check what's already completed
python run_pipeline.py --check-only
```

**Manual Step-by-Step Execution:**

Alternatively, run scripts individually in order:

```bash
# Step 1: Download data (if needed)
python datadownload.py

# Step 2: Prepare metadata
python scripts/01_prepare_data.py

# Step 3: Extract and aggregate features
python scripts/02_extract_features.py

# Step 4: Train models
python scripts/03_train_classical.py    # Random Forest
python scripts/03_train_xgboost.py      # XGBoost
python scripts/04_train_neural.py       # Neural Networks (PyTorch)

# Step 5: Identify biomarkers (requires trained model)
python scripts/06_identify_biomarkers.py

# Step 6: Temporal stability analysis (optional)
python scripts/08_temporal_stability.py

# Step 7: Cross-lingual analysis (optional, requires LMIC data)
python scripts/09_cross_lingual_analysis.py
```

See `scripts/README.md` for detailed pipeline runner options.

### Using Notebooks

For interactive analysis:

```bash
jupyter notebook notebooks/01_feature_analysis.ipynb
```

### Results

All outputs are saved in the `results/` directory:
- `results/models/` - Trained model checkpoints
- `results/plots/` - Feature importance plots and model comparisons
- `results/tables/` - Performance metrics and biomarker tables
- `results/biomarkers/` - Detailed biomarker analysis

## Datasets

- **DAIC-WOZ**: Clinical interviews for depression/stress detection
- **AVEC 2019 Challenge Dataset**: Multimodal affect recognition
- **SEMAINE Database**: Emotional speech with stress annotations

## Research Contributions

This project contributes to the field through:

1. **Biomarker Discovery**: Identification of vocal indicators of stress and fatigue
2. **Model Comparison**: Comprehensive evaluation of classical ML vs neural approaches
3. **Interpretability**: Explainable models suitable for clinical contexts
4. **Generalization Analysis**: Cross-speaker and cross-session evaluation
5. **Reproducible Research**: Complete workflow for audio-based health research

## Expected Deliverables

- Trained stress/fatigue detection models
- Feature importance and interpretability analyses
- Performance and statistical comparison tables with visualizations
- Technical report (10-15 pages including results plots, biomarker analysis, and model comparisons)
- Reproducibility documentation (scripts, configs, seeds)
- Optional interactive demo (Flask/Gradio)

## Clinical Applications

- **Mental Health Monitoring**: Early detection of stress and fatigue
- **Occupational Safety**: Workplace stress assessment
- **Well-being Assessment**: Personal health tracking
- **Telemedicine**: Remote health monitoring

## Contributing

This is a research project with potential healthcare applications. Contributions that improve interpretability, clinical relevance, or reproducibility are welcome.

## License

[Specify license]

## Acknowledgments

- DAIC-WOZ, AVEC, and SEMAINE dataset creators
- Clinical research collaborators
- Open-source audio processing community

