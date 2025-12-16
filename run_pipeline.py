#!/usr/bin/env python3
"""
Complete Pipeline Runner for Speech2Health Project

This script orchestrates the entire pipeline from data download to model evaluation
and biomarker identification.

Usage:
    python run_pipeline.py [--skip-download] [--skip-training] [--models MODEL1 MODEL2]
    python run_pipeline.py --step prepare  # Run only data preparation
    python run_pipeline.py --all           # Run everything
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70 + "\n")

def print_step(step_num, total_steps, description):
    """Print step information."""
    print(f"\n[{step_num}/{total_steps}] {description}")
    print("-" * 70)

def run_script(script_path, description, required=True):
    """Run a Python script and handle errors."""
    script_path = project_root / script_path
    
    if not script_path.exists():
        if required:
            print(f"ERROR: Required script not found: {script_path}")
            return False
        else:
            print(f"SKIP: Optional script not found: {script_path}")
            return True
    
    print(f"Running: {description}")
    print(f"Command: python {script_path}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(project_root),
            check=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {description} failed with error: {e}")
        return False

def check_file_exists(file_path, description):
    """Check if a required file exists."""
    file_path = project_root / file_path
    exists = file_path.exists()
    if exists:
        print(f"✓ {description} exists")
    else:
        print(f"✗ {description} not found: {file_path}")
    return exists

def main():
    parser = argparse.ArgumentParser(
        description="Run the complete Speech2Health pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run everything
  python run_pipeline.py --all

  # Run only model training
  python run_pipeline.py --step training

  # Train only specific models
  python run_pipeline.py --models random_forest xgboost

  # Run only biomarker analysis
  python run_pipeline.py --step biomarkers
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the complete pipeline (all steps)"
    )
    
    parser.add_argument(
        "--step",
        choices=["download", "prepare", "features", "training", "biomarkers", "temporal", "crosslingual"],
        help="Run a specific step only"
    )
    
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training steps"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["random_forest", "xgboost", "neural"],
        default=["random_forest", "xgboost", "neural"],
        help="Which models to train (default: all)"
    )
    
    parser.add_argument(
        "--skip-biomarkers",
        action="store_true",
        help="Skip biomarker identification"
    )
    
    parser.add_argument(
        "--skip-temporal",
        action="store_true",
        help="Skip temporal stability analysis"
    )
    
    parser.add_argument(
        "--skip-crosslingual",
        action="store_true",
        help="Skip cross-lingual analysis"
    )
    
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check which steps have been completed, don't run anything"
    )
    
    args = parser.parse_args()
    
    # If no specific step, run full pipeline unless check-only
    run_full = args.all or (args.step is None and not args.check_only)
    
    print_header("Speech2Health Pipeline Runner")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    
    # Define pipeline steps
    steps = []
    step_num = 0
    
    # Step 1: Data Preparation
    if run_full or args.step == "prepare":
        steps.append(("prepare", "Data Preparation", "scripts/01_prepare_data.py", True))
    
    # Step 3: Feature Extraction
    if run_full or args.step == "features":
        steps.append(("features", "Feature Extraction", "scripts/02_extract_features.py", True))
    
    # Step 4: Model Training
    if (run_full or args.step == "training") and not args.skip_training:
        if "random_forest" in args.models:
            steps.append(("train_rf", "Train Random Forest", "scripts/03_train_classical.py", True))
        if "xgboost" in args.models:
            steps.append(("train_xgb", "Train XGBoost", "scripts/03_train_xgboost.py", True))
        if "neural" in args.models:
            steps.append(("train_nn", "Train Neural Network", "scripts/04_train_neural.py", True))
    
    # Step 5: Biomarker Identification
    if (run_full or args.step == "biomarkers") and not args.skip_biomarkers:
        steps.append(("biomarkers", "Biomarker Identification (SHAP)", "scripts/06_identify_biomarkers.py", True))
    
    # Step 6: Temporal Stability
    if (run_full or args.step == "temporal") and not args.skip_temporal:
        steps.append(("temporal", "Temporal Stability Analysis", "scripts/08_temporal_stability.py", False))
    
    # Step 7: Cross-Lingual Analysis
    if (run_full or args.step == "crosslingual") and not args.skip_crosslingual:
        steps.append(("crosslingual", "Cross-Lingual Analysis", "scripts/09_cross_lingual_analysis.py", False))
    
    if not steps:
        print("No steps to run. Use --all or --step to specify what to run.")
        return
    
    # Check-only mode: verify what's already done
    if args.check_only:
        print_header("Pipeline Status Check")
        print("Checking which steps have been completed...\n")
        
        status = {
            "Data Files": check_file_exists("data/raw", "Raw data directory"),
            "Metadata": check_file_exists("data/features/metadata.csv", "Metadata file"),
            "Aggregated Features": check_file_exists("data/features/aggregated_features.csv", "Aggregated features"),
            "Random Forest Model": check_file_exists("results/models/random_forest_model.pkl", "Random Forest model"),
            "XGBoost Model": check_file_exists("results/models/xgboost_model.pkl", "XGBoost model"),
            "Neural Network Model": check_file_exists("results/models/neural_model.pt", "Neural network model"),
            "Biomarkers": check_file_exists("results/biomarkers/significant_biomarkers.csv", "Biomarker results"),
            "Temporal Analysis": check_file_exists("results/biomarkers/temporal_stability.csv", "Temporal stability results"),
        }
        
        print("\n" + "=" * 70)
        print("Status Summary:")
        completed = sum(1 for v in status.values() if v)
        total = len(status)
        print(f"Completed: {completed}/{total} checks passed")
        print("=" * 70)
        return
    
    # Run pipeline steps
    total_steps = len(steps)
    failed_steps = []
    
    print_header(f"Running Pipeline ({total_steps} steps)")
    
    for idx, (step_id, description, script_path, required) in enumerate(steps, 1):
        print_step(idx, total_steps, description)
        
        success = run_script(script_path, description, required)
        
        if not success:
            if required:
                print(f"\nERROR: Required step '{description}' failed!")
                print("Pipeline stopped. Fix the error and re-run.")
                failed_steps.append(description)
                break
            else:
                print(f"Warning: Optional step '{description}' failed, continuing...")
                failed_steps.append(description)
    
    # Final summary
    print_header("Pipeline Summary")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total steps: {total_steps}")
    print(f"Successful: {total_steps - len(failed_steps)}")
    
    if failed_steps:
        print(f"Failed/ Skipped: {len(failed_steps)}")
        print("Failed steps:")
        for step in failed_steps:
            print(f"  - {step}")
    else:
        print("✓ All steps completed successfully!")
    
    print("\n" + "=" * 70)
    print("Results Location:")
    print("  - Models: results/models/")
    print("  - Performance: results/tables/")
    print("  - Biomarkers: results/biomarkers/")
    print("  - Plots: results/plots/")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
