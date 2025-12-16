"""
Temporal stability analysis of stress indicators across sessions.
Analyzes how features change over time within and across sessions.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def load_session_data(features_path="data/features/aggregated_features.csv",
                     metadata_path="data/processed/metadata.csv"):
    """Load features with session/temporal information."""
    print("Loading features and metadata...")
    
    df = pd.read_csv(features_path)
    metadata = pd.read_csv(metadata_path) if os.path.exists(metadata_path) else None
    
    # Merge metadata if available
    if metadata is not None and 'participant_id' in metadata.columns:
        # Try to merge on participant_id
        df = df.merge(metadata[['participant_id']], on='participant_id', how='left')
    
    # Remove NaN labels
    df = df.dropna(subset=['label'])
    
    print(f"  Loaded {len(df)} samples")
    return df

def analyze_within_session_stability(df):
    """
    Analyze temporal stability within a session.
    For DAIC-WOZ, we have aggregated features per participant.
    This function would analyze segment-level features if available.
    """
    print("\n Within-Session Stability Analysis:")
    print("  (Note: DAIC-WOZ provides aggregated features per participant)")
    print("  For segment-level analysis, raw frame-level features are needed.")
    
    # If we had segment-level data, we would:
    # 1. Group by participant_id and segment/session
    # 2. Calculate feature variance across segments
    # 3. Identify features with high/low temporal stability
    
    # For now, calculate feature stability across participants
    feature_cols = [col for col in df.columns if col not in ['participant_id', 'label']]
    
    # Calculate coefficient of variation (std/mean) as stability measure
    stability_metrics = {}
    
    for feat in feature_cols:
        feat_values = df[feat].values
        # Remove NaN/Inf
        feat_clean = feat_values[~np.isnan(feat_values) & np.isfinite(feat_values)]
        
        if len(feat_clean) > 0 and np.mean(feat_clean) != 0:
            cv = np.std(feat_clean) / np.mean(feat_clean)  # Coefficient of variation
            stability_metrics[feat] = {
                'cv': cv,
                'std': np.std(feat_clean),
                'mean': np.mean(feat_clean),
                'stability_score': 1 / (1 + cv)  # Higher = more stable
            }
    
    stability_df = pd.DataFrame(stability_metrics).T
    stability_df = stability_df.sort_values('stability_score', ascending=False)
    
    print(f"\n  Top 10 Most Stable Features:")
    print(stability_df.head(10)[['mean', 'std', 'cv', 'stability_score']].to_string())
    
    print(f"\n  Top 10 Most Variable Features:")
    print(stability_df.tail(10)[['mean', 'std', 'cv', 'stability_score']].to_string())
    
    # Save
    os.makedirs("results/biomarkers", exist_ok=True)
    stability_df.to_csv("results/biomarkers/temporal_stability.csv")
    print(f"\n   Saved to results/biomarkers/temporal_stability.csv")
    
    return stability_df

def analyze_cross_session_stability(df, metadata_path=None):
    """
    Analyze stability across multiple sessions (if available).
    DAIC-WOZ typically has one session per participant, but this function
    handles multiple sessions if metadata indicates them.
    """
    print("\n Cross-Session Stability Analysis:")
    
    # Check if we have multiple sessions per participant
    if metadata_path and os.path.exists(metadata_path):
        metadata = pd.read_csv(metadata_path)
        
        # Look for session indicators in metadata or filenames
        # For DAIC-WOZ, typically one session per participant
        
        # Count sessions per participant
        session_counts = df.groupby('participant_id').size()
        multi_session_participants = session_counts[session_counts > 1]
        
        if len(multi_session_participants) > 0:
            print(f"  Found {len(multi_session_participants)} participants with multiple sessions")
            
            # Calculate feature stability across sessions for these participants
            feature_cols = [col for col in df.columns if col not in ['participant_id', 'label']]
            
            stability_by_participant = {}
            
            for pid in multi_session_participants.index[:10]:  # Analyze first 10
                participant_data = df[df['participant_id'] == pid]
                
                if len(participant_data) < 2:
                    continue
                
                # Calculate variance across sessions for each feature
                for feat in feature_cols:
                    feat_values = participant_data[feat].values
                    feat_clean = feat_values[~np.isnan(feat_values) & np.isfinite(feat_values)]
                    
                    if len(feat_clean) >= 2:
                        if feat not in stability_by_participant:
                            stability_by_participant[feat] = []
                        stability_by_participant[feat].append(np.std(feat_clean))
            
            # Average stability across participants
            avg_stability = {feat: np.mean(vars) for feat, vars in stability_by_participant.items()}
            stability_df = pd.DataFrame([avg_stability]).T
            stability_df.columns = ['avg_cross_session_variance']
            stability_df = stability_df.sort_values('avg_cross_session_variance')
            
            print(f"\n  Features with Lowest Cross-Session Variance (most stable):")
            print(stability_df.head(10).to_string())
            
            os.makedirs("results/biomarkers", exist_ok=True)
            stability_df.to_csv("results/biomarkers/cross_session_stability.csv")
            print(f"\n   Saved to results/biomarkers/cross_session_stability.csv")
            
            return stability_df
        else:
            print("  No participants with multiple sessions found")
            print("  (DAIC-WOZ typically has one session per participant)")
    else:
        print("  No metadata available for cross-session analysis")
    
    return None

def plot_temporal_trajectories(df, top_features=5):
    """Plot temporal trajectories of top features (if segment-level data available)."""
    print("\n Generating Temporal Trajectory Plots:")
    
    # For DAIC-WOZ aggregated data, we can't plot true temporal trajectories
    # But we can visualize feature distributions by label
    
    feature_cols = [col for col in df.columns if col not in ['participant_id', 'label']]
    
    # Select top features by variance
    feature_vars = df[feature_cols].var().sort_values(ascending=False)
    top_feat_names = feature_vars.head(top_features).index.tolist()
    
    # Create distribution plots
    fig, axes = plt.subplots(1, min(top_features, 5), figsize=(15, 3))
    if top_features == 1:
        axes = [axes]
    
    for idx, feat in enumerate(top_feat_names[:5]):
        ax = axes[idx] if top_features > 1 else axes[0]
        
        # Plot distributions by label
        for label in [0, 1]:
            label_data = df[df['label'] == label][feat].dropna()
            ax.hist(label_data, alpha=0.5, label=f'Label {label}', bins=20)
        
        ax.set_xlabel(feat[:30])  # Truncate long names
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_title(f'Feature: {feat[:40]}')
    
    plt.tight_layout()
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig("results/plots/temporal_stability_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   Saved plot to results/plots/temporal_stability_distributions.png")

def calculate_stress_indicator_stability(df, biomarker_features=None):
    """
    Calculate stability of known stress indicators.
    High stability = reliable biomarker across time.
    """
    print("\n Stress Indicator Stability:")
    
    # If biomarker features provided (from SHAP analysis), analyze those
    if biomarker_features is None:
        # Load significant biomarkers if available
        biomarker_path = "results/biomarkers/significant_biomarkers.csv"
        if os.path.exists(biomarker_path):
            biomarkers_df = pd.read_csv(biomarker_path)
            # Use tree-based or SHAP-based features
            biomarker_features = biomarkers_df['tree_based'].dropna().head(20).tolist()
    
    if biomarker_features is None:
        print("  No biomarker features specified. Skipping.")
        return None
    
    # Filter to features that exist
    available_features = [f for f in biomarker_features if f in df.columns]
    
    if len(available_features) == 0:
        print("  No biomarker features found in data")
        return None
    
    print(f"  Analyzing stability of {len(available_features)} biomarker features...")
    
    stability_results = []
    
    for feat in available_features:
        feat_values = df[feat].values
        feat_clean = feat_values[~np.isnan(feat_values) & np.isfinite(feat_values)]
        
        if len(feat_clean) > 0:
            # Coefficient of variation
            cv = np.std(feat_clean) / np.mean(feat_clean) if np.mean(feat_clean) != 0 else np.inf
            
            # Stability score (inverse of CV)
            stability_score = 1 / (1 + cv)
            
            # Separability: difference between classes
            class0_mean = df[df['label'] == 0][feat].mean()
            class1_mean = df[df['label'] == 1][feat].mean()
            separability = abs(class1_mean - class0_mean) / (np.std(feat_clean) + 1e-10)
            
            stability_results.append({
                'feature': feat,
                'coefficient_of_variation': cv,
                'stability_score': stability_score,
                'separability': separability,
                'mean_class0': class0_mean,
                'mean_class1': class1_mean
            })
    
    stability_df = pd.DataFrame(stability_results)
    stability_df = stability_df.sort_values('stability_score', ascending=False)
    
    print(f"\n  Top 10 Most Stable Biomarkers:")
    print(stability_df.head(10)[['feature', 'stability_score', 'separability']].to_string())
    
    os.makedirs("results/biomarkers", exist_ok=True)
    stability_df.to_csv("results/biomarkers/stress_indicator_stability.csv", index=False)
    print(f"\n   Saved to results/biomarkers/stress_indicator_stability.csv")
    
    return stability_df

def main():
    """Main temporal stability analysis pipeline."""
    print("=" * 70)
    print("Temporal Stability Analysis")
    print("=" * 70)
    
    # Load data
    df = load_session_data()
    
    # Analyze stability
    within_session_stability = analyze_within_session_stability(df)
    cross_session_stability = analyze_cross_session_stability(df)
    
    # Calculate stress indicator stability
    stress_stability = calculate_stress_indicator_stability(df)
    
    # Generate plots
    plot_temporal_trajectories(df)
    
    print("\n" + "=" * 70)
    print(" Temporal Stability Analysis Complete!")
    print("=" * 70)
    print("\nResults saved to:")
    print("  - results/biomarkers/temporal_stability.csv")
    if cross_session_stability is not None:
        print("  - results/biomarkers/cross_session_stability.csv")
    if stress_stability is not None:
        print("  - results/biomarkers/stress_indicator_stability.csv")
    print("  - results/plots/temporal_stability_distributions.png")

if __name__ == "__main__":
    main()


