import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Handle imports - add project root if needed
try:
    from code.data.preprocessing import FeaturePreprocessor
except ImportError:
    # If running directly, add project root to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from code.data.preprocessing import FeaturePreprocessor

class FeatureBuilder:
    def __init__(self, metadata_path="data/processed/metadata.csv",
                 out_dir="data/features"):
        self.metadata = pd.read_csv(metadata_path)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.preprocessor = FeaturePreprocessor()

    def load_feature_file(self, path, has_header=False):
        """Load feature file, handling both header and no-header formats, and .txt/.csv files."""
        if path is None or not isinstance(path, str) or not os.path.exists(path):
            return None
        try:
            # Handle both .txt and .csv files
            if path.endswith('.txt'):
                # OpenFace files are tab or space separated, try different separators
                try:
                    df = pd.read_csv(path, sep='\t', header=0 if has_header else None)
                except:
                    try:
                        df = pd.read_csv(path, sep=' ', header=0 if has_header else None)
                    except:
                        df = pd.read_csv(path, header=0 if has_header else None)
            else:
                # CSV files
                if has_header:
                    df = pd.read_csv(path)
                else:
                    # COVAREP/FORMANT have no headers
                    df = pd.read_csv(path, header=None)
            
            # Convert to numeric, dropping non-numeric columns
            # This handles string columns (like timestamps) that can't be aggregated
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) == 0:
                return None
            return numeric_df.values
        except Exception as e:
            # Silent failure - just return None (many participants may not have all features)
            return None

    def build_all(self, aggregate=True):
        """
        Build aggregated speaker-level features from frame-level features.
        
        CRITICAL: If aggregate=True (default), converts frame-level features
        to speaker-level using mean, std, min, max, median, 25th, 75th percentiles.
        This is essential for >80% accuracy with Random Forest.
        """
        print("Building aggregated speaker-level features...")
        print(f"  Aggregation: {'ENABLED (recommended for >80% accuracy)' if aggregate else 'DISABLED'}")

        all_features = []
        all_labels = []
        participant_ids = []

        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata)):
            pid = row["participant_id"]
            
            # Load frame-level features
            covarep = self.load_feature_file(row["covarep_path"], has_header=False)
            formant = self.load_feature_file(row["formant_path"], has_header=False)
            au = self.load_feature_file(row["au_path"], has_header=True)
            gaze = self.load_feature_file(row["gaze_path"], has_header=True)
            pose = self.load_feature_file(row["pose_path"], has_header=True)
            landmarks = self.load_feature_file(row["landmarks_path"], has_header=True)

            # Aggregate each feature type to speaker-level statistics
            aggregated_parts = []
            
            if covarep is not None and len(covarep) > 0 and len(covarep.shape) >= 2:
                if aggregate:
                    agg = self.preprocessor.aggregate_features(covarep, method='all')
                    if agg is not None:
                        aggregated_parts.append(agg)
                else:
                    aggregated_parts.append(covarep.flatten())
            
            if formant is not None and len(formant) > 0 and len(formant.shape) >= 2:
                if aggregate:
                    agg = self.preprocessor.aggregate_features(formant, method='all')
                    if agg is not None:
                        aggregated_parts.append(agg)
                else:
                    aggregated_parts.append(formant.flatten())
            
            # OpenFace features (AU, gaze, pose, landmarks)
            for feat_array, feat_name in [(au, "au"), (gaze, "gaze"), (pose, "pose"), (landmarks, "landmarks")]:
                if feat_array is not None and len(feat_array) > 0 and len(feat_array.shape) >= 2:
                    if aggregate:
                        agg = self.preprocessor.aggregate_features(feat_array, method='all')
                        if agg is not None:
                            aggregated_parts.append(agg)
                    else:
                        aggregated_parts.append(feat_array.flatten())

            if aggregated_parts:
                # Concatenate all aggregated features for this participant
                participant_feature = np.concatenate(aggregated_parts)
                all_features.append(participant_feature)
                all_labels.append(row["phq8_binary"])
                participant_ids.append(pid)

        if not all_features:
            print("  Warning: No features extracted!")
            return None, None, None

        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Save as CSV for classical ML
        feature_df = pd.DataFrame(X)
        feature_df['participant_id'] = participant_ids
        feature_df['label'] = y
        
        csv_path = os.path.join(self.out_dir, "aggregated_features.csv")
        feature_df.to_csv(csv_path, index=False)
        
        print(f"\n Features saved to: {csv_path}")
        print(f"   Shape: {X.shape} (participants × features)")
        print(f"   Features per participant: {X.shape[1] if len(X) > 0 else 0}")
        print(f"   Participants with features: {len(participant_ids)}")

        return X, y, participant_ids