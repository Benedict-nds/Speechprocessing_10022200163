"""
Audio and feature preprocessing utilities.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional audio processing imports
try:
    import librosa
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: librosa/soundfile not available. Audio preprocessing disabled.")


class AudioPreprocessor:
    """Preprocesses audio files for feature extraction."""
    
    def __init__(self, 
                 target_sr=16000,
                 mono=True,
                 normalize_volume=True,
                 vad_enabled=True):
        self.target_sr = target_sr
        self.mono = mono
        self.normalize_volume = normalize_volume
        self.vad_enabled = vad_enabled
    
    def load_audio(self, audio_path):
        """Load audio file."""
        if not AUDIO_AVAILABLE:
            return None
        
        if not os.path.exists(audio_path):
            return None
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.target_sr, mono=self.mono)
            return audio, sr
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None
    
    def normalize_audio(self, audio):
        """Normalize audio volume."""
        if not self.normalize_volume:
            return audio
        
        # Peak normalization
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom
        return audio
    
    def apply_vad(self, audio, sr):
        """Apply Voice Activity Detection (simple energy-based)."""
        if not AUDIO_AVAILABLE:
            return audio
        
        if not self.vad_enabled:
            return audio
        
        # Simple energy-based VAD
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate frame energy
        energy = librosa.feature.rms(
            y=audio, 
            frame_length=frame_length, 
            hop_length=hop_length
        )[0]
        
        # Threshold: mean + 0.5 * std
        threshold = energy.mean() + 0.5 * energy.std()
        
        # Find voice frames
        voice_frames = energy > threshold
        
        # Convert frames to samples
        frame_indices = np.where(voice_frames)[0]
        if len(frame_indices) == 0:
            return audio  # Return original if no voice detected
        
        start_frame = frame_indices[0]
        end_frame = frame_indices[-1] + 1
        
        start_sample = start_frame * hop_length
        end_sample = end_frame * hop_length
        
        return audio[start_sample:end_sample]
    
    def preprocess(self, audio_path, output_path=None):
        """Preprocess audio file."""
        result = self.load_audio(audio_path)
        if result is None:
            return None
        
        audio, sr = result
        
        # Normalize volume
        audio = self.normalize_audio(audio)
        
        # Apply VAD
        audio = self.apply_vad(audio, sr)
        
        # Save if output path provided
        if output_path and AUDIO_AVAILABLE:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, sr)
        
        return audio, sr


class FeaturePreprocessor:
    """Preprocesses extracted features."""
    
    def __init__(self):
        pass
    
    def load_covarep(self, covarep_path):
        """Load COVAREP features."""
        if not covarep_path or not os.path.exists(covarep_path):
            return None
        
        try:
            # COVAREP files have no header, comma-separated
            features = pd.read_csv(covarep_path, header=None).values
            return features
        except Exception as e:
            print(f"Error loading COVAREP {covarep_path}: {e}")
            return None
    
    def load_formant(self, formant_path):
        """Load FORMANT features."""
        if not formant_path or not os.path.exists(formant_path):
            return None
        
        try:
            # FORMANT files have no header, comma-separated
            features = pd.read_csv(formant_path, header=None).values
            return features
        except Exception as e:
            print(f"Error loading FORMANT {formant_path}: {e}")
            return None
    
    def load_openface_features(self, feature_path):
        """Load OpenFace feature file (comma-separated)."""
        if not feature_path or not os.path.exists(feature_path):
            return None
        
        try:
            # OpenFace CLNF files are comma-separated with header
            features = pd.read_csv(feature_path, sep=',')
            # Return as numpy array (values only, no header)
            return features.values
        except Exception as e:
            print(f"Error loading OpenFace features {feature_path}: {e}")
            return None
    
    def align_features(self, features_dict, target_length=None):
        """Align features to same temporal length."""
        # Find minimum length
        lengths = {k: len(v) for k, v in features_dict.items() if v is not None}
        if not lengths:
            return None
        
        min_length = min(lengths.values())
        
        if target_length is not None:
            min_length = min(min_length, target_length)
        
        # Truncate or pad features
        aligned = {}
        for key, features in features_dict.items():
            if features is None:
                continue
            
            if len(features) > min_length:
                # Truncate
                aligned[key] = features[:min_length]
            elif len(features) < min_length:
                # Pad with last value
                if len(features.shape) == 1:
                    padding = np.repeat(features[-1:], min_length - len(features))
                    aligned[key] = np.concatenate([features, padding])
                else:
                    padding = np.repeat(features[-1:, :], min_length - len(features), axis=0)
                    aligned[key] = np.concatenate([features, padding], axis=0)
            else:
                aligned[key] = features
        
        return aligned
    
    def aggregate_features(self, features, method='all'):
        """
        Aggregate temporal features to fixed-length vector.
        
        CRITICAL: This converts frame-level features (e.g., 98k frames) 
        to speaker-level features (~150-300 features per speaker).
        This aggregation is essential for achieving >80% accuracy.
        
        Methods:
        - 'all': Includes mean, std, min, max, median, 25th percentile, 75th percentile
        - 'mean', 'std', 'min', 'max', 'median', 'p25', 'p75': Individual statistics
        """
        if features is None:
            return None
        
        if method == 'mean':
            return np.mean(features, axis=0)
        elif method == 'std':
            return np.std(features, axis=0)
        elif method == 'min':
            return np.min(features, axis=0)
        elif method == 'max':
            return np.max(features, axis=0)
        elif method == 'median':
            return np.median(features, axis=0)
        elif method == 'p25':
            return np.percentile(features, 25, axis=0)
        elif method == 'p75':
            return np.percentile(features, 75, axis=0)
        elif method == 'all':
            # Concatenate all statistics (CRITICAL for >80% accuracy)
            # This includes: mean, std, min, max, median, 25th percentile, 75th percentile
            return np.concatenate([
                np.mean(features, axis=0),
                np.std(features, axis=0),
                np.min(features, axis=0),
                np.max(features, axis=0),
                np.median(features, axis=0),
                np.percentile(features, 25, axis=0),  # 25th percentile
                np.percentile(features, 75, axis=0)   # 75th percentile
            ])
        else:
            return np.mean(features, axis=0)


class DataPreprocessor:
    """Main preprocessing pipeline."""
    
    def __init__(self, 
                 metadata_path="data/processed/metadata.csv",
                 processed_dir="data/processed",
                 preprocess_audio=True,
                 preprocess_features=True):
        self.metadata_path = metadata_path
        self.processed_dir = processed_dir
        self.preprocess_audio = preprocess_audio
        self.preprocess_features = preprocess_features
        
        self.audio_preprocessor = AudioPreprocessor()
        self.feature_preprocessor = FeaturePreprocessor()
        
        os.makedirs(processed_dir, exist_ok=True)
        os.makedirs(os.path.join(processed_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(processed_dir, "features"), exist_ok=True)
    
    def preprocess_all(self):
        """Preprocess all data."""
        print("Loading metadata...")
        metadata = pd.read_csv(self.metadata_path)
        
        print(f"Preprocessing {len(metadata)} participants...")
        
        results = []
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata)):
            pid = row['participant_id']
            result = self.preprocess_participant(row)
            results.append(result)
        
        # Create summary
        successful = sum(1 for r in results if r['success'])
        print(f"\nPreprocessing complete: {successful}/{len(metadata)} successful")
        
        return results
    
    def preprocess_participant(self, row):
        """Preprocess a single participant's data."""
        pid = row['participant_id']
        result = {
            'participant_id': pid,
            'success': False,
            'audio_processed': False,
            'features_processed': False,
            'errors': []
        }
        
        # Preprocess audio
        if self.preprocess_audio and pd.notna(row.get('audio_path')):
            audio_path = row['audio_path']
            output_path = os.path.join(
                self.processed_dir, 
                "audio", 
                f"{pid}_processed.wav"
            )
            
            try:
                preprocess_result = self.audio_preprocessor.preprocess(audio_path, output_path)
                if preprocess_result is not None:
                    result['audio_processed'] = True
            except Exception as e:
                result['errors'].append(f"Audio preprocessing: {e}")
        
        # Preprocess features
        if self.preprocess_features:
            try:
                features = {}
                
                # Load COVAREP
                if pd.notna(row.get('covarep_path')):
                    covarep = self.feature_preprocessor.load_covarep(row['covarep_path'])
                    if covarep is not None:
                        features['covarep'] = covarep
                
                # Load FORMANT
                if pd.notna(row.get('formant_path')):
                    formant = self.feature_preprocessor.load_formant(row['formant_path'])
                    if formant is not None:
                        features['formant'] = formant
                
                # Load OpenFace features
                for feat_type in ['au_path', 'gaze_path', 'pose_path']:
                    if pd.notna(row.get(feat_type)):
                        feat_data = self.feature_preprocessor.load_openface_features(
                            row[feat_type]
                        )
                        if feat_data is not None:
                            features[feat_type.replace('_path', '')] = feat_data
                
                # Save features
                if features:
                    output_path = os.path.join(
                        self.processed_dir,
                        "features",
                        f"{pid}.npz"
                    )
                    np.savez(output_path, **features)
                    result['features_processed'] = True
                
            except Exception as e:
                result['errors'].append(f"Feature preprocessing: {e}")
        
        result['success'] = result['audio_processed'] or result['features_processed']
        return result


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_all()

