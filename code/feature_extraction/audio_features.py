"""
Raw audio feature extraction using praat-parselmouth and librosa.
Extracts pitch, jitter, shimmer, energy, MFCCs from raw audio files.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import audio processing libraries
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("Warning: librosa not available. Install with: pip install librosa soundfile")

try:
    import parselmouth
    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    print("Warning: parselmouth not available. Install with: pip install praat-parselmouth")

try:
    import pyworld as pw
    PYWORLD_AVAILABLE = True
except ImportError:
    PYWORLD_AVAILABLE = False
    print("Warning: pyworld not available. Install with: pip install pyworld")

class AudioFeatureExtractor:
    """Extract acoustic-prosodic features from raw audio files."""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def extract_pitch(self, audio, sr, method='parselmouth'):
        """
        Extract pitch (F0) features.
        
        Returns:
            f0: Fundamental frequency contour
        """
        if method == 'parselmouth' and PARSELMOUTH_AVAILABLE:
            try:
                # Convert to parselmouth Sound object
                sound = parselmouth.Sound(audio, sampling_frequency=sr)
                
                # Extract pitch using Praat algorithm
                pitch = sound.to_pitch_ac(
                    time_step=0.01,  # 10ms steps
                    voicing_threshold=0.45,
                    octave_cost=0.01,
                    octave_jump_cost=0.35,
                    voiced_unvoiced_cost=0.14
                )
                
                # Get F0 values
                f0_values = pitch.selected_array['frequency']
                f0_values[f0_values == 0] = np.nan  # Unvoiced = NaN
                
                return f0_values
            except Exception as e:
                print(f"  Parselmouth pitch extraction failed: {e}")
                return None
        
        elif method == 'pyworld' and PYWORLD_AVAILABLE:
            try:
                # PyWorld pitch extraction
                f0, t = pw.harvest(audio, sr)
                f0[f0 == 0] = np.nan
                return f0
            except Exception as e:
                print(f"  PyWorld pitch extraction failed: {e}")
                return None
        
        elif LIBROSA_AVAILABLE:
            try:
                # Librosa pitch estimation (less accurate but available)
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    audio,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7')
                )
                return f0
            except Exception as e:
                print(f"  Librosa pitch extraction failed: {e}")
                return None
        
        return None
    
    def calculate_jitter(self, f0):
        """
        Calculate jitter (pitch period variation).
        
        Returns:
            local_jitter: Local jitter (%)
            relative_jitter: Relative jitter
        """
        if f0 is None or len(f0) < 2:
            return None, None
        
        # Remove NaN and zero values
        f0_clean = f0[~np.isnan(f0) & (f0 > 0)]
        if len(f0_clean) < 2:
            return None, None
        
        # Calculate periods (1/f0)
        periods = 1.0 / f0_clean
        
        # Local jitter: average absolute difference between consecutive periods
        period_diffs = np.abs(np.diff(periods))
        local_jitter = np.mean(period_diffs) / np.mean(periods) * 100
        
        # Relative jitter: std of periods / mean period
        relative_jitter = np.std(periods) / np.mean(periods)
        
        return local_jitter, relative_jitter
    
    def calculate_shimmer(self, audio, f0):
        """
        Calculate shimmer (amplitude variation).
        
        Returns:
            local_shimmer: Local shimmer (%)
            relative_shimmer: Relative shimmer
        """
        if f0 is None or len(f0) < 2 or not LIBROSA_AVAILABLE:
            return None, None
        
        # Get amplitude envelope
        frame_length = int(0.025 * self.sample_rate)  # 25ms
        hop_length = int(0.010 * self.sample_rate)    # 10ms
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Align with F0 (downsample RMS to match F0 length)
        if len(rms) != len(f0):
            # Interpolate or downsample
            from scipy.interpolate import interp1d
            f = interp1d(np.linspace(0, 1, len(rms)), rms, kind='linear')
            rms = f(np.linspace(0, 1, len(f0)))
        
        # Remove NaN and zero
        valid_mask = ~np.isnan(f0) & (f0 > 0)
        if np.sum(valid_mask) < 2:
            return None, None
        
        rms_clean = rms[valid_mask]
        
        # Local shimmer: average absolute difference between consecutive amplitudes
        amp_diffs = np.abs(np.diff(rms_clean))
        local_shimmer = np.mean(amp_diffs) / np.mean(rms_clean) * 100
        
        # Relative shimmer: std / mean
        relative_shimmer = np.std(rms_clean) / np.mean(rms_clean)
        
        return local_shimmer, relative_shimmer
    
    def extract_mfccs(self, audio, sr, n_mfcc=13, include_deltas=True):
        """
        Extract MFCC features.
        
        Returns:
            mfccs: MFCC coefficients
            mfcc_delta: Delta MFCCs
            mfcc_delta2: Delta-delta MFCCs
        """
        if not LIBROSA_AVAILABLE:
            return None, None, None
        
        try:
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=2048,
                hop_length=512
            )
            
            mfcc_delta = None
            mfcc_delta2 = None
            
            if include_deltas:
                # Delta (first derivative)
                mfcc_delta = librosa.feature.delta(mfccs)
                
                # Delta-delta (second derivative)
                mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
            
            return mfccs, mfcc_delta, mfcc_delta2
        
        except Exception as e:
            print(f"  MFCC extraction failed: {e}")
            return None, None, None
    
    def extract_energy(self, audio, sr):
        """
        Extract energy features.
        
        Returns:
            rms_energy: RMS energy contour
        """
        if not LIBROSA_AVAILABLE:
            return None
        
        try:
            frame_length = int(0.025 * sr)  # 25ms
            hop_length = int(0.010 * sr)    # 10ms
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            return rms
        except Exception as e:
            print(f"  Energy extraction failed: {e}")
            return None
    
    def extract_all_features(self, audio_path):
        """
        Extract all features from an audio file.
        
        Returns:
            features_dict: Dictionary of extracted features
        """
        if not LIBROSA_AVAILABLE:
            print(f"  Cannot extract features: librosa not available")
            return None
        
        if not os.path.exists(audio_path):
            print(f"  Audio file not found: {audio_path}")
            return None
        
        print(f"  Extracting features from {os.path.basename(audio_path)}...")
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            features = {}
            
            # Extract pitch
            f0 = self.extract_pitch(audio, sr, method='parselmouth' if PARSELMOUTH_AVAILABLE else 'pyworld')
            if f0 is not None:
                features['f0'] = f0
                
                # Calculate jitter from F0
                local_jitter, relative_jitter = self.calculate_jitter(f0)
                features['local_jitter'] = local_jitter
                features['relative_jitter'] = relative_jitter
            
            # Extract shimmer
            local_shimmer, relative_shimmer = self.calculate_shimmer(audio, f0)
            if local_shimmer is not None:
                features['local_shimmer'] = local_shimmer
                features['relative_shimmer'] = relative_shimmer
            
            # Extract MFCCs
            mfccs, mfcc_delta, mfcc_delta2 = self.extract_mfccs(audio, sr)
            if mfccs is not None:
                features['mfccs'] = mfccs
                if mfcc_delta is not None:
                    features['mfcc_delta'] = mfcc_delta
                if mfcc_delta2 is not None:
                    features['mfcc_delta2'] = mfcc_delta2
            
            # Extract energy
            energy = self.extract_energy(audio, sr)
            if energy is not None:
                features['energy'] = energy
            
            return features
        
        except Exception as e:
            print(f"  Error extracting features from {audio_path}: {e}")
            return None
    
    def aggregate_features(self, features_dict):
        """
        Aggregate frame-level features to speaker-level statistics.
        
        Returns:
            aggregated: Dictionary of aggregated statistics
        """
        aggregated = {}
        
        for feat_name, feat_values in features_dict.items():
            if feat_values is None:
                continue
            
            # Flatten if 2D
            if feat_values.ndim == 2:
                feat_values = feat_values.flatten()
            
            # Remove NaN and inf
            feat_clean = feat_values[~np.isnan(feat_values) & np.isfinite(feat_values)]
            
            if len(feat_clean) == 0:
                continue
            
            # Calculate statistics
            aggregated[f'{feat_name}_mean'] = np.mean(feat_clean)
            aggregated[f'{feat_name}_std'] = np.std(feat_clean)
            aggregated[f'{feat_name}_min'] = np.min(feat_clean)
            aggregated[f'{feat_name}_max'] = np.max(feat_clean)
            aggregated[f'{feat_name}_median'] = np.median(feat_clean)
            aggregated[f'{feat_name}_q25'] = np.percentile(feat_clean, 25)
            aggregated[f'{feat_name}_q75'] = np.percentile(feat_clean, 75)
        
        return aggregated

def extract_audio_features_batch(audio_dir, output_path="data/features/raw_audio_features.csv"):
    """
    Extract features from all audio files in a directory.
    
    Args:
        audio_dir: Directory containing audio files
        output_path: Path to save aggregated features CSV
    """
    print("=" * 70)
    print("Raw Audio Feature Extraction")
    print("=" * 70)
    
    extractor = AudioFeatureExtractor()
    
    # Find audio files
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).glob(f'**/*{ext}'))
    
    if len(audio_files) == 0:
        print(f"  No audio files found in {audio_dir}")
        return
    
    print(f"  Found {len(audio_files)} audio files")
    
    # Extract features
    all_features = []
    participant_ids = []
    
    for audio_file in audio_files:
        # Extract participant ID from filename or path
        # Try to match DAIC-WOZ naming pattern (e.g., "302_P", "303")
        filename = audio_file.stem
        if '_' in filename:
            pid = filename.split('_')[0]
        else:
            pid = filename
        
        features = extractor.extract_all_features(str(audio_file))
        
        if features is not None:
            aggregated = extractor.aggregate_features(features)
            if aggregated:
                all_features.append(aggregated)
                participant_ids.append(pid)
    
    if len(all_features) == 0:
        print("  No features extracted successfully")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    df.insert(0, 'participant_id', participant_ids)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n Extracted features from {len(df)} audio files")
    print(f"   Saved to {output_path}")
    print(f"   Feature count: {len(df.columns) - 1}")
    
    return df

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        audio_dir = sys.argv[1]
        extract_audio_features_batch(audio_dir)
    else:
        print("Usage: python audio_features.py <audio_directory>")


