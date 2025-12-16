"""
Script to extract raw audio features using praat-parselmouth and librosa.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from code.feature_extraction.audio_features import extract_audio_features_batch

def main():
    """Extract raw audio features from audio files."""
    # Default: look for audio files in data/raw/
    audio_dir = "data/raw"
    
    # Alternatively, specify directory with audio files
    if len(sys.argv) > 1:
        audio_dir = sys.argv[1]
    
    print("Raw Audio Feature Extraction")
    print(f"Looking for audio files in: {audio_dir}")
    
    # Extract features
    extract_audio_features_batch(audio_dir, output_path="data/features/raw_audio_features.csv")
    
    print("\n" + "=" * 70)
    print(" Raw audio feature extraction complete!")
    print("=" * 70)
    print("\nNote: This extracts features directly from raw audio files.")
    print("For DAIC-WOZ, you may need to download audio files separately.")
    print("The pre-extracted COVAREP/FORMANT features are already available.")

if __name__ == "__main__":
    main()


