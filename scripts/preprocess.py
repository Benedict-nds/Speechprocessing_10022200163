"""
Preprocessing script for audio and features.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from code.data.preprocessing import DataPreprocessor
from code.data.validation import DataValidator


def main():
    print("=" * 60)
    print("SPEECH2HEALTH DATA PREPROCESSING")
    print("=" * 60)
    
    # Step 1: Validate data
    print("\n[Step 1] Validating data...")
    validator = DataValidator()
    is_valid = validator.validate_all()
    
    if not is_valid:
        print("\n  Data validation found issues. Continuing anyway...")
    
    # Step 2: Preprocess data
    print("\n[Step 2] Preprocessing data...")
    preprocessor = DataPreprocessor(
        preprocess_audio=True,
        preprocess_features=True
    )
    
    results = preprocessor.preprocess_all()
    
    # Step 3: Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r['success'])
    audio_processed = sum(1 for r in results if r['audio_processed'])
    features_processed = sum(1 for r in results if r['features_processed'])
    
    print(f"Total participants: {len(results)}")
    print(f"Successfully processed: {successful}")
    print(f"  - Audio processed: {audio_processed}")
    print(f"  - Features processed: {features_processed}")
    
    # Count errors
    errors = [r for r in results if r['errors']]
    if errors:
        print(f"\n  {len(errors)} participants had errors:")
        for r in errors[:5]:  # Show first 5
            print(f"  - Participant {r['participant_id']}: {r['errors']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    print("\n Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

