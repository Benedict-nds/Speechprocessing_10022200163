"""
Cross-lingual analysis and adaptation script.
Analyzes performance across languages and demonstrates adaptation techniques.
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from code.data.cross_lingual import CrossLingualDataLoader, CrossLingualAdapter, create_cross_lingual_dataset_template

def main():
    """Cross-lingual analysis pipeline."""
    print("=" * 70)
    print("Cross-Lingual Adaptation Framework")
    print("=" * 70)
    
    # Check if cross-lingual data exists
    cross_lingual_dir = "data/cross_lingual"
    
    if not os.path.exists(cross_lingual_dir):
        print(f"\n  Cross-lingual data directory not found: {cross_lingual_dir}")
        print("  Creating template structure...")
        create_cross_lingual_dataset_template(cross_lingual_dir)
        print("\n   Template created. To use:")
        print("    1. Add your LMIC datasets to data/cross_lingual/<language>/")
        print("    2. Ensure CSV files have 'language' column or language in filename")
        print("    3. Re-run this script")
        return
    
    # Initialize loader
    loader = CrossLingualDataLoader(source_language='en')
    
    # Load multilingual dataset
    df = loader.load_multilingual_dataset(cross_lingual_dir)
    
    if df is None or len(df) == 0:
        print("\n  No multilingual data found.")
        print("  Add CSV files with language labels to proceed.")
        return
    
    # Prepare features
    X, y, languages = loader.prepare_cross_lingual_features(df)
    
    # Split by language
    language_datasets = loader.split_by_language(df)
    
    print("\n" + "=" * 70)
    print("Language-Specific Analysis")
    print("=" * 70)
    
    # Analyze each language
    for lang_code, lang_df in language_datasets.items():
        lang_name = loader.language_codes.get(lang_code, lang_code)
        print(f"\n{lang_name} ({lang_code}):")
        
        if 'label' in lang_df.columns:
            label_dist = lang_df['label'].value_counts()
            print(f"  Labels: {dict(label_dist)}")
        
        feature_cols = [col for col in lang_df.columns if col not in ['participant_id', 'label', 'language']]
        print(f"  Features: {len(feature_cols)}")
    
    print("\n" + "=" * 70)
    print(" Cross-Lingual Analysis Complete!")
    print("=" * 70)
    print("\nSee code/data/cross_lingual.py for implementation details.")

if __name__ == "__main__":
    main()


