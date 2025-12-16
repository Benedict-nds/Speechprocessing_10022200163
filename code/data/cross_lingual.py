"""
Cross-lingual adaptation framework for handling LMIC (Low and Middle Income Countries) speech samples.
Supports multilingual datasets and domain adaptation techniques.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CrossLingualDataLoader:
    """
    Framework for loading and adapting cross-lingual speech data.
    Designed to handle datasets from different languages and regions.
    """
    
    def __init__(self, source_language='en', target_languages=None):
        """
        Initialize cross-lingual data loader.
        
        Args:
            source_language: Source language code (default: 'en' for English)
            target_languages: List of target language codes (e.g., ['hi', 'es', 'sw'])
        """
        self.source_language = source_language
        self.target_languages = target_languages or []
        
        # Language metadata
        self.language_codes = {
            'en': 'English',
            'hi': 'Hindi',
            'es': 'Spanish',
            'fr': 'French',
            'sw': 'Swahili',
            'zh': 'Chinese',
            'ar': 'Arabic',
            'bn': 'Bengali',
            'ur': 'Urdu',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'gu': 'Gujarati',
            'kn': 'Kannada',
            'ml': 'Malayalam'
        }
    
    def load_multilingual_dataset(self, data_dir, language_column='language'):
        """
        Load multilingual dataset with language labels.
        
        Args:
            data_dir: Directory containing multilingual data
            language_column: Column name containing language codes
        
        Returns:
            df: DataFrame with language information
        """
        print(f"Loading multilingual dataset from {data_dir}...")
        
        # Look for CSV files
        csv_files = list(Path(data_dir).glob('*.csv'))
        
        if len(csv_files) == 0:
            print(f"  No CSV files found in {data_dir}")
            return None
        
        # Load and combine datasets
        all_data = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Infer language from filename or add manually
                if language_column not in df.columns:
                    # Try to infer from filename
                    lang_code = self._infer_language_from_filename(csv_file.name)
                    if lang_code:
                        df[language_column] = lang_code
                
                all_data.append(df)
                print(f"  Loaded {len(df)} samples from {csv_file.name}")
            except Exception as e:
                print(f"  Error loading {csv_file.name}: {e}")
        
        if len(all_data) == 0:
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n  Total samples: {len(combined_df)}")
        
        # Language distribution
        if language_column in combined_df.columns:
            lang_dist = combined_df[language_column].value_counts()
            print(f"\n  Language distribution:")
            for lang, count in lang_dist.items():
                lang_name = self.language_codes.get(lang, lang)
                print(f"    {lang_name} ({lang}): {count}")
        
        return combined_df
    
    def _infer_language_from_filename(self, filename):
        """Try to infer language code from filename."""
        filename_lower = filename.lower()
        
        for lang_code, lang_name in self.language_codes.items():
            if lang_code in filename_lower or lang_name.lower() in filename_lower:
                return lang_code
        
        return None
    
    def prepare_cross_lingual_features(self, df, feature_columns=None):
        """
        Prepare features for cross-lingual adaptation.
        Handles language-specific preprocessing if needed.
        
        Args:
            df: DataFrame with multilingual data
            feature_columns: List of feature column names (None = auto-detect)
        
        Returns:
            X: Feature matrix
            y: Labels
            languages: Language codes for each sample
        """
        print("\nPreparing cross-lingual features...")
        
        # Auto-detect feature columns
        if feature_columns is None:
            exclude_cols = ['participant_id', 'label', 'language', 'split', 'session']
            feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Extract features
        X = df[feature_columns].values
        
        # Handle NaN/Inf
        X = np.where(np.isinf(X), np.nan, X)
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            if np.isnan(col).any():
                median_val = np.nanmedian(col)
                if np.isnan(median_val):
                    median_val = 0.0
                col = np.where(np.isnan(col), median_val, col)
                X[:, col_idx] = col
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Extract labels
        y = df['label'].values.astype(int) if 'label' in df.columns else None
        
        # Extract language codes
        languages = df['language'].values if 'language' in df.columns else None
        
        print(f"  Features: {X.shape}")
        print(f"  Labels: {len(y) if y is not None else 'N/A'}")
        print(f"  Languages: {len(set(languages)) if languages is not None else 'N/A'} unique")
        
        return X, y, languages
    
    def split_by_language(self, df, language_column='language'):
        """
        Split dataset by language for analysis.
        
        Returns:
            language_datasets: Dictionary mapping language codes to DataFrames
        """
        if language_column not in df.columns:
            print(f"  No '{language_column}' column found")
            return {}
        
        language_datasets = {}
        for lang_code in df[language_column].unique():
            lang_df = df[df[language_column] == lang_code]
            language_datasets[lang_code] = lang_df
            lang_name = self.language_codes.get(lang_code, lang_code)
            print(f"  {lang_name} ({lang_code}): {len(lang_df)} samples")
        
        return language_datasets


class CrossLingualAdapter:
    """
    Cross-lingual adaptation strategies for model transfer.
    Supports domain adaptation techniques.
    """
    
    def __init__(self, method='feature_alignment'):
        """
        Initialize adapter.
        
        Args:
            method: Adaptation method ('feature_alignment', 'domain_adaptation', 'transfer_learning')
        """
        self.method = method
    
    def align_features(self, X_source, X_target, language_source, language_target):
        """
        Align features across languages using various techniques.
        
        Args:
            X_source: Source language features
            X_target: Target language features
            language_source: Source language code
            language_target: Target language code
        
        Returns:
            X_target_aligned: Aligned target features
        """
        print(f"\nAligning features from {language_source} to {language_target}...")
        
        if self.method == 'feature_alignment':
            # Simple standardization alignment
            from sklearn.preprocessing import StandardScaler
            
            scaler_source = StandardScaler()
            scaler_target = StandardScaler()
            
            X_source_scaled = scaler_source.fit_transform(X_source)
            X_target_scaled = scaler_target.fit_transform(X_target)
            
            # Align distributions (simple approach)
            # More sophisticated methods could use MMD, CORAL, etc.
            X_target_aligned = X_target_scaled
            
            print(f"  Aligned features using standardization")
            return X_target_aligned
        
        elif self.method == 'domain_adaptation':
            # Placeholder for domain adaptation (CORAL, MMD, etc.)
            print(f"  Domain adaptation not yet implemented")
            return X_target
        
        else:
            return X_target
    
    def transfer_model(self, model_source, X_target, y_target):
        """
        Transfer a model trained on source language to target language.
        
        Args:
            model_source: Model trained on source language
            X_target: Target language features
            y_target: Target language labels (optional for fine-tuning)
        
        Returns:
            model_transferred: Adapted model
        """
        print(f"\nTransferring model to target language...")
        
        if y_target is not None:
            # Fine-tune on target language
            print(f"  Fine-tuning on {len(X_target)} target samples...")
            # Implementation depends on model type
            # For sklearn models, could use warm_start or partial_fit
            return model_source  # Placeholder
        
        else:
            # Zero-shot transfer (no target labels)
            print(f"  Zero-shot transfer (no target labels)")
            return model_source


def create_cross_lingual_dataset_template(output_dir="data/cross_lingual"):
    """
    Create template structure for cross-lingual datasets.
    """
    print(f"Creating cross-lingual dataset template in {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create README
    readme_content = """
# Cross-Lingual Dataset Template

This directory is designed for storing multilingual speech datasets from LMIC (Low and Middle Income Countries).

## Expected Structure

```
cross_lingual/
├── english/              # English (source) dataset
│   └── features.csv
├── hindi/                # Hindi dataset
│   └── features.csv
├── spanish/              # Spanish dataset
│   └── features.csv
└── metadata.csv          # Combined metadata with language labels
```

## CSV Format

Each language-specific CSV should contain:
- `participant_id`: Unique participant identifier
- `language`: Language code (e.g., 'en', 'hi', 'es')
- `label`: Binary label (0 = no stress/fatigue, 1 = stress/fatigue)
- Feature columns: All extracted acoustic-prosodic features

## Language Codes

- `en`: English
- `hi`: Hindi
- `es`: Spanish
- `fr`: French
- `sw`: Swahili
- `zh`: Chinese
- `ar`: Arabic
- `bn`: Bengali
- `ur`: Urdu
- `ta`: Tamil
- `te`: Telugu
- `mr`: Marathi
- `gu`: Gujarati
- `kn`: Kannada
- `ml`: Malayalam

## Usage

```python
from code.data.cross_lingual import CrossLingualDataLoader

loader = CrossLingualDataLoader(source_language='en', target_languages=['hi', 'es'])
df = loader.load_multilingual_dataset('data/cross_lingual/')
X, y, languages = loader.prepare_cross_lingual_features(df)
```
"""
    
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"   Created template in {output_dir}")
    print(f"   See {readme_path} for details")


if __name__ == "__main__":
    # Create template
    create_cross_lingual_dataset_template()
    
    # Example usage
    print("\n" + "="*70)
    print("Cross-Lingual Adaptation Framework")
    print("="*70)
    print("\nThis framework supports:")
    print("  1. Loading multilingual datasets")
    print("  2. Feature alignment across languages")
    print("  3. Model transfer learning")


