import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from code.feature_extraction.build_features import FeatureBuilder

if __name__ == "__main__":
    fb = FeatureBuilder()
    fb.build_all(aggregate=True)  # CRITICAL: aggregate=True for >80% accuracy