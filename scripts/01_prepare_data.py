import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from code.data.loader import DAICWOZLoader

def main():
    loader = DAICWOZLoader()
    metadata = loader.build_metadata()
    print(metadata.head())

if __name__ == "__main__":
    main()