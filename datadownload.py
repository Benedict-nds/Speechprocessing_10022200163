#!/usr/bin/env python3
"""
Download and extract DAIC-WOZ participant data.

This script downloads participant zip files from the DAIC-WOZ dataset,
extracts them, and cleans up zip files to save disk space.
"""

import os
import sys
import urllib.request
import zipfile
import pandas as pd
from pathlib import Path

# Base URL for DAIC-WOZ participant data
BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"

def get_participant_id_column(df):
    """Dynamically find the participant ID column name."""
    possible_names = [
        'participant_id', 'Participant_ID', 'participant_ID', 
        'Participant_id', 'PID', 'pid', 'id', 'ID'
    ]
    
    for name in possible_names:
        if name in df.columns:
            return name
    
    # If not found, return first column
    return df.columns[0]

def download_participant(participant_id, base_url=BASE_URL, output_dir="data/raw"):
    """Download and extract a single participant's data."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct URLs
    zip_filename = f"{participant_id}.zip"
    zip_url = base_url + zip_filename
    zip_path = os.path.join(output_dir, zip_filename)
    extract_path = os.path.join(output_dir, str(participant_id))
    
    # Skip if already extracted
    if os.path.isdir(extract_path) or os.path.isdir(os.path.join(output_dir, f"{participant_id}_P")):
        print(f"  Participant {participant_id} already exists, skipping...")
        return True
    
    try:
        # Download zip file
        print(f"  Downloading {participant_id}...")
        urllib.request.urlretrieve(zip_url, zip_path)
        
        # Extract zip file
        print(f"  Extracting {participant_id}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Clean up zip file
        os.remove(zip_path)
        print(f"  Completed {participant_id}")
        return True
        
    except urllib.error.HTTPError as e:
        print(f"  Error downloading {participant_id}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False
    except Exception as e:
        print(f"  Error processing {participant_id}: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False

def cleanup_zip_files(data_dir="data/raw"):
    """Remove all .zip files from the data directory."""
    zip_files = list(Path(data_dir).glob("*.zip"))
    if zip_files:
        print(f"\nCleaning up {len(zip_files)} zip files...")
        for zip_file in zip_files:
            try:
                os.remove(zip_file)
                print(f"  Removed {zip_file.name}")
            except Exception as e:
                print(f"  Error removing {zip_file.name}: {e}")
    else:
        print("\nNo zip files to clean up.")

def main():
    """Main download function."""
    print("=" * 70)
    print("DAIC-WOZ Data Download")
    print("=" * 70)
    
    # Check for split CSV files
    split_files = {
        'dev': 'data/raw/dev_split_Depression_AVEC2017.csv',
        'test': 'data/raw/test_split_Depression_AVEC2017.csv'
    }
    
    # Try to find split files
    found_splits = []
    for split_name, file_path in split_files.items():
        if os.path.exists(file_path):
            found_splits.append((split_name, file_path))
        else:
            print(f"Warning: {file_path} not found. Skipping {split_name} split.")
    
    if not found_splits:
        print("\nError: No split CSV files found!")
        print("Expected files:")
        for _, path in split_files.items():
            print(f"  - {path}")
        print("\nPlease ensure split CSV files are in data/raw/ directory.")
        return
    
    # Load participant IDs from split files
    all_participants = []
    for split_name, file_path in found_splits:
        try:
            df = pd.read_csv(file_path)
            pid_col = get_participant_id_column(df)
            participants = df[pid_col].dropna().unique().tolist()
            all_participants.extend(participants)
            print(f"\nFound {len(participants)} participants in {split_name} split")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    # Remove duplicates
    all_participants = list(set(all_participants))
    print(f"\nTotal unique participants to download: {len(all_participants)}")
    
    # Download each participant
    print("\nStarting downloads...")
    successful = 0
    failed = 0
    
    for i, pid in enumerate(all_participants, 1):
        print(f"\n[{i}/{len(all_participants)}] Processing participant {pid}...")
        if download_participant(pid):
            successful += 1
        else:
            failed += 1
    
    # Clean up any remaining zip files
    cleanup_zip_files()
    
    # Summary
    print("\n" + "=" * 70)
    print("Download Summary")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(all_participants)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
