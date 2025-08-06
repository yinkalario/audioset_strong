#!/usr/bin/env python3
"""
AudioSet Weak Label Negative Sample Generator

This script generates negative metadata for target sound types by extracting 10-second
audio segments that do NOT contain the target sounds from AudioSet weak labeling files.
Weak labels provide segment-level annotations without precise temporal boundaries.

Features:
- Processes AudioSet weak labeling CSV files (unbalanced_train, balanced_train, eval)
- Identifies 10-second segments without target sound labels
- Extracts negative samples for balanced machine learning training
- Handles multiple target labels per sound type
- Generates organized negative metadata for each target sound type

Process:
1. Load AudioSet weak labeling CSV files
2. Parse positive_labels column to identify target sounds
3. Extract segments that do NOT contain any target labels
4. Save negative samples with segment information

Input Files:
- unbalanced_train_segments.csv (2M+ segments)
- balanced_train_segments.csv (~22K segments)
- eval_segments.csv (~20K segments)

Output Structure:
- meta/{target_name}/raw/neg_weak/{target_name}_{dataset}_segments.csv

Usage:
    python generate_raw_neg_weak_meta.py --input-dir meta --output-dir meta

Author: Yin Cao
Date: 2025
"""

import pandas as pd
from typing import List, Set
import argparse
from pathlib import Path

# Target labels and names configuration
target_labels = ['/t/dd00002']  # Baby cry, infant cry
target_name = 'baby_cry'

# target_labels = ['/m/01d3sd', '/m/07q0yl5']  # Snoring, Snort
# target_name = 'snore'

# target_labels = ['/m/032s66', '/m/04zjc', '/m/073cg4']  # Gunshot/gunfire, Machine gun, Cap gun
# target_name = 'gun'


def load_weak_segments_data(file_path: str) -> pd.DataFrame:
    """Load weak segments CSV data from file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load CSV file with proper column names
    # Format: YTID, start_seconds, end_seconds, positive_labels
    # Note: positive_labels are quoted and may contain commas
    try:
        df = pd.read_csv(file_path,
                         names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                         skiprows=3,  # Skip the header comments
                         quotechar='"',  # Handle quoted fields properly
                         skipinitialspace=True,  # Skip spaces after delimiter
                         on_bad_lines='skip')  # Skip problematic lines
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        # Try alternative parsing
        df = pd.read_csv(file_path,
                         names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                         skiprows=3,
                         sep=', ',  # Use comma-space as separator
                         quotechar='"',
                         engine='python')  # Use Python engine for more flexibility

    return df


def get_segments_with_target_labels(df: pd.DataFrame, target_labels: List[str]) -> Set[str]:
    """Get all segment IDs (YTID) that contain any of the target labels in positive_labels."""
    target_segments = set()
    
    for _, row in df.iterrows():
        ytid = row['YTID']
        positive_labels = str(row['positive_labels'])
        
        # Check if any target label is in the positive_labels string
        for target_label in target_labels:
            if target_label in positive_labels:
                target_segments.add(ytid)
                break  # Found one target label, no need to check others for this segment
    
    return target_segments


def extract_negative_weak_samples(df: pd.DataFrame, target_labels: List[str]) -> pd.DataFrame:
    """Extract all samples from segments that do NOT contain target labels."""
    # Get segments that contain target labels
    positive_segments = get_segments_with_target_labels(df, target_labels)
    
    # Filter out all segments that contain target labels
    negative_df = df[~df['YTID'].isin(positive_segments)].copy()
    
    print(f"  Total segments in dataset: {len(df)}")
    print(f"  Segments with target labels: {len(positive_segments)}")
    print(f"  Negative segments: {len(negative_df)}")
    
    return negative_df


def save_weak_metadata(df: pd.DataFrame, output_path):
    """Save weak metadata to CSV file."""
    if df.empty:
        print(f"  No data to save to {output_path}")
        return

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save CSV with header
    df.to_csv(output_path, index=False, 
              columns=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'])

    print(f"  Saved {len(df)} negative weak segments to {output_path}")


def process_file(input_file: str, target_labels: List[str], target_name: str, output_dir: str):
    """Process a single input file and generate negative weak metadata."""
    print(f"Processing {input_file} for negative weak labels against {target_labels} ({target_name})")

    # Load data
    df = load_weak_segments_data(input_file)

    if df.empty:
        print(f"  No data found in {input_file}")
        return

    # Extract negative samples
    negative_df = extract_negative_weak_samples(df, target_labels)

    # Determine file type and generate output filename
    base_name = Path(input_file).stem
    
    # Create target-specific output directory with raw/neg_weak structure
    target_output_dir = Path(output_dir) / target_name / 'raw' / 'neg_weak'

    # Generate output path with specific naming convention
    output_filename = f"{target_name}_neg_weak_{base_name}.csv"
    output_path = target_output_dir / output_filename

    # Save results
    save_weak_metadata(negative_df, output_path)


def main():
    """Main function to process AudioSet weak labeling files."""
    parser = argparse.ArgumentParser(
        description='Generate raw negative weak metadata for target sound types'
    )
    parser.add_argument('--input-dir', default='meta',
                        help='Input directory containing AudioSet files')
    parser.add_argument('--output-dir', default='meta',
                        help='Output directory for generated metadata')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Input files to process (weak labeling versions)
    input_files = [
        'unbalanced_train_segments.csv',
        'balanced_train_segments.csv',
        'eval_segments.csv'
    ]

    print("Generating raw negative weak metadata...")
    print(f"Target labels: {target_labels}")
    print(f"Target name: {target_name}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    print(f"Processing negative weak samples for: {target_labels} ({target_name})")

    # Process each input file
    for input_file in input_files:
        input_path = Path(input_dir) / input_file
        if input_path.exists():
            process_file(str(input_path), target_labels, target_name, output_dir)
        else:
            print(f"Warning: Input file not found: {input_path}")

    print()
    print("Weak negative processing complete!")


if __name__ == "__main__":
    main()
