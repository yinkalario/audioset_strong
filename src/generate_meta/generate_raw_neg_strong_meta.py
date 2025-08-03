#!/usr/bin/env python3
"""
Generate raw negative metadata for target sound types.

This script extracts negative labels (audio clips that do NOT contain target sounds)
from the original AudioSet strong labeling files and organizes them by sound type.
"""

import pandas as pd
from typing import List, Set
import argparse
from pathlib import Path

# Target labels and names configuration
target_labels = ['/m/01d3sd', '/m/07q0yl5']  # Snoring, Snort
target_name = 'snore'

# target_labels = ['/t/dd00002']  # Baby cry, infant cry
# target_name = 'baby_cry'

# target_labels = ['/m/032s66', '/m/04zjc', '/m/073cg4']  # Gunshot/gunfire, Machine gun, Cap gun
# target_name = 'gun'


def load_tsv_data(file_path: str) -> pd.DataFrame:
    """Load TSV data from file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine the number of columns based on the file
    if 'framed_posneg' in file_path:
        columns = ['segment_id', 'start_time_seconds', 'end_time_seconds', 'label', 'present']
    else:
        columns = ['segment_id', 'start_time_seconds', 'end_time_seconds', 'label']

    df = pd.read_csv(file_path, sep='\t', names=columns, skiprows=1)
    return df


def get_segments_with_target_labels(df: pd.DataFrame, target_labels: List[str]) -> Set[str]:
    """Get all segment IDs that contain any of the target labels."""
    has_present_column = 'present' in df.columns

    if has_present_column:
        # For framed_posneg files, only consider PRESENT target events
        target_segments = df[
            (df['label'].isin(target_labels)) &
            (df['present'] == 'PRESENT')
        ]['segment_id'].unique()
    else:
        # For regular strong label files, get all segments with target labels
        target_segments = df[df['label'].isin(target_labels)]['segment_id'].unique()

    return set(target_segments)


def extract_negative_samples(df: pd.DataFrame, target_labels: List[str]) -> pd.DataFrame:
    """Extract all samples from segments that do NOT contain target labels."""
    has_present_column = 'present' in df.columns

    # Get segments that contain target labels
    positive_segments = get_segments_with_target_labels(df, target_labels)

    # Filter out all events from segments that contain target labels
    negative_df = df[~df['segment_id'].isin(positive_segments)].copy()

    # For framed_posneg files, only keep PRESENT events (NOT_PRESENT is meaningless)
    if has_present_column:
        negative_df = negative_df[negative_df['present'] == 'PRESENT'].copy()

    print(f"  Total segments in dataset: {df['segment_id'].nunique()}")
    print(f"  Segments with target labels: {len(positive_segments)}")
    print(f"  Negative segments: {negative_df['segment_id'].nunique()}")
    print(f"  Total events in negative segments: {len(negative_df)}")

    return negative_df


def save_metadata(df: pd.DataFrame, output_path, has_present_column: bool = False):
    """Save metadata to TSV file."""
    if df.empty:
        print(f"  No data to save to {output_path}")
        return

    # Ensure proper column order
    if has_present_column:
        columns = ['segment_id', 'start_time_seconds', 'end_time_seconds', 'label', 'present']
    else:
        columns = ['segment_id', 'start_time_seconds', 'end_time_seconds', 'label']

    df = df[columns]

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save with header
    with open(output_path, 'w') as f:
        f.write('\t'.join(columns) + '\n')
        df.to_csv(f, sep='\t', index=False, header=False)

    print(f"  Saved {len(df)} negative events to {output_path}")


def process_file(input_file: str, target_labels: List[str], target_name: str, output_dir: str):
    """Process a single input file and generate negative metadata."""
    print(f"Processing {input_file} for negative labels against {target_labels} ({target_name})")

    # Load data
    df = load_tsv_data(input_file)

    if df.empty:
        print(f"  No data found in {input_file}")
        return

    # Extract negative samples
    negative_df = extract_negative_samples(df, target_labels)

    # Determine file type and generate output filename
    base_name = Path(input_file).stem
    has_present_column = 'framed_posneg' in input_file

    # Create target-specific output directory with raw/neg_strong structure
    target_output_dir = Path(output_dir) / target_name / 'raw' / 'neg_strong'

    # Generate output path
    output_filename = f"{target_name}_{base_name}.tsv"
    output_path = target_output_dir / output_filename

    # Save results
    save_metadata(negative_df, output_path, has_present_column)


def main():
    """Main function to process AudioSet strong labeling files."""
    parser = argparse.ArgumentParser(
        description='Generate raw negative metadata for target sound types'
    )
    parser.add_argument('--input-dir', default='meta',
                        help='Input directory containing AudioSet files')
    parser.add_argument('--output-dir', default='meta',
                        help='Output directory for generated metadata')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Input files to process (only strong labeled versions)
    input_files = [
        'audioset_train_strong.tsv',
        'audioset_eval_strong.tsv',
        'audioset_eval_strong_framed_posneg.tsv'
    ]

    print("Generating raw negative metadata...")
    print(f"Target labels: {target_labels}")
    print(f"Target name: {target_name}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    print(f"Processing negative samples for: {target_labels} ({target_name})")

    # Process each input file
    for input_file in input_files:
        input_path = Path(input_dir) / input_file
        if input_path.exists():
            process_file(str(input_path), target_labels, target_name, output_dir)
        else:
            print(f"Warning: Input file not found: {input_path}")

    print()
    print("Processing complete!")


if __name__ == "__main__":
    main()
