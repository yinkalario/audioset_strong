#!/usr/bin/env python3
"""
Generate target sound metadata from AudioSet strong labels.

This script extracts metadata for specific target sound types from AudioSet strong label files,
separating them into overlapped and non-overlapped categories based on temporal overlap
with other sound events.
"""

import pandas as pd
from typing import List, Tuple, Dict
import argparse
from pathlib import Path

# Target labels and names configuration
target_labels = ['/m/032s66', '/m/04zjc', '/m/073cg4']  # Gunshot/gunfire, Machine gun, Cap gun
target_name = 'gun'

# target_labels = ['/m/01d3sd', '/m/07q0yl5']  # Snoring, Snort
# target_name = 'snore'

# target_labels = ['/m/032s66', '/m/04zjc', '/m/073cg4']  # Gunshot/gunfire, Machine gun, Cap gun
# target_name = 'gun'


def load_tsv_data(file_path: str) -> pd.DataFrame:
    """Load TSV data from file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine the number of columns based on the file
    if 'framed_posneg' in file_path:
        # This file has 5 columns: segment_id, start_time_seconds, end_time_seconds, label, present
        columns = ['segment_id', 'start_time_seconds', 'end_time_seconds', 'label', 'present']
    else:
        # Regular strong label files have 4 columns
        columns = ['segment_id', 'start_time_seconds', 'end_time_seconds', 'label']

    df = pd.read_csv(file_path, sep='\t', names=columns, skiprows=1)
    return df


def check_temporal_overlap(target_start: float, target_end: float,
                           other_start: float, other_end: float) -> bool:
    """Check if two time intervals overlap."""
    return not (target_end <= other_start or target_start >= other_end)


def categorize_target_events(df: pd.DataFrame,
                             target_labels: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    Categorize target sound events into overlapped and non-overlapped based on
    temporal overlap with other sound events in the same segment.

    Returns:
        Tuple of (non_overlapped_events, overlapped_events)
    """
    non_overlapped = []
    overlapped = []

    # Check if this is a framed_posneg file (has 'present' column)
    has_present_column = 'present' in df.columns

    # Group by segment_id to process each audio segment separately
    for segment_id, segment_data in df.groupby('segment_id'):
        # Get all target events in this segment (any of the target labels)
        if has_present_column:
            # For framed_posneg files, only get PRESENT target events
            target_events = segment_data[
                (segment_data['label'].isin(target_labels)) &
                (segment_data['present'] == 'PRESENT')
            ]
            # Get all other PRESENT events (non-target sounds that are present)
            other_events = segment_data[
                (~segment_data['label'].isin(target_labels)) &
                (segment_data['present'] == 'PRESENT')
            ]
        else:
            # For regular strong label files, get all target events
            target_events = segment_data[segment_data['label'].isin(target_labels)]
            # Get all non-target events in this segment
            other_events = segment_data[~segment_data['label'].isin(target_labels)]

        for _, target_event in target_events.iterrows():
            target_start = target_event['start_time_seconds']
            target_end = target_event['end_time_seconds']

            # Check if this target event overlaps with any other event
            has_overlap = False
            for _, other_event in other_events.iterrows():
                other_start = other_event['start_time_seconds']
                other_end = other_event['end_time_seconds']

                if check_temporal_overlap(target_start, target_end, other_start, other_end):
                    has_overlap = True
                    break

            # Convert to dictionary for easier handling
            event_dict = target_event.to_dict()

            if has_overlap:
                overlapped.append(event_dict)
            else:
                non_overlapped.append(event_dict)

    return non_overlapped, overlapped


def save_metadata(events: List[Dict], output_path, has_present_column: bool = False):
    """Save events to TSV file. Skip saving if no events."""
    if not events:
        print(f"No events to save for {output_path} - skipping file creation")
        return

    df = pd.DataFrame(events)

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

    print(f"Saved {len(events)} events to {output_path}")


def process_file(input_file: str, target_labels: List[str], target_name: str, output_dir: str):
    """Process a single input file and generate overlapped/non-overlapped metadata."""
    print(f"Processing {input_file} for target labels {target_labels} ({target_name})")

    # Load data
    df = load_tsv_data(input_file)

    # Check if any target label exists in the data
    found_labels = [label for label in target_labels if label in df['label'].values]
    if not found_labels:
        print(f"Warning: None of target labels {target_labels} found in {input_file}")
        return

    print(f"  Found labels: {found_labels}")

    # Categorize events
    non_overlapped, overlapped = categorize_target_events(df, target_labels)

    # Determine file type and generate output filenames
    base_name = Path(input_file).stem
    has_present_column = 'framed_posneg' in input_file

    # Create target-specific output directory with raw/pos structure
    target_output_dir = Path(output_dir) / target_name / 'raw' / 'pos'

    # Generate output paths
    nov_output = target_output_dir / f"{target_name}_nov_{base_name}.tsv"
    ov_output = target_output_dir / f"{target_name}_ov_{base_name}.tsv"

    # Save results
    save_metadata(non_overlapped, nov_output, has_present_column)
    save_metadata(overlapped, ov_output, has_present_column)

    print(f"  Non-overlapped events: {len(non_overlapped)}")
    print(f"  Overlapped events: {len(overlapped)}")


def main():
    """Main function to process all AudioSet strong label files."""
    parser = argparse.ArgumentParser(
        description='Generate target sound metadata from AudioSet strong labels'
    )
    parser.add_argument('--input-dir', default='meta',
                        help='Input directory containing TSV files')
    parser.add_argument('--output-dir', default='meta',
                        help='Output directory for generated metadata')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Input files to process
    input_files = [
        'audioset_train_strong.tsv',
        'audioset_eval_strong.tsv',
        'audioset_eval_strong_framed_posneg.tsv'
    ]

    print("Generating target sound metadata...")
    print(f"Target labels: {target_labels}")
    print(f"Target name: {target_name}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()

    print(f"Processing target: {target_labels} ({target_name})")

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
