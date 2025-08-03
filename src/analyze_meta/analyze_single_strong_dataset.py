#!/usr/bin/env python3
"""
AudioSet Strong Label Dataset Analyzer

This script provides comprehensive analysis of individual AudioSet strong labeling
datasets, generating detailed visualizations and statistics for label distribution
analysis. It supports all AudioSet strong label formats including the framed_posneg
format with proper PRESENT/NOT_PRESENT filtering.

Features:
- Analyzes individual strong labeling TSV files separately
- Calculates label distributions by total duration (not event count)
- Handles framed_posneg format with PRESENT event filtering
- Generates comprehensive 6-panel analysis visualizations
- Creates detailed top-N label plots with duration information
- Exports complete statistics to CSV for further analysis
- Supports label ID to display name mapping

Analysis Components:
- Duration-based label distribution (primary metric)
- Top labels by total duration
- Cumulative duration coverage analysis
- Duration distribution histograms
- Label frequency band analysis
- Detailed top-N label visualization

Supported Formats:
- Standard strong labels: segment_id, start_time, end_time, label
- Framed posneg: segment_id, start_time, end_time, label, present

Usage:
    python analyze_single_strong_dataset.py --input-file dataset.tsv
           --output-prefix out/analysis --dataset-name "Train"

Author: Yin Cao
Date: 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse
from pathlib import Path


def load_label_mapping(file_path: str) -> dict:
    """Load label ID to display name mapping."""
    if not Path(file_path).exists():
        print(f"Warning: Label mapping file not found: {file_path}")
        return {}
    
    try:
        df = pd.read_csv(file_path, sep='\t', names=['label_id', 'display_name'], skiprows=1)
        return dict(zip(df['label_id'], df['display_name']))
    except Exception as e:
        print(f"Warning: Could not load label mapping: {e}")
        return {}


def load_strong_data(file_path: str) -> pd.DataFrame:
    """Load strong labeling TSV data."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if this is a framed_posneg file
    is_framed = 'framed_posneg' in file_path
    
    if is_framed:
        # Format: segment_id, start_time_seconds, end_time_seconds, label, present
        df = pd.read_csv(file_path, sep='\t', 
                         names=['segment_id', 'start_time_seconds', 'end_time_seconds', 'label', 'present'],
                         skiprows=1)
        # Filter for only PRESENT events
        df = df[df['present'] == 'PRESENT'].copy()
        print(f"Loaded {len(df)} PRESENT events from framed_posneg file")
    else:
        # Format: segment_id, start_time_seconds, end_time_seconds, label
        df = pd.read_csv(file_path, sep='\t',
                         names=['segment_id', 'start_time_seconds', 'end_time_seconds', 'label'],
                         skiprows=1)
        print(f"Loaded {len(df)} events from strong file")
    
    return df


def calculate_label_durations(df: pd.DataFrame) -> dict:
    """Calculate total duration for each label."""
    label_durations = defaultdict(float)
    
    for _, row in df.iterrows():
        duration = float(row['end_time_seconds']) - float(row['start_time_seconds'])
        label_durations[row['label']] += duration
    
    return dict(label_durations)


def create_analysis_plots(label_durations: dict, label_mapping: dict, 
                         output_prefix: str, dataset_name: str):
    """Create analysis plots for the dataset."""
    
    # Convert to sorted list
    sorted_labels = sorted(label_durations.items(), key=lambda x: x[1], reverse=True)
    
    # Prepare data
    labels = [item[0] for item in sorted_labels]
    durations = [item[1] for item in sorted_labels]
    
    # Add display names
    display_names = []
    for label in labels:
        if label in label_mapping:
            name = label_mapping[label]
            display_names.append(name[:30] + '...' if len(name) > 30 else name)
        else:
            display_names.append(label)
    
    # Calculate statistics
    total_duration = sum(durations)
    total_labels = len(labels)
    
    print(f"\n{dataset_name} Dataset Statistics:")
    print(f"  Total duration: {total_duration:.1f} seconds ({total_duration/3600:.1f} hours)")
    print(f"  Total unique labels: {total_labels}")
    print(f"  Average duration per label: {total_duration/total_labels:.1f} seconds")
    print(f"  Most frequent: {display_names[0]} ({durations[0]:.1f}s)")
    print(f"  Least frequent: {display_names[-1]} ({durations[-1]:.1f}s)")
    
    # Create comprehensive analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'{dataset_name} Strong Label Distribution Analysis (by Duration)', 
                 fontsize=16, fontweight='bold')
    
    # 1. Top 20 labels by duration
    top_20 = min(20, len(labels))
    axes[0, 0].barh(range(top_20), durations[:top_20], color='skyblue')
    axes[0, 0].set_yticks(range(top_20))
    axes[0, 0].set_yticklabels(display_names[:top_20], fontsize=8)
    axes[0, 0].set_xlabel('Duration (seconds)')
    axes[0, 0].set_title('Top 20 Labels by Duration')
    axes[0, 0].invert_yaxis()
    
    # 2. Duration distribution histogram
    axes[0, 1].hist(durations, bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[0, 1].set_xlabel('Duration (seconds)')
    axes[0, 1].set_ylabel('Number of Labels')
    axes[0, 1].set_title('Label Duration Distribution')
    axes[0, 1].set_yscale('log')
    
    # 3. Cumulative percentage
    cumulative_durations = np.cumsum(durations)
    cumulative_percentage = (cumulative_durations / total_duration) * 100
    axes[0, 2].plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage)
    axes[0, 2].set_xlabel('Label Rank')
    axes[0, 2].set_ylabel('Cumulative Percentage (%)')
    axes[0, 2].set_title('Cumulative Duration Coverage')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Top 10 pie chart
    top_10 = min(10, len(labels))
    top_10_percentage = [(dur / total_duration) * 100 for dur in durations[:top_10]]
    axes[1, 0].pie(top_10_percentage, labels=[name[:15] + '...' if len(name) > 15 else name
                                              for name in display_names[:top_10]],
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    axes[1, 0].set_title('Top 10 Labels Distribution (%)')
    
    # 5. Duration bands
    duration_bands = {
        'Very Long (>100s)': sum(1 for d in durations if d > 100),
        'Long (10-100s)': sum(1 for d in durations if 10 <= d <= 100),
        'Medium (1-10s)': sum(1 for d in durations if 1 <= d < 10),
        'Short (<1s)': sum(1 for d in durations if d < 1)
    }
    
    band_names = list(duration_bands.keys())
    band_counts = list(duration_bands.values())
    
    axes[1, 1].bar(band_names, band_counts, color=['red', 'orange', 'yellow', 'lightblue'])
    axes[1, 1].set_ylabel('Number of Labels')
    axes[1, 1].set_title('Labels by Duration Bands')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. Log scale top 20
    axes[1, 2].bar(range(top_20), durations[:top_20], color='coral')
    axes[1, 2].set_xticks(range(top_20))
    axes[1, 2].set_xticklabels([name[:10] + '...' if len(name) > 10 else name
                                for name in display_names[:top_20]], rotation=45, ha='right')
    axes[1, 2].set_ylabel('Duration (seconds, log scale)')
    axes[1, 2].set_title('Top 20 Labels (Log Scale)')
    axes[1, 2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  Saved analysis plot: {output_prefix}_analysis.png")
    
    # Create detailed plot
    plt.figure(figsize=(15, 10))
    top_50 = min(50, len(labels))
    
    plt.barh(range(top_50), durations[:top_50], color='skyblue', alpha=0.8)
    plt.yticks(range(top_50), display_names[:top_50], fontsize=8)
    plt.xlabel('Duration (seconds)')
    plt.title(f'Top 50 {dataset_name} Strong Labels by Duration', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    # Add duration labels on bars
    for i, duration in enumerate(durations[:top_50]):
        plt.text(duration + max(durations[:top_50]) * 0.01, i, f'{duration:.1f}s',
                 va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_detailed.png', dpi=300, bbox_inches='tight')
    print(f"  Saved detailed plot: {output_prefix}_detailed.png")
    
    # Save statistics
    stats_data = []
    for i, (label, duration) in enumerate(sorted_labels):
        display_name = label_mapping.get(label, label)
        percentage = (duration / total_duration) * 100
        stats_data.append({
            'rank': i + 1,
            'label_id': label,
            'display_name': display_name,
            'duration_seconds': round(duration, 3),
            'percentage': round(percentage, 6)
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(f'{output_prefix}_stats.csv', index=False)
    print(f"  Saved statistics: {output_prefix}_stats.csv")
    
    plt.close('all')
    return stats_df


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze single strong label dataset')
    parser.add_argument('--input-file', required=True,
                        help='Path to input TSV file')
    parser.add_argument('--output-prefix', required=True,
                        help='Output file prefix (e.g., out/strong_label_distribution_train)')
    parser.add_argument('--dataset-name', required=True,
                        help='Dataset name for titles (e.g., Train, Eval, Eval Framed)')
    parser.add_argument('--mid-to-display', default='meta/mid_to_display_name.tsv',
                        help='Path to label mapping file')
    
    args = parser.parse_args()
    
    print(f"Analyzing {args.dataset_name} dataset: {args.input_file}")
    
    # Load label mapping
    label_mapping = load_label_mapping(args.mid_to_display)
    print(f"Loaded {len(label_mapping)} label mappings")
    
    # Load data
    df = load_strong_data(args.input_file)
    
    if df.empty:
        print("No data to analyze!")
        return
    
    # Calculate durations
    label_durations = calculate_label_durations(df)
    
    # Create output directory
    Path(args.output_prefix).parent.mkdir(parents=True, exist_ok=True)
    
    # Create analysis
    create_analysis_plots(label_durations, label_mapping, args.output_prefix, args.dataset_name)
    
    print(f"Analysis complete for {args.dataset_name}!")


if __name__ == "__main__":
    main()
