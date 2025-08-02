#!/usr/bin/env python3
"""
Analyze weak label distribution in AudioSet weak labeling data.

This script analyzes the distribution of labels in AudioSet weak labeling files
by counting label occurrence frequency (since weak labels don't have precise timing).
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import argparse
from pathlib import Path


def load_weak_segments_data(file_path: str) -> pd.DataFrame:
    """Load weak segments CSV data from file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load CSV file with proper column names
    # Format: YTID, start_seconds, end_seconds, positive_labels
    df = pd.read_csv(file_path, 
                     names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
                     skiprows=3,  # Skip the header comments
                     quotechar='"',  # Handle quoted fields properly
                     skipinitialspace=True,  # Skip spaces after delimiter
                     on_bad_lines='skip')  # Skip problematic lines
    
    return df


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


def extract_labels_from_segments(df: pd.DataFrame) -> Counter:
    """Extract and count all labels from the positive_labels column."""
    label_counts = Counter()
    
    for _, row in df.iterrows():
        positive_labels = str(row['positive_labels'])
        
        # Remove quotes and split by comma
        labels = [label.strip().strip('"') for label in positive_labels.split(',')]
        
        # Count each label
        for label in labels:
            if label and label != 'nan':  # Skip empty or nan labels
                label_counts[label] += 1
    
    return label_counts


def create_label_distribution_analysis(label_counts: Counter, label_mapping: dict, 
                                     output_dir: str = 'out'):
    """Create comprehensive label distribution analysis plots."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Convert to DataFrame for easier manipulation
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    
    # Add display names
    display_names = []
    for label in labels:
        if label in label_mapping:
            display_names.append(f"{label_mapping[label][:30]}...")  # Truncate long names
        else:
            display_names.append(label)
    
    df = pd.DataFrame({
        'label_id': labels,
        'display_name': display_names,
        'count': counts
    }).sort_values('count', ascending=False)
    
    # Calculate statistics
    total_segments = sum(counts)
    total_labels = len(labels)
    
    print(f"\nWeak Label Distribution Statistics:")
    print(f"Total segments: {total_segments:,}")
    print(f"Total unique labels: {total_labels:,}")
    print(f"Average occurrences per label: {total_segments/total_labels:.1f}")
    print(f"Most frequent label: {df.iloc[0]['display_name']} ({df.iloc[0]['count']:,} occurrences)")
    print(f"Least frequent label: {df.iloc[-1]['display_name']} ({df.iloc[-1]['count']:,} occurrences)")
    
    # Create comprehensive analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('AudioSet Weak Label Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Top 20 labels by count
    top_20 = df.head(20)
    axes[0, 0].barh(range(len(top_20)), top_20['count'])
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20['display_name'], fontsize=8)
    axes[0, 0].set_xlabel('Occurrence Count')
    axes[0, 0].set_title('Top 20 Labels by Occurrence Count')
    axes[0, 0].invert_yaxis()
    
    # 2. Distribution histogram
    axes[0, 1].hist(counts, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Occurrence Count')
    axes[0, 1].set_ylabel('Number of Labels')
    axes[0, 1].set_title('Distribution of Label Occurrence Counts')
    axes[0, 1].set_yscale('log')
    
    # 3. Cumulative percentage
    cumulative_counts = np.cumsum(df['count'])
    cumulative_percentage = (cumulative_counts / total_segments) * 100
    axes[0, 2].plot(range(1, len(cumulative_percentage) + 1), cumulative_percentage)
    axes[0, 2].set_xlabel('Label Rank')
    axes[0, 2].set_ylabel('Cumulative Percentage (%)')
    axes[0, 2].set_title('Cumulative Label Coverage')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Log-scale distribution
    axes[1, 0].bar(range(len(top_20)), top_20['count'])
    axes[1, 0].set_xticks(range(len(top_20)))
    axes[1, 0].set_xticklabels(top_20['display_name'], rotation=45, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('Occurrence Count (log scale)')
    axes[1, 0].set_title('Top 20 Labels (Log Scale)')
    axes[1, 0].set_yscale('log')
    
    # 5. Percentage distribution (top 20)
    top_20_percentage = (top_20['count'] / total_segments) * 100
    axes[1, 1].pie(top_20_percentage[:10], labels=top_20['display_name'][:10], 
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    axes[1, 1].set_title('Top 10 Labels Distribution (%)')
    
    # 6. Frequency bands
    frequency_bands = {
        'Very High (>10k)': sum(1 for c in counts if c > 10000),
        'High (1k-10k)': sum(1 for c in counts if 1000 <= c <= 10000),
        'Medium (100-1k)': sum(1 for c in counts if 100 <= c < 1000),
        'Low (10-100)': sum(1 for c in counts if 10 <= c < 100),
        'Very Low (<10)': sum(1 for c in counts if c < 10)
    }
    
    band_names = list(frequency_bands.keys())
    band_counts = list(frequency_bands.values())
    
    axes[1, 2].bar(band_names, band_counts, color=['red', 'orange', 'yellow', 'lightblue', 'lightgray'])
    axes[1, 2].set_ylabel('Number of Labels')
    axes[1, 2].set_title('Labels by Frequency Bands')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weak_label_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive analysis to {output_dir}/weak_label_distribution_analysis.png")
    
    # Create detailed top labels plot
    plt.figure(figsize=(15, 10))
    top_50 = df.head(50)
    
    plt.barh(range(len(top_50)), top_50['count'], color='skyblue', alpha=0.8)
    plt.yticks(range(len(top_50)), top_50['display_name'], fontsize=8)
    plt.xlabel('Occurrence Count')
    plt.title('Top 50 AudioSet Weak Labels by Occurrence Count', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    
    # Add count labels on bars
    for i, count in enumerate(top_50['count']):
        plt.text(count + max(top_50['count']) * 0.01, i, f'{count:,}', 
                va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_weak_labels_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Saved detailed top labels plot to {output_dir}/top_weak_labels_detailed.png")
    
    # Save statistics to CSV
    df['percentage'] = (df['count'] / total_segments) * 100
    df['rank'] = range(1, len(df) + 1)
    df.to_csv(f'{output_dir}/weak_label_distribution_stats.csv', index=False)
    print(f"Saved statistics to {output_dir}/weak_label_distribution_stats.csv")
    
    return df


def main():
    """Main function to analyze weak label distribution."""
    parser = argparse.ArgumentParser(
        description='Analyze weak label distribution in AudioSet weak labeling data'
    )
    parser.add_argument('--train-file', default='meta/unbalanced_train_segments.csv',
                        help='Path to weak training segments file')
    parser.add_argument('--mid-to-display', default='meta/mid_to_display_name.tsv',
                        help='Path to label ID to display name mapping file')
    parser.add_argument('--output-dir', default='out',
                        help='Output directory for plots and statistics')
    
    args = parser.parse_args()
    
    print("Analyzing AudioSet weak label distribution...")
    print(f"Input file: {args.train_file}")
    print(f"Label mapping: {args.mid_to_display}")
    print(f"Output directory: {args.output_dir}")
    
    # Load data
    print("\nLoading weak segments data...")
    df = load_weak_segments_data(args.train_file)
    print(f"Loaded {len(df):,} segments")
    
    # Load label mapping
    print("Loading label mapping...")
    label_mapping = load_label_mapping(args.mid_to_display)
    print(f"Loaded {len(label_mapping):,} label mappings")
    
    # Extract and count labels
    print("Extracting and counting labels...")
    label_counts = extract_labels_from_segments(df)
    
    # Create analysis
    print("Creating distribution analysis...")
    stats_df = create_label_distribution_analysis(label_counts, label_mapping, args.output_dir)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
