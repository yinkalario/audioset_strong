#!/usr/bin/env python3
"""
AudioSet Weak Label Distribution Analyzer

This script provides comprehensive analysis of AudioSet weak labeling datasets,
focusing on label occurrence frequency and distribution patterns across different
dataset splits. Weak labels provide segment-level annotations without precise
temporal boundaries, making occurrence counting the primary analysis metric.

Features:
- Occurrence-based label distribution analysis (10-second segments)
- Multi-dataset comparison (unbalanced_train, balanced_train, eval)
- Individual dataset analysis with separate visualizations
- Combined cross-dataset distribution analysis
- Statistical export and reporting capabilities
- Label ID to display name mapping integration

Analysis Metrics:
- Label occurrence frequency across datasets
- Relative distribution percentages
- Cross-dataset label coverage comparison
- Top-N label identification and ranking
- Dataset-specific distribution patterns

Visualization Components:
- 4-panel comprehensive analysis per dataset
- Cross-dataset comparison visualizations
- Top label frequency charts
- Distribution pattern analysis
- Individual dataset breakdowns

Dataset Coverage:
- unbalanced_train_segments.csv (~2M segments)
- balanced_train_segments.csv (~22K segments)
- eval_segments.csv (~20K segments)

Usage:
    python analyze_weak_label_distribution.py --unbalanced-train file1.csv
           --balanced-train file2.csv --eval-file file3.csv --output-dir out

Author: Yin Cao
Date: 2025
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


def create_label_distribution_analysis(all_label_counts: dict, label_mapping: dict,
                                     output_dir: str = 'out'):
    """Create comprehensive label distribution analysis plots for multiple datasets."""

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Combine all labels from all datasets
    all_labels = set()
    for dataset_counts in all_label_counts.values():
        all_labels.update(dataset_counts.keys())

    # Create comparison DataFrame
    comparison_data = []
    for label in all_labels:
        row = {'label_id': label}

        # Add display name
        if label in label_mapping:
            row['display_name'] = f"{label_mapping[label][:30]}..."
        else:
            row['display_name'] = label

        # Add counts for each dataset
        total_count = 0
        for dataset_name, dataset_counts in all_label_counts.items():
            count = dataset_counts.get(label, 0)
            row[f'{dataset_name}_count'] = count
            total_count += count

        row['total_count'] = total_count
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data).sort_values('total_count', ascending=False)
    
    # Calculate statistics for each dataset
    dataset_names = [name for name in all_label_counts.keys()]

    print(f"\nWeak Label Distribution Statistics:")
    print(f"Total unique labels across all datasets: {len(df):,}")

    for dataset_name in dataset_names:
        dataset_total = sum(all_label_counts[dataset_name].values())
        dataset_labels = len(all_label_counts[dataset_name])
        print(f"\n{dataset_name.replace('_', ' ').title()}:")
        print(f"  Total segments: {dataset_total:,}")
        print(f"  Unique labels: {dataset_labels:,}")
        if dataset_labels > 0:
            print(f"  Average occurrences per label: {dataset_total/dataset_labels:.1f}")
        else:
            print(f"  Average occurrences per label: N/A (no labels)")

    print(f"\nOverall most frequent label: {df.iloc[0]['display_name']} ({df.iloc[0]['total_count']:,} total occurrences)")
    print(f"Overall least frequent label: {df.iloc[-1]['display_name']} ({df.iloc[-1]['total_count']:,} total occurrences)")
    
    # Create comprehensive analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('AudioSet Weak Label Distribution Analysis (Multi-Dataset)', fontsize=16, fontweight='bold')

    # 1. Top 20 labels by total count across all datasets
    top_20 = df.head(20)

    # Create stacked bar chart for top 20 labels
    bottom = np.zeros(len(top_20))
    colors = plt.cm.Set3(np.linspace(0, 1, len(dataset_names)))

    for i, dataset_name in enumerate(dataset_names):
        counts = top_20[f'{dataset_name}_count'].values
        axes[0, 0].barh(range(len(top_20)), counts, left=bottom,
                        label=dataset_name.replace('_', ' ').title(), color=colors[i])
        bottom += counts

    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20['display_name'], fontsize=8)
    axes[0, 0].set_xlabel('Occurrence Count')
    axes[0, 0].set_title('Top 20 Labels by Total Occurrence Count')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].invert_yaxis()
    
    # 2. Dataset comparison for top 10 labels
    top_10 = df.head(10)
    x_pos = np.arange(len(top_10))
    width = 0.25

    for i, dataset_name in enumerate(dataset_names):
        counts = top_10[f'{dataset_name}_count'].values
        axes[0, 1].bar(x_pos + i * width, counts, width,
                       label=dataset_name.replace('_', ' ').title(), color=colors[i])

    axes[0, 1].set_xlabel('Labels')
    axes[0, 1].set_ylabel('Occurrence Count')
    axes[0, 1].set_title('Top 10 Labels Comparison Across Datasets')
    axes[0, 1].set_xticks(x_pos + width)
    axes[0, 1].set_xticklabels([name[:15] + '...' if len(name) > 15 else name
                                for name in top_10['display_name']], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # 3. Total counts by dataset
    dataset_totals = []
    for dataset_name in dataset_names:
        total = sum(all_label_counts[dataset_name].values())
        dataset_totals.append(total)

    axes[1, 0].bar(dataset_names, dataset_totals, color=colors[:len(dataset_names)])
    axes[1, 0].set_ylabel('Total Segments')
    axes[1, 0].set_title('Total Segments by Dataset')
    axes[1, 0].set_xticklabels([name.replace('_', ' ').title() for name in dataset_names])

    # 4. Label distribution comparison
    total_all = df['total_count'].sum()
    top_10_percentage = (top_10['total_count'] / total_all) * 100
    axes[1, 1].pie(top_10_percentage, labels=[name[:15] + '...' if len(name) > 15 else name
                                              for name in top_10['display_name']],
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    axes[1, 1].set_title('Top 10 Labels Distribution (%)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/weak_label_distribution_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive analysis to {output_dir}/weak_label_distribution_analysis.png")
    
    # Create detailed top labels plot
    plt.figure(figsize=(15, 10))
    top_50 = df.head(50)

    plt.barh(range(len(top_50)), top_50['total_count'], color='skyblue', alpha=0.8)
    plt.yticks(range(len(top_50)), top_50['display_name'], fontsize=8)
    plt.xlabel('Total Occurrence Count')
    plt.title('Top 50 AudioSet Weak Labels by Total Occurrence Count', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)

    # Add count labels on bars
    for i, count in enumerate(top_50['total_count']):
        plt.text(count + max(top_50['total_count']) * 0.01, i, f'{count:,}',
                 va='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_weak_labels_detailed.png', dpi=300, bbox_inches='tight')
    print(f"Saved detailed top labels plot to {output_dir}/top_weak_labels_detailed.png")

    # Save statistics to CSV
    total_all_segments = df['total_count'].sum()
    df['percentage'] = (df['total_count'] / total_all_segments) * 100
    df['rank'] = range(1, len(df) + 1)
    df.to_csv(f'{output_dir}/weak_label_distribution_stats.csv', index=False)
    print(f"Saved statistics to {output_dir}/weak_label_distribution_stats.csv")

    return df


def create_individual_analysis(label_counts_dict: dict, label_mapping: dict,
                              output_dir: str, dataset_name: str):
    """Create analysis for a single dataset."""

    # Get the single dataset counts
    dataset_counts = list(label_counts_dict.values())[0]

    # Convert to DataFrame
    labels = list(dataset_counts.keys())
    counts = list(dataset_counts.values())

    # Add display names
    display_names = []
    for label in labels:
        if label in label_mapping:
            display_names.append(f"{label_mapping[label][:30]}...")
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

    print(f"\n{dataset_name.replace('_', ' ').title()} Dataset Statistics:")
    print(f"  Total segments: {total_segments:,}")
    print(f"  Unique labels: {total_labels:,}")
    print(f"  Average occurrences per label: {total_segments/total_labels:.1f}")
    print(f"  Most frequent: {df.iloc[0]['display_name']} ({df.iloc[0]['count']:,})")
    print(f"  Least frequent: {df.iloc[-1]['display_name']} ({df.iloc[-1]['count']:,})")

    # Create individual analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{dataset_name.replace("_", " ").title()} Dataset - Weak Label Distribution',
                 fontsize=16, fontweight='bold')

    # 1. Top 20 labels
    top_20 = df.head(20)
    axes[0, 0].barh(range(len(top_20)), top_20['count'], color='skyblue')
    axes[0, 0].set_yticks(range(len(top_20)))
    axes[0, 0].set_yticklabels(top_20['display_name'], fontsize=8)
    axes[0, 0].set_xlabel('Occurrence Count')
    axes[0, 0].set_title('Top 20 Labels')
    axes[0, 0].invert_yaxis()

    # 2. Distribution histogram
    axes[0, 1].hist(counts, bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
    axes[0, 1].set_xlabel('Occurrence Count')
    axes[0, 1].set_ylabel('Number of Labels')
    axes[0, 1].set_title('Label Count Distribution')
    axes[0, 1].set_yscale('log')

    # 3. Top 10 pie chart
    top_10 = df.head(10)
    top_10_percentage = (top_10['count'] / total_segments) * 100
    axes[1, 0].pie(top_10_percentage, labels=[name[:15] + '...' if len(name) > 15 else name
                                              for name in top_10['display_name']],
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    axes[1, 0].set_title('Top 10 Labels Distribution (%)')

    # 4. Frequency bands
    frequency_bands = {
        'Very High (>1k)': sum(1 for c in counts if c > 1000),
        'High (100-1k)': sum(1 for c in counts if 100 <= c <= 1000),
        'Medium (10-100)': sum(1 for c in counts if 10 <= c < 100),
        'Low (<10)': sum(1 for c in counts if c < 10)
    }

    band_names = list(frequency_bands.keys())
    band_counts = list(frequency_bands.values())

    axes[1, 1].bar(band_names, band_counts, color=['red', 'orange', 'yellow', 'lightblue'])
    axes[1, 1].set_ylabel('Number of Labels')
    axes[1, 1].set_title('Labels by Frequency Bands')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/weak_label_distribution_analysis_{dataset_name}.png',
                dpi=300, bbox_inches='tight')
    print(f"  Saved individual analysis to {output_dir}/weak_label_distribution_analysis_{dataset_name}.png")

    # Save individual statistics
    df['percentage'] = (df['count'] / total_segments) * 100
    df['rank'] = range(1, len(df) + 1)
    df.to_csv(f'{output_dir}/weak_label_distribution_stats_{dataset_name}.csv', index=False)
    print(f"  Saved individual stats to {output_dir}/weak_label_distribution_stats_{dataset_name}.csv")

    plt.close()
    return df


def main():
    """Main function to analyze weak label distribution."""
    parser = argparse.ArgumentParser(
        description='Analyze weak label distribution in AudioSet weak labeling data'
    )
    parser.add_argument('--unbalanced-train', default='meta/unbalanced_train_segments.csv',
                        help='Path to unbalanced training segments file')
    parser.add_argument('--balanced-train', default='meta/balanced_train_segments.csv',
                        help='Path to balanced training segments file')
    parser.add_argument('--eval-file', default='meta/eval_segments.csv',
                        help='Path to evaluation segments file')
    parser.add_argument('--mid-to-display', default='meta/mid_to_display_name.tsv',
                        help='Path to label ID to display name mapping file')
    parser.add_argument('--output-dir', default='out',
                        help='Output directory for plots and statistics')

    args = parser.parse_args()

    print("Analyzing AudioSet weak label distribution...")
    print(f"Unbalanced train file: {args.unbalanced_train}")
    print(f"Balanced train file: {args.balanced_train}")
    print(f"Eval file: {args.eval_file}")
    print(f"Label mapping: {args.mid_to_display}")
    print(f"Output directory: {args.output_dir}")

    # Define input files
    input_files = {
        'unbalanced_train': args.unbalanced_train,
        'balanced_train': args.balanced_train,
        'eval': args.eval_file
    }

    # Load label mapping
    print("\nLoading label mapping...")
    label_mapping = load_label_mapping(args.mid_to_display)
    print(f"Loaded {len(label_mapping):,} label mappings")

    # Process each dataset
    all_label_counts = {}
    for dataset_name, file_path in input_files.items():
        if Path(file_path).exists() and file_path != "/dev/null":
            print(f"\nLoading {dataset_name} data...")
            df = load_weak_segments_data(file_path)
            print(f"Loaded {len(df):,} segments from {dataset_name}")

            if len(df) > 0:
                print(f"Extracting and counting labels from {dataset_name}...")
                label_counts = extract_labels_from_segments(df)
                all_label_counts[dataset_name] = label_counts
            else:
                print(f"Warning: No data in {dataset_name}")
        else:
            print(f"Warning: File not found or skipped: {file_path}")

    if not all_label_counts:
        print("Error: No valid input files found!")
        return

    # Create combined analysis
    print("\nCreating combined distribution analysis...")
    combined_stats = create_label_distribution_analysis(all_label_counts, label_mapping, args.output_dir)

    # Create separate analysis for each dataset
    print("\nCreating individual dataset analyses...")
    for dataset_name, label_counts in all_label_counts.items():
        print(f"Analyzing {dataset_name} separately...")

        # Create individual analysis
        individual_counts = {dataset_name: label_counts}
        individual_stats = create_individual_analysis(individual_counts, label_mapping,
                                                    args.output_dir, dataset_name)
        print(f"Completed individual analysis for {dataset_name}")

    print("\nAnalysis complete!")
    print(f"Results saved to {args.output_dir}/")
    print("Generated files:")
    print("- Combined analysis: weak_label_distribution_analysis.png")
    print("- Individual analyses: weak_label_distribution_analysis_{dataset}.png")
    print("- Combined stats: weak_label_distribution_stats.csv")
    print("- Individual stats: weak_label_distribution_stats_{dataset}.csv")


if __name__ == "__main__":
    main()
