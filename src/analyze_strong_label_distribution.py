#!/usr/bin/env python3
"""
Analyze label distribution in AudioSet training data.

This script calculates the ratio of each label in audioset_train_strong.tsv
and generates visualization plots to show the class imbalance.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import argparse


def load_label_names(mid_to_display_path: str) -> dict:
    """Load mapping from MID to display names."""
    label_names = {}
    with open(mid_to_display_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and '\t' in line:
                mid, name = line.split('\t', 1)
                label_names[mid] = name
    return label_names


def analyze_label_distribution(train_file: str, mid_to_display_path: str):
    """Analyze and visualize label distribution in training data."""
    print("Loading training data...")

    # Load training data
    df = pd.read_csv(train_file, sep='\t',
                     names=['segment_id', 'start_time', 'end_time', 'label'],
                     skiprows=1)

    # Load label names
    label_names = load_label_names(mid_to_display_path)

    print(f"Total events in training data: {len(df)}")
    print(f"Unique segments: {df['segment_id'].nunique()}")
    print(f"Unique labels: {df['label'].nunique()}")

    # Calculate duration for each event
    df['duration'] = df['end_time'] - df['start_time']

    # Calculate total duration per label
    label_durations = df.groupby('label')['duration'].sum()
    label_counts = df['label'].value_counts()

    # Calculate ratios based on duration
    total_duration = df['duration'].sum()
    label_duration_ratios = label_durations / total_duration

    # Create comprehensive statistics
    label_stats = pd.DataFrame({
        'label': label_durations.index,
        'total_duration_seconds': label_durations.values,
        'total_duration_hours': label_durations.values / 3600,
        'event_count': label_counts.reindex(label_durations.index).values,
        'duration_ratio': label_duration_ratios.values,
        'duration_percentage': label_duration_ratios.values * 100
    })

    # Add average duration per event
    label_stats['avg_duration_per_event'] = (
        label_stats['total_duration_seconds'] / label_stats['event_count']
    )

    # Add display names
    label_stats['display_name'] = label_stats['label'].map(
        lambda x: label_names.get(x, x)
    )

    # Sort by total duration (descending)
    label_stats = label_stats.sort_values(
        'total_duration_seconds', ascending=False
    ).reset_index(drop=True)

    print("\nTop 20 labels by total duration:")
    print(label_stats.head(20)[
        ['display_name', 'total_duration_hours', 'duration_percentage', 'event_count']
    ].to_string(index=False))

    print("\nBottom 20 labels by total duration:")
    print(label_stats.tail(20)[
        ['display_name', 'total_duration_hours', 'duration_percentage', 'event_count']
    ].to_string(index=False))

    # Create visualizations
    create_distribution_plots(label_stats)

    # Create output directory and save detailed statistics
    os.makedirs('out', exist_ok=True)
    output_file = 'out/strong_label_distribution_stats.csv'
    label_stats.to_csv(output_file, index=False)
    print(f"\nDetailed statistics saved to: {output_file}")
    
    return label_stats


def create_distribution_plots(label_stats: pd.DataFrame):
    """Create various distribution plots."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Top 30 labels bar plot (by duration)
    plt.subplot(2, 3, 1)
    top_30 = label_stats.head(30)
    bars = plt.bar(range(len(top_30)), top_30['total_duration_hours'])
    plt.title('Top 30 Labels by Total Duration', fontsize=14, fontweight='bold')
    plt.xlabel('Label Rank')
    plt.ylabel('Total Duration (Hours)')
    plt.xticks(range(0, len(top_30), 5), range(1, len(top_30)+1, 5))

    # Add value labels on bars for top 10
    for i, bar in enumerate(bars[:10]):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.1f}h', ha='center', va='bottom', fontsize=8)

    # 2. Log scale distribution (by duration)
    plt.subplot(2, 3, 2)
    plt.bar(range(len(label_stats)), label_stats['total_duration_hours'])
    plt.yscale('log')
    plt.title('All Labels Duration Distribution (Log Scale)', fontsize=14, fontweight='bold')
    plt.xlabel('Label Rank')
    plt.ylabel('Total Duration (Hours, Log Scale)')
    
    # 3. Cumulative percentage (by duration)
    plt.subplot(2, 3, 3)
    cumulative_pct = label_stats['duration_percentage'].cumsum()
    plt.plot(range(len(cumulative_pct)), cumulative_pct, linewidth=2)
    plt.title('Cumulative Duration Percentage', fontsize=14, fontweight='bold')
    plt.xlabel('Label Rank')
    plt.ylabel('Cumulative Duration Percentage (%)')
    plt.grid(True, alpha=0.3)

    # Add horizontal lines for reference
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.7, label='50%')
    plt.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='80%')
    plt.axhline(y=95, color='g', linestyle='--', alpha=0.7, label='95%')
    plt.legend()

    # 4. Histogram of total durations
    plt.subplot(2, 3, 4)
    plt.hist(label_stats['total_duration_hours'], bins=50, edgecolor='black', alpha=0.7)
    plt.title('Histogram of Total Duration per Label', fontsize=14, fontweight='bold')
    plt.xlabel('Total Duration (Hours)')
    plt.ylabel('Number of Labels')
    plt.yscale('log')
    
    # 5. Box plot of duration percentages
    plt.subplot(2, 3, 5)
    plt.boxplot(label_stats['duration_percentage'], vert=True)
    plt.title('Box Plot of Duration Percentages', fontsize=14, fontweight='bold')
    plt.ylabel('Duration Percentage (%)')

    # Add statistics text
    stats_text = f"""Duration Statistics:
    Mean: {label_stats['duration_percentage'].mean():.3f}%
    Median: {label_stats['duration_percentage'].median():.3f}%
    Std: {label_stats['duration_percentage'].std():.3f}%
    Max: {label_stats['duration_percentage'].max():.3f}%
    Min: {label_stats['duration_percentage'].min():.6f}%"""

    plt.text(1.2, label_stats['duration_percentage'].median(), stats_text,
             verticalalignment='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # 6. Top 20 labels with names (by duration)
    plt.subplot(2, 3, 6)
    top_20 = label_stats.head(20)
    y_pos = np.arange(len(top_20))

    bars = plt.barh(y_pos, top_20['total_duration_hours'])
    plt.title('Top 20 Labels by Duration', fontsize=14, fontweight='bold')
    plt.xlabel('Total Duration (Hours)')
    plt.yticks(y_pos, [name[:25] + '...' if len(name) > 25 else name
                       for name in top_20['display_name']], fontsize=8)
    plt.gca().invert_yaxis()

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                f'{width:.1f}h', ha='left', va='center', fontsize=8)

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs('out', exist_ok=True)

    # Save the plot
    output_plot = 'out/strong_label_distribution_analysis.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Distribution plots saved to: {output_plot}")
    plt.close()
    
    # Create a separate detailed plot for top labels
    create_detailed_top_labels_plot(label_stats)


def create_detailed_top_labels_plot(label_stats: pd.DataFrame):
    """Create a detailed plot focusing on top labels."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Top 50 labels
    top_50 = label_stats.head(50)
    
    # Left plot: Bar chart with label names (by duration)
    y_pos = np.arange(len(top_50))
    bars = ax1.barh(y_pos, top_50['total_duration_hours'],
                    color=plt.cm.viridis(np.linspace(0, 1, len(top_50))))
    ax1.set_title('Top 50 Labels by Duration', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Total Duration (Hours)', fontsize=12)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{name[:30]}..." if len(name) > 30 else name
                         for name in top_50['display_name']], fontsize=8)
    ax1.invert_yaxis()

    # Add percentage labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                f'{top_50.iloc[i]["duration_percentage"]:.2f}%',
                ha='left', va='center', fontsize=7)
    
    # Right plot: Pie chart for top 10 (by duration)
    top_10 = label_stats.head(10)
    others_duration = label_stats.iloc[10:]['total_duration_hours'].sum()
    others_pct = label_stats.iloc[10:]['duration_percentage'].sum()

    pie_data = list(top_10['total_duration_hours']) + [others_duration]
    pie_labels = list(top_10['display_name']) + [f'Others ({len(label_stats)-10} labels)']
    pie_percentages = list(top_10['duration_percentage']) + [others_pct]

    wedges, texts, autotexts = ax2.pie(pie_data, labels=None, autopct='%1.1f%%',
                                       startangle=90,
                                       colors=plt.cm.Set3(np.linspace(0, 1, len(pie_data))))
    ax2.set_title('Top 10 Labels by Duration', fontsize=16, fontweight='bold')

    # Create legend
    legend_labels = [f'{label} ({pct:.2f}%)' for label, pct in zip(pie_labels, pie_percentages)]
    ax2.legend(wedges, legend_labels, title="Labels", loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()

    # Create output directory if it doesn't exist
    os.makedirs('out', exist_ok=True)

    # Save the detailed plot
    output_plot = 'out/top_strong_labels_detailed.png'
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Detailed top labels plot saved to: {output_plot}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Analyze AudioSet label distribution')
    parser.add_argument('--train-file', default='meta/audioset_train_strong.tsv',
                        help='Path to training TSV file')
    parser.add_argument('--mid-to-display', default='meta/mid_to_display_name.tsv',
                        help='Path to MID to display name mapping file')
    args = parser.parse_args()
    
    if not os.path.exists(args.train_file):
        print(f"Error: Training file not found: {args.train_file}")
        return
    
    if not os.path.exists(args.mid_to_display):
        print(f"Error: MID to display name file not found: {args.mid_to_display}")
        return
    
    print("AudioSet Label Distribution Analysis")
    print("=" * 50)
    
    label_stats = analyze_label_distribution(args.train_file, args.mid_to_display)
    
    print("\nAnalysis complete!")
    print(f"Total unique labels: {len(label_stats)}")
    print(f"Most duration label: {label_stats.iloc[0]['display_name']} "
          f"({label_stats.iloc[0]['total_duration_hours']:.1f} hours)")
    print(f"Least duration label: {label_stats.iloc[-1]['display_name']} "
          f"({label_stats.iloc[-1]['total_duration_hours']:.3f} hours)")
    duration_ratio = (label_stats.iloc[0]['total_duration_hours'] /
                      label_stats.iloc[-1]['total_duration_hours'])
    print(f"Duration imbalance ratio (max/min): {duration_ratio:.1f}:1")


if __name__ == "__main__":
    main()
