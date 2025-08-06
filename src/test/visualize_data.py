#!/usr/bin/env python3
"""
Data visualization script for AudioSet processed data.

This script demonstrates the benefits of using CSV format for easy data inspection
and visualization. Shows data distribution, head trimming effects, and sample statistics.
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def visualize_data_distribution():
    """Visualize the processed AudioSet data distribution."""
    print("=== AudioSet Data Visualization ===")

    # Check if processed data exists (prefer parquet, fallback to CSV)
    parquet_path = "meta/baby_cry/processed/metadata.parquet"
    csv_path = "meta/baby_cry/processed/metadata.csv"

    if Path(parquet_path).exists():
        metadata_path = parquet_path
        print("Loading processed data (parquet)...")
        df = pd.read_parquet(metadata_path)
    elif Path(csv_path).exists():
        metadata_path = csv_path
        print("Loading processed data (CSV)...")
        df = pd.read_csv(metadata_path)
    else:
        print("Error: No processed data found. Run data processor first:")
        print("PYTHONPATH=. python -m src.data.data_processor --config configs/baby_cry.yaml")
        return

    print(f"✓ Loaded {len(df):,} total samples")

    # Basic statistics
    print("\n=== Data Distribution ===")

    # Count by data type and positive/negative
    summary = df.groupby(['data_type', 'is_positive']).size().unstack(fill_value=0)
    print("\nSample counts by type:")
    print(summary)

    # Count by source file
    print(f"\nSample counts by source file:")
    source_counts = df['source_file'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count:,}")

    # Head trimming analysis for weak samples
    print(f"\n=== Head Trimming Analysis ===")
    weak_df = df[df['data_type'] == 'weak'].copy()

    if not weak_df.empty:
        # Parse labels for analysis (handle both string and list formats)
        def parse_labels(labels_data):
            """TODO: Add docstring for parse_labels."""
            if isinstance(labels_data, list):
                return labels_data  # Already parsed (from parquet)
            try:
                if pd.isna(labels_data) or labels_data == '' or labels_data == '[]':
                    return []
            except (ValueError, TypeError):
                # Handle cases where pd.isna fails on arrays
                if labels_data is None:
                    return []

            if isinstance(labels_data, str):
                if labels_data.startswith('[') and labels_data.endswith(']'):
                    labels_str = labels_data.strip('[]')
                    if not labels_str:
                        return []
                    labels = [label.strip().strip("'\"") for label in labels_str.split(',')]
                    return [label for label in labels if label]
                else:
                    return [label.strip() for label in str(labels_data).split(',') if label.strip()]
            else:
                return []

        weak_df['parsed_labels'] = weak_df['labels'].apply(parse_labels)

        # Check head labels
        head_labels = {"/m/09x0r", "/m/04rlf"}  # speech, music

        def has_head_label(labels):
            """TODO: Add docstring for has_head_label."""
            return any(label in head_labels for label in labels)

        weak_df['has_head'] = weak_df['parsed_labels'].apply(has_head_label)

        # Analyze by source file
        for source in weak_df['source_file'].unique():
            source_data = weak_df[weak_df['source_file'] == source]
            total = len(source_data)
            head_count = source_data['has_head'].sum()
            head_pct = head_count / total * 100 if total > 0 else 0

            print(f"\n{source}:")
            print(f"  Total samples: {total:,}")
            print(f"  Head label samples: {head_count:,} ({head_pct:.1f}%)")

            if 'unbalanced' in source:
                print(f"  → Head trimming applied (kept ~40%)")
            else:
                print(f"  → No head trimming (balanced/eval data)")

    # Create visualizations if matplotlib available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        print(f"\n=== Creating Visualizations ===")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AudioSet Data Distribution Analysis', fontsize=16)

        # 1. Data type distribution
        ax1 = axes[0, 0]
        type_counts = df['data_type'].value_counts()
        ax1.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        ax1.set_title('Distribution by Data Type')

        # 2. Positive vs Negative
        ax2 = axes[0, 1]
        pos_counts = df['is_positive'].value_counts()
        labels = ['Negative', 'Positive']
        ax2.pie(pos_counts.values, labels=labels, autopct='%1.1f%%')
        ax2.set_title('Positive vs Negative Samples')

        # 3. Source file distribution
        ax3 = axes[1, 0]
        source_counts = df['source_file'].value_counts()
        ax3.bar(range(len(source_counts)), source_counts.values)
        ax3.set_xticks(range(len(source_counts)))
        ax3.set_xticklabels(source_counts.index, rotation=45, ha='right')
        ax3.set_title('Samples by Source File')
        ax3.set_ylabel('Count')

        # 4. Duration distribution (for strong samples)
        ax4 = axes[1, 1]
        strong_df = df[df['data_type'] == 'strong'].copy()
        if not strong_df.empty:
            strong_df['duration'] = strong_df['end_time'] - strong_df['start_time']
            ax4.hist(strong_df['duration'], bins=50, alpha=0.7)
            ax4.set_title('Duration Distribution (Strong Samples)')
            ax4.set_xlabel('Duration (seconds)')
            ax4.set_ylabel('Count')
        else:
            ax4.text(0.5, 0.5, 'No strong samples', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Duration Distribution (Strong Samples)')

        plt.tight_layout()

        # Save plot
        output_path = "artifacts/data_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization: {output_path}")

        # Show plot
        plt.show()

    except ImportError:
        print("Matplotlib not available, skipping visualizations")
        print("Install with: pip install matplotlib seaborn")

    print(f"\n=== CSV Benefits Demonstrated ===")
    print("✓ Easy data inspection with pandas")
    print("✓ Quick statistics and grouping")
    print("✓ Simple visualization creation")
    print("✓ Human-readable format for debugging")
    print("\nTo switch to parquet later for production:")
    print("  df.to_parquet('metadata.parquet') # ~10x faster loading")


def inspect_sample_data():
    """Show sample data for inspection."""
    print("\n=== Sample Data Inspection ===")

    # Use same loading logic as main function
    parquet_path = "meta/baby_cry/processed/metadata.parquet"
    csv_path = "meta/baby_cry/processed/metadata.csv"

    if Path(parquet_path).exists():
        df = pd.read_parquet(parquet_path)
    elif Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        print("No processed data found.")
        return

    # Show first few samples of each type
    print("\nPositive samples (first 3):")
    pos_samples = df[df['is_positive'] == True].head(3)
    for _, row in pos_samples.iterrows():
        print(f"  {row['clip_id']}: {row['start_time']:.1f}-{row['end_time']:.1f}s, labels={row['labels']}")

    print("\nStrong negative samples (first 3):")
    strong_neg_samples = df[(df['data_type'] == 'strong') & (df['is_positive'] == False)].head(3)
    for _, row in strong_neg_samples.iterrows():
        print(f"  {row['clip_id']}: {row['start_time']:.1f}-{row['end_time']:.1f}s, labels={row['labels']}")

    print("\nWeak negative samples (first 3):")
    weak_samples = df[df['data_type'] == 'weak'].head(3)
    for _, row in weak_samples.iterrows():
        print(f"  {row['clip_id']}: {row['start_time']:.1f}-{row['end_time']:.1f}s, labels={row['labels']}")


if __name__ == "__main__":
    visualize_data_distribution()
    inspect_sample_data()
