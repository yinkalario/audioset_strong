#!/usr/bin/env python3
"""
AudioSet Data Processor

Transforms raw AudioSet metadata into training-ready format with advanced
data balancing and sampling weight computation.

Key Features:
- Smart head-class trimming (unbalanced data only)
- Multi-label processing with proper logic
- √-frequency weight computation for balanced sampling
- Dual output format (CSV + Parquet) for inspection and efficiency
- AudioSet directory structure and missing file handling
- Configurable audio parameters and clip lengths

Author: Yin Cao
"""

import argparse
import random
from math import sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml


class AudioSetDataProcessor:
    """Processes AudioSet metadata for training."""
    
    def __init__(self, config_path: str, filter_missing_audio: bool = True):
        """Initialize with configuration.

        Args:
            config_path: Path to configuration YAML file
            filter_missing_audio: Whether to filter out samples with missing audio files
        """
        self.config = yaml.safe_load(open(config_path))
        self.output_dir = Path(self.config["processed_data_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract parameters
        self.head_labels = set(self.config["head_labels"])
        self.head_keep_frac = self.config["head_keep_frac"]
        self.completion_rare_cutoff = self.config["completion_rare_cutoff"]
        self.filter_missing_audio = filter_missing_audio

        # Audio path configuration
        self.audio_root = Path(self.config["audio_root"])
        self._build_audio_path_cache()

        print(f"[DataProcessor] Head labels: {self.head_labels}")
        print(f"[DataProcessor] Head keep fraction: {self.head_keep_frac}")
        print(f"[DataProcessor] Audio root: {self.audio_root}")
        print(f"[DataProcessor] Found {len(self.audio_path_cache)} audio files")

    def _build_audio_path_cache(self) -> None:
        """Build a cache of YID -> full_audio_path for efficient lookup."""
        self.audio_path_cache = {}

        if not self.audio_root.exists():
            print(f"Warning: Audio root {self.audio_root} not found. Audio paths will be relative.")
            return

        # Search patterns for AudioSet directory structure
        search_patterns = [
            "unbalanced_train_segments/unbalanced_train_segments_part*/Y*.wav",
            "balanced_train_segments/Y*.wav",
            "eval_segments/Y*.wav"
        ]

        for pattern in search_patterns:
            for audio_file in self.audio_root.glob(pattern):
                # Extract YID from filename (remove 'Y' prefix and '.wav' suffix)
                yid = audio_file.stem[1:]  # Remove 'Y' prefix
                self.audio_path_cache[yid] = str(audio_file)

        print(f"[DataProcessor] Cached {len(self.audio_path_cache)} audio file paths")

    def _get_audio_path(self, yid: str) -> str:
        """Get the full audio path for a given YID (only called for existing files)."""
        return self.audio_path_cache[yid]

    def _add_audio_paths_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add audio paths and filter missing files.

        Args:
            df: DataFrame with clip_id column

        Returns:
            DataFrame with audio_path column and missing files filtered out
        """
        # Extract YID from clip_id (format: YID_start_end)
        df['yid'] = df['clip_id'].apply(lambda x: x.split('_')[0])

        # Filter missing audio files if enabled
        if self.filter_missing_audio:
            initial_count = len(df)
            df = df[df['yid'].isin(self.audio_path_cache.keys())].copy()
            removed_count = initial_count - len(df)
            if removed_count > 0:
                print(f"  Removed {removed_count} samples with missing audio files")

        # Add audio paths
        if self.filter_missing_audio or len(self.audio_path_cache) > 0:
            # Use actual paths for existing files
            df['audio_path'] = df['yid'].apply(self._get_audio_path)
        else:
            # Fallback for testing without audio files
            df['audio_path'] = df['yid'].apply(lambda yid: f"Y{yid}.wav")

        # Clean up temporary column
        df = df.drop('yid', axis=1)
        return df

    def load_strong_metadata(self, paths: List[str], is_positive: bool) -> pd.DataFrame:
        """Load strong labeling TSV files."""
        all_data = []

        for path in paths:
            path_obj = Path(path)
            if not path_obj.exists():
                print(f"Warning: {path} not found, skipping")
                continue
                
            print(f"Loading strong metadata: {path}")
            
            # Load TSV, skip header if present
            try:
                # Try reading first line to detect header
                first_line = pd.read_csv(path, sep='\t', nrows=1)
                if 'start_time_seconds' in first_line.columns or 'clip_id' in first_line.columns:
                    # Has header, skip it
                    df = pd.read_csv(path, sep='\t', skiprows=1, names=[
                        'clip_id', 'start_time', 'end_time', 'label'
                    ])
                else:
                    # No header
                    df = pd.read_csv(path, sep='\t', names=[
                        'clip_id', 'start_time', 'end_time', 'label'
                    ])
            except:
                # Fallback: assume no header
                df = pd.read_csv(path, sep='\t', names=[
                    'clip_id', 'start_time', 'end_time', 'label'
                ])
            
            # Add metadata
            df['is_positive'] = is_positive
            df['source_file'] = path_obj.name
            df['data_type'] = 'strong'

            # Filter missing audio files and add paths
            df = self._add_audio_paths_and_filter(df)

            # For positive samples, labels list is empty (target sound)
            # For negative samples, labels list contains the negative label
            if is_positive:
                df['labels'] = [[] for _ in range(len(df))]
            else:
                df['labels'] = df['label'].apply(lambda x: [x])

            all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(result)} strong samples (positive={is_positive})")
        return result
    
    def load_weak_metadata(self, paths: List[str]) -> pd.DataFrame:
        """Load weak labeling CSV files."""
        all_data = []

        for path in paths:
            path_obj = Path(path)
            if not path_obj.exists():
                print(f"Warning: {path} not found, skipping")
                continue
                
            print(f"Loading weak metadata: {path}")
            
            # Load CSV with header
            df = pd.read_csv(path)
            
            # Rename columns to standard names
            df = df.rename(columns={
                'YTID': 'clip_id',
                'start_seconds': 'start_time', 
                'end_seconds': 'end_time',
                'positive_labels': 'label'
            })
            
            # Add metadata
            df['is_positive'] = False  # All weak samples are negative
            df['source_file'] = path_obj.name
            df['data_type'] = 'weak'

            # Filter missing audio files and add paths
            df = self._add_audio_paths_and_filter(df)

            # Parse multi-labels (e.g., "/m/04rlf,/m/09x0r" -> ["/m/04rlf", "/m/09x0r"])
            df['labels'] = df['label'].apply(self._parse_labels)

            all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        result = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(result)} weak samples")
        return result
    
    def _parse_labels(self, label_str: str) -> List[str]:
        """Parse comma-separated labels."""
        if pd.isna(label_str) or label_str == '':
            return []
        return [label.strip() for label in str(label_str).split(',')]
    
    def apply_head_trimming(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply head-class trimming to weak negative samples.

        Only applies to unbalanced training data, not balanced or eval data.
        """
        if df.empty:
            return df

        # Only apply to weak negative samples from unbalanced training data
        weak_mask = df['data_type'] == 'weak'
        unbalanced_mask = df['source_file'].str.contains('unbalanced', na=False)
        target_mask = weak_mask & unbalanced_mask

        target_df = df[target_mask].copy()
        other_df = df[~target_mask].copy()

        if target_df.empty:
            print("No unbalanced weak samples found for head trimming")
            return df

        print(f"Applying head trimming to {len(target_df)} unbalanced weak samples")
        print(f"Keeping {len(other_df)} other samples (balanced/eval) without trimming")

        # Check if any label in the sample is a head label
        def has_head_label(labels: List[str]) -> bool:
            return any(label in self.head_labels for label in labels)

        target_df['is_head'] = target_df['labels'].apply(has_head_label)

        head_count = target_df['is_head'].sum()
        print(f"Found {head_count} samples with head labels in unbalanced data")

        # Apply random trimming to head samples
        head_seed = self.config.get("head_trim_seed", 42)
        random.seed(head_seed)  # For reproducibility
        keep_mask = target_df['is_head'].apply(
            lambda is_head: not is_head or random.random() < self.head_keep_frac
        )

        trimmed_target = target_df[keep_mask].copy()
        kept_head = trimmed_target['is_head'].sum()

        print(f"After trimming: kept {kept_head}/{head_count} head samples "
              f"({kept_head/head_count:.2%} kept)")

        # Remove temporary column
        trimmed_target = trimmed_target.drop('is_head', axis=1)

        # Combine back
        result = pd.concat([other_df, trimmed_target], ignore_index=True)
        print(f"Total samples after head trimming: {len(result)}")

        return result
    
    def build_label_mappings(self, df: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Build label-to-indices mappings for sampling."""
        strong_neg_map = {}
        weak_neg_map = {}

        # Strong negative mappings
        strong_neg_df = df[(df['data_type'] == 'strong') & (~df['is_positive'])]
        for idx, row in strong_neg_df.iterrows():
            for label in row['labels']:
                if label not in strong_neg_map:
                    strong_neg_map[label] = []
                strong_neg_map[label].append(idx)
        
        # Weak negative mappings  
        weak_neg_df = df[df['data_type'] == 'weak']
        for idx, row in weak_neg_df.iterrows():
            for label in row['labels']:
                if label not in weak_neg_map:
                    weak_neg_map[label] = []
                weak_neg_map[label].append(idx)
        
        # Handle rare labels
        strong_neg_map = self._handle_rare_labels(strong_neg_map)
        weak_neg_map = self._handle_rare_labels(weak_neg_map)
        
        print(f"Strong negative labels: {len(strong_neg_map)}")
        print(f"Weak negative labels: {len(weak_neg_map)}")
        
        return strong_neg_map, weak_neg_map
    
    def _handle_rare_labels(self, label_map: Dict) -> Dict:
        """Bucket ultra-rare labels together."""
        rare_indices = []
        labels_to_remove = []
        
        for label, indices in label_map.items():
            if len(indices) < self.completion_rare_cutoff:
                rare_indices.extend(indices)
                labels_to_remove.append(label)
        
        # Remove rare labels and add them to "other_rare" bucket
        for label in labels_to_remove:
            del label_map[label]
        
        if rare_indices:
            label_map["other_rare"] = rare_indices
            print(f"Bucketed {len(labels_to_remove)} rare labels into 'other_rare' "
                  f"({len(rare_indices)} samples)")
        
        return label_map
    
    def compute_sqrt_weights(self, label_map: Dict) -> Tuple[List[str], List[float]]:
        """Compute √-frequency weights for balanced sampling."""
        if not label_map:
            return [], []

        labels = list(label_map.keys())
        counts = [len(label_map[label]) for label in labels]
        
        # Compute √-frequency weights
        sqrt_weights = [sqrt(count) for count in counts]
        total_weight = sum(sqrt_weights)
        
        if total_weight == 0:
            probs = [1.0 / len(labels)] * len(labels)
        else:
            probs = [w / total_weight for w in sqrt_weights]
        
        return labels, probs
    
    def process_and_save(self) -> None:
        """Main processing pipeline."""
        print("=== AudioSet Data Processing ===")

        # Load all metadata
        pos_strong = self.load_strong_metadata(
            self.config["pos_strong_paths"], is_positive=True
        )
        neg_strong = self.load_strong_metadata(
            self.config["neg_strong_paths"], is_positive=False
        )
        neg_weak = self.load_weak_metadata(
            self.config["neg_weak_paths"]
        )
        
        # Combine all data
        all_data = []
        if not pos_strong.empty:
            all_data.append(pos_strong)
        if not neg_strong.empty:
            all_data.append(neg_strong)
        if not neg_weak.empty:
            all_data.append(neg_weak)
        
        if not all_data:
            raise ValueError("No data loaded!")
        
        df = pd.concat(all_data, ignore_index=True)
        print(f"Total samples before processing: {len(df)}")
        
        # Apply head trimming
        df = self.apply_head_trimming(df)
        
        # Build label mappings
        strong_neg_map, weak_neg_map = self.build_label_mappings(df)
        
        # Compute sampling weights
        strong_labels, strong_probs = self.compute_sqrt_weights(strong_neg_map)
        weak_labels, weak_probs = self.compute_sqrt_weights(weak_neg_map)

        # Ensure proper data types
        df['start_time'] = df['start_time'].astype(float)
        df['end_time'] = df['end_time'].astype(float)
        df['is_positive'] = df['is_positive'].astype(bool)

        # Save both formats: CSV for inspection, Parquet for efficient loading
        csv_path = self.output_dir / "metadata.csv"
        parquet_path = self.output_dir / "metadata.parquet"

        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)

        print(f"Saved metadata (CSV for inspection): {csv_path}")
        print(f"Saved metadata (Parquet for loading): {parquet_path}")

        # Save label mappings and weights as pickle
        mappings = {
            'strong_neg_map': strong_neg_map,
            'weak_neg_map': weak_neg_map,
            'strong_labels': strong_labels,
            'strong_probs': strong_probs,
            'weak_labels': weak_labels,
            'weak_probs': weak_probs
        }

        mappings_path = self.output_dir / "label_mappings.pkl"
        pd.Series(mappings).to_pickle(mappings_path)
        print(f"Saved label mappings: {mappings_path}")

        # Print summary
        print("\n=== Processing Summary ===")
        pos_count = df['is_positive'].sum()
        strong_neg_count = ((df['data_type'] == 'strong') & (~df['is_positive'])).sum()
        weak_neg_count = (df['data_type'] == 'weak').sum()
        print(f"Positive samples: {pos_count:,}")
        print(f"Strong negative samples: {strong_neg_count:,}")
        print(f"Weak negative samples: {weak_neg_count:,}")
        print(f"Strong negative labels: {len(strong_labels)}")
        print(f"Weak negative labels: {len(weak_labels)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Process AudioSet metadata")
    parser.add_argument("--config", default="configs/baby_cry.yaml",
                        help="Path to config file")
    args = parser.parse_args()
    processor = AudioSetDataProcessor(args.config)
    processor.process_and_save()


if __name__ == "__main__":
    main()
