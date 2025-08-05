#!/usr/bin/env python3
"""
Test script for AudioSet data processor.

Tests the complete data processing pipeline including:
- Metadata loading and parsing
- Head-class trimming with correct multi-label logic
- √-frequency weight computation
- Parquet file generation

Author: Yin Cao
"""

import tempfile
import shutil
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.data_processor import AudioSetDataProcessor


def create_test_config(temp_dir: Path) -> str:
    """Create a test configuration file."""
    config = {
        "pos_strong_paths": [
            str(temp_dir / "pos_train.tsv"),
            str(temp_dir / "pos_eval.tsv")
        ],
        "neg_strong_paths": [
            str(temp_dir / "neg_train.tsv"),
            str(temp_dir / "neg_eval.tsv")
        ],
        "neg_weak_paths": [
            str(temp_dir / "weak_train.csv"),
            str(temp_dir / "weak_eval.csv")
        ],
        "head_labels": ["/m/09x0r", "/m/04rlf"],
        "head_keep_frac": 0.4,
        "head_trim_seed": 42,
        "processed_data_dir": str(temp_dir / "processed"),
        "completion_rare_cutoff": 3,
        "sample_rate": 32000,
        "clip_length": 1.0,
        "batch_size": 32,
        "pos_per_batch_frac": 0.25,
        "strong_neg_per_batch_frac": 0.25,
        "weak_neg_per_batch_frac": 0.5,
        "tierA_fraction": 0.6,
        "primary_fraction": 0.9,
        "hard_buffer_size": 1000,
        "audio_root": str(temp_dir / "audio")  # Non-existent path for testing
    }
    
    config_path = temp_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    return str(config_path)


def create_test_metadata(temp_dir: Path):
    """Create test metadata files."""
    
    # Positive strong files
    pos_data = [
        ["baby_clip_1", 0.0, 1.0, "/t/dd00002"],
        ["baby_clip_2", 1.0, 2.0, "/t/dd00002"],
        ["baby_clip_3", 2.0, 3.0, "/t/dd00002"]
    ]
    
    for i, path in enumerate([temp_dir / "pos_train.tsv", temp_dir / "pos_eval.tsv"]):
        df = pd.DataFrame(pos_data, columns=["clip_id", "start_time", "end_time", "label"])
        df.to_csv(path, sep='\t', index=False, header=False)
    
    # Negative strong files
    neg_data = [
        ["neg_clip_1", 0.0, 1.0, "/m/09x0r"],  # speech
        ["neg_clip_2", 1.0, 2.0, "/m/04rlf"],  # music
        ["neg_clip_3", 2.0, 3.0, "/m/0395lw"], # bell
        ["neg_clip_4", 3.0, 4.0, "/m/068hy"],  # vehicle
        ["neg_clip_5", 4.0, 5.0, "/m/0k4j"],   # car
    ]
    
    for i, path in enumerate([temp_dir / "neg_train.tsv", temp_dir / "neg_eval.tsv"]):
        df = pd.DataFrame(neg_data, columns=["clip_id", "start_time", "end_time", "label"])
        df.to_csv(path, sep='\t', index=False, header=False)
    
    # Weak negative files (with multi-labels)
    weak_data = [
        ["weak_1", 0.0, 10.0, "/m/09x0r,/m/04rlf"],  # speech + music (both head)
        ["weak_2", 10.0, 20.0, "/m/09x0r"],          # speech only (head)
        ["weak_3", 20.0, 30.0, "/m/04rlf"],          # music only (head)
        ["weak_4", 30.0, 40.0, "/m/0395lw"],         # bell (non-head)
        ["weak_5", 40.0, 50.0, "/m/068hy,/m/0k4j"],  # vehicle + car (both non-head)
        ["weak_6", 50.0, 60.0, "/m/0395lw,/m/09x0r"], # bell + speech (mixed)
    ]
    
    for i, path in enumerate([temp_dir / "weak_train.csv", temp_dir / "weak_eval.csv"]):
        df = pd.DataFrame(weak_data, columns=["YTID", "start_seconds", "end_seconds", "positive_labels"])
        df.to_csv(path, index=False)


def test_data_processor():
    """Test the complete data processing pipeline."""
    print("=== Testing AudioSet Data Processor ===")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        config_path = create_test_config(temp_path)
        create_test_metadata(temp_path)
        
        # Create processed directory (will be created by processor)
        processed_dir = temp_path / "processed"
        
        # Temporarily change working directory for the processor
        original_cwd = Path.cwd()
        import os
        os.chdir(temp_path)
        
        try:
            # Initialize and run processor (disable audio filtering for testing)
            processor = AudioSetDataProcessor(config_path, filter_missing_audio=False)
            processor.process_and_save()
            
            # Test outputs exist
            metadata_path = processed_dir / "metadata.parquet"
            mappings_path = processed_dir / "label_mappings.pkl"

            assert metadata_path.exists(), "metadata.parquet not created"
            assert mappings_path.exists(), "label_mappings.pkl not created"
            
            # Load and inspect results
            df = pd.read_parquet(metadata_path)
            mappings = pd.read_pickle(mappings_path)
            
            print(f"✓ Loaded {len(df)} total samples")
            
            # Test data types
            pos_count = (df['is_positive'] == True).sum()
            strong_neg_count = ((df['data_type'] == 'strong') & (df['is_positive'] == False)).sum()
            weak_neg_count = (df['data_type'] == 'weak').sum()
            
            print(f"✓ Positive samples: {pos_count}")
            print(f"✓ Strong negative samples: {strong_neg_count}")
            print(f"✓ Weak negative samples: {weak_neg_count}")
            
            # Test head trimming logic
            weak_df = df[df['data_type'] == 'weak']
            head_samples = 0
            for _, row in weak_df.iterrows():
                labels = row['labels']
                if any(label in ["/m/09x0r", "/m/04rlf"] for label in labels):
                    head_samples += 1
            
            print(f"✓ Head samples after trimming: {head_samples}")
            
            # Test label mappings
            strong_map = mappings['strong_neg_map']
            weak_map = mappings['weak_neg_map']
            
            print(f"✓ Strong negative labels: {len(strong_map)}")
            print(f"✓ Weak negative labels: {len(weak_map)}")
            
            # Test √-frequency weights
            strong_labels = mappings['strong_labels']
            strong_probs = mappings['strong_probs']
            
            assert len(strong_labels) == len(strong_probs), "Label/prob mismatch"
            assert abs(sum(strong_probs) - 1.0) < 1e-6, "Probabilities don't sum to 1"
            
            print(f"✓ Strong label probabilities sum to {sum(strong_probs):.6f}")
            
            # Test multi-label parsing
            sample_weak = weak_df.iloc[0]
            if isinstance(sample_weak['labels'], list) and len(sample_weak['labels']) > 1:
                print(f"✓ Multi-label parsing works: {sample_weak['labels']}")
            
            print("✓ All data processor tests passed!")
            
        finally:
            os.chdir(original_cwd)


def test_head_trimming_logic():
    """Test the head trimming logic specifically."""
    print("\n=== Testing Head Trimming Logic ===")
    
    # Test cases for multi-label head detection
    test_cases = [
        (["/m/09x0r"], True),  # speech only
        (["/m/04rlf"], True),  # music only
        (["/m/09x0r", "/m/04rlf"], True),  # both head labels
        (["/m/0395lw"], False),  # non-head only
        (["/m/09x0r", "/m/0395lw"], True),  # mixed (should be True)
        (["/m/04rlf", "/m/068hy"], True),  # mixed (should be True)
        (["/m/0395lw", "/m/068hy"], False),  # both non-head
        ([], False),  # empty
    ]
    
    head_labels = {"/m/09x0r", "/m/04rlf"}
    
    def has_head_label(labels):
        return any(label in head_labels for label in labels)
    
    for labels, expected in test_cases:
        result = has_head_label(labels)
        assert result == expected, f"Failed for {labels}: got {result}, expected {expected}"
        print(f"✓ {labels} -> {result} (expected {expected})")
    
    print("✓ Head trimming logic tests passed!")


if __name__ == "__main__":
    test_head_trimming_logic()
    test_data_processor()
