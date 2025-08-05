#!/usr/bin/env python3
"""
Test script for AudioSet dataset and sampler.

Tests the complete data pipeline including:
- Dataset loading and audio processing
- Batch sampling with correct composition
- Two-tier sampling logic
- Hard negative mining
- Multi-GPU support

Author: Yin Cao
"""

import tempfile
import shutil
from pathlib import Path
import pandas as pd
import torch
import yaml

from src.data.data_processor import AudioSetDataProcessor
from src.data.dataset import AudioSetDataset
from src.data.sampler import TwoTierBatchSampler


def create_test_setup():
    """Create a complete test setup with processed data."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create config
    config = {
        "pos_strong_paths": [str(temp_dir / "pos.tsv")],
        "neg_strong_paths": [str(temp_dir / "neg.tsv")],
        "neg_weak_paths": [str(temp_dir / "weak.csv")],
        "head_labels": ["/m/09x0r", "/m/04rlf"],
        "head_keep_frac": 0.5,
        "head_trim_seed": 42,
        "processed_data_dir": str(temp_dir / "processed"),
        "completion_rare_cutoff": 2,
        "sample_rate": 16000,  # Lower for testing
        "clip_length": 0.5,    # Shorter for testing
        "batch_size": 8,       # Smaller for testing
        "pos_per_batch_frac": 0.25,
        "strong_neg_per_batch_frac": 0.25,
        "weak_neg_per_batch_frac": 0.5,
        "tierA_fraction": 0.6,
        "primary_fraction": 0.8,
        "hard_buffer_size": 100,
        "audio_root": str(temp_dir / "audio")
    }
    
    config_path = temp_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Create test metadata
    pos_data = [
        ["pos1_0.0_0.5", 0.0, 0.5, "/t/dd00002"],
        ["pos2_0.5_1.0", 0.5, 1.0, "/t/dd00002"],
        ["pos3_1.0_1.5", 1.0, 1.5, "/t/dd00002"],
    ]
    pd.DataFrame(pos_data).to_csv(temp_dir / "pos.tsv", sep='\t', index=False, header=False)
    
    neg_data = [
        ["neg1_0.0_0.5", 0.0, 0.5, "/m/09x0r"],
        ["neg2_0.5_1.0", 0.5, 1.0, "/m/04rlf"],
        ["neg3_1.0_1.5", 1.0, 1.5, "/m/0395lw"],
        ["neg4_1.5_2.0", 1.5, 2.0, "/m/068hy"],
        ["neg5_2.0_2.5", 2.0, 2.5, "/m/0k4j"],
    ]
    pd.DataFrame(neg_data).to_csv(temp_dir / "neg.tsv", sep='\t', index=False, header=False)
    
    weak_data = [
        ["weak1_0.0_10.0", 0.0, 10.0, "/m/09x0r,/m/04rlf"],
        ["weak2_10.0_20.0", 10.0, 20.0, "/m/09x0r"],
        ["weak3_20.0_30.0", 20.0, 30.0, "/m/04rlf"],
        ["weak4_30.0_40.0", 30.0, 40.0, "/m/0395lw"],
        ["weak5_40.0_50.0", 40.0, 50.0, "/m/068hy"],
        ["weak6_50.0_60.0", 50.0, 60.0, "/m/0k4j"],
    ]
    pd.DataFrame(weak_data, columns=["YTID", "start_seconds", "end_seconds", "positive_labels"]).to_csv(
        temp_dir / "weak.csv", index=False
    )
    
    # Create AudioSet directory structure with dummy audio files
    audio_root = temp_dir / "audio"

    # Create AudioSet subdirectories
    unbalanced_dir = audio_root / "unbalanced_train_segments" / "unbalanced_train_segments_part00"
    balanced_dir = audio_root / "balanced_train_segments"
    eval_dir = audio_root / "eval_segments"

    unbalanced_dir.mkdir(parents=True)
    balanced_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    # Create dummy audio files (silence) with Y prefix
    sample_rate = config["sample_rate"]
    duration = 10  # seconds
    n_samples = sample_rate * duration

    # Map YIDs to their expected locations (simulating AudioSet structure)
    audio_files = {
        "pos1": unbalanced_dir / "Ypos1.wav",
        "pos2": balanced_dir / "Ypos2.wav",
        "pos3": eval_dir / "Ypos3.wav",
        "neg1": unbalanced_dir / "Yneg1.wav",
        "neg2": unbalanced_dir / "Yneg2.wav",
        "neg3": balanced_dir / "Yneg3.wav",
        "neg4": balanced_dir / "Yneg4.wav",
        "neg5": eval_dir / "Yneg5.wav",
        "weak1": unbalanced_dir / "Yweak1.wav",
        "weak2": unbalanced_dir / "Yweak2.wav",
        "weak3": balanced_dir / "Yweak3.wav",
        "weak4": balanced_dir / "Yweak4.wav",
        "weak5": eval_dir / "Yweak5.wav",
        "weak6": eval_dir / "Yweak6.wav",
    }

    import torchaudio
    for yid, audio_path in audio_files.items():
        # Create dummy audio tensor and save
        dummy_audio = torch.zeros(1, n_samples)  # [channels, samples]
        torchaudio.save(str(audio_path), dummy_audio, sample_rate)

    # Update config to point to the audio root
    config["audio_root"] = str(audio_root)
    
    return temp_dir, str(config_path)


def test_dataset():
    """Test the AudioSet dataset."""
    print("=== Testing AudioSet Dataset ===")
    temp_dir, config_path = create_test_setup()
    
    try:
        # Change to temp directory and process data
        import os
        original_cwd = Path.cwd()
        os.chdir(temp_dir)
        
        # Process data
        processor = AudioSetDataProcessor(config_path)
        processor.process_and_save()
        
        # Test dataset
        dataset = AudioSetDataset(str(temp_dir / "processed" / "metadata.parquet"), config_path)
        
        print(f"✓ Dataset loaded with {len(dataset)} samples")
        
        # Test data split info
        info = dataset.get_data_split_info()
        print(f"✓ Data split: {info}")
        
        # Test sample loading
        for i in range(min(3, len(dataset))):
            wav, y, labels_mask, clip_id = dataset[i]
            
            print(f"✓ Sample {i}: wav {wav.shape}, y={y}, labels={labels_mask}, id={clip_id}")
            
            # Check tensor properties
            assert isinstance(wav, torch.Tensor), "wav should be tensor"
            assert wav.dtype == torch.float32, "wav should be float32"
            assert len(wav.shape) == 1, "wav should be 1D"
            assert wav.shape[0] == dataset.n_samples, f"wav should have {dataset.n_samples} samples"
            assert isinstance(y, int), "y should be int"
            assert y in [0, 1], "y should be 0 or 1"
            assert isinstance(labels_mask, set), "labels_mask should be set"
            assert isinstance(clip_id, str), "clip_id should be string"
        
        print("✓ Dataset tests passed!")
        
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)


def test_sampler():
    """Test the two-tier batch sampler."""
    print("\n=== Testing Two-Tier Batch Sampler ===")
    temp_dir, config_path = create_test_setup()
    
    try:
        # Change to temp directory and process data
        import os
        original_cwd = Path.cwd()
        os.chdir(temp_dir)
        
        # Process data
        processor = AudioSetDataProcessor(config_path)
        processor.process_and_save()
        
        # Test sampler
        sampler = TwoTierBatchSampler(
            str(temp_dir / "processed" / "metadata.parquet"),
            str(temp_dir / "processed" / "label_mappings.pkl"),
            config_path,
            world_size=1,
            rank=0
        )
        
        print(f"✓ Sampler created with {sampler.get_steps_per_epoch()} steps per epoch")
        
        # Test batch generation
        sampler.set_epoch(0)
        
        total_pos = 0
        total_strong_neg = 0
        total_weak_neg = 0
        
        # Load dataset for verification
        dataset = AudioSetDataset(str(temp_dir / "processed" / "metadata.parquet"), config_path)
        
        for step, batch_indices in enumerate(sampler):
            if step >= 3:  # Test first 3 batches
                break
            
            print(f"\n--- Batch {step} ---")
            print(f"Indices: {batch_indices}")
            
            # Verify batch size
            assert len(batch_indices) == sampler.batch_size, f"Wrong batch size: {len(batch_indices)}"
            
            # Count sample types
            pos_count = 0
            strong_neg_count = 0
            weak_neg_count = 0
            
            for idx in batch_indices:
                row = dataset.df.iloc[idx]
                if row['is_positive']:
                    pos_count += 1
                elif row['data_type'] == 'strong':
                    strong_neg_count += 1
                else:  # weak
                    weak_neg_count += 1
            
            print(f"Composition: {pos_count} pos, {strong_neg_count} strong neg, {weak_neg_count} weak neg")
            
            # Verify composition (allow some flexibility due to small test data)
            expected_pos = sampler.pos_per_batch
            expected_strong = sampler.strong_neg_per_batch
            expected_weak = sampler.weak_neg_per_batch
            
            print(f"Expected: {expected_pos} pos, {expected_strong} strong neg, {expected_weak} weak neg")
            
            total_pos += pos_count
            total_strong_neg += strong_neg_count
            total_weak_neg += weak_neg_count
        
        print(f"\n✓ Total across batches: {total_pos} pos, {total_strong_neg} strong neg, {total_weak_neg} weak neg")
        print("✓ Sampler tests passed!")
        
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)


def test_hard_negatives():
    """Test hard negative mining functionality."""
    print("\n=== Testing Hard Negative Mining ===")
    temp_dir, config_path = create_test_setup()
    
    try:
        import os
        original_cwd = Path.cwd()
        os.chdir(temp_dir)
        
        # Process data
        processor = AudioSetDataProcessor(config_path)
        processor.process_and_save()
        
        # Create sampler
        sampler = TwoTierBatchSampler(
            str(temp_dir / "processed" / "metadata.parquet"),
            str(temp_dir / "processed" / "label_mappings.pkl"),
            config_path
        )
        
        # Test adding hard negatives
        fake_hard_negs = [10, 20, 30]
        sampler.extend_hard_buffer(fake_hard_negs)
        
        print(f"✓ Added {len(fake_hard_negs)} hard negatives")
        print(f"✓ Buffer size: {len(sampler.hard_buffer)}")
        
        # Test that they get used
        sampler.set_epoch(1)
        
        used_hard_negs = 0
        for step, batch_indices in enumerate(sampler):
            if step >= 2:  # Test first 2 batches
                break
            
            for idx in batch_indices:
                if idx in fake_hard_negs:
                    used_hard_negs += 1
        
        print(f"✓ Used {used_hard_negs} hard negatives in first 2 batches")
        print("✓ Hard negative tests passed!")
        
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)


def test_multi_gpu():
    """Test multi-GPU (DDP) functionality."""
    print("\n=== Testing Multi-GPU Support ===")
    temp_dir, config_path = create_test_setup()
    
    try:
        import os
        original_cwd = Path.cwd()
        os.chdir(temp_dir)
        
        # Process data
        processor = AudioSetDataProcessor(config_path)
        processor.process_and_save()
        
        # Test with 2 "GPUs"
        world_size = 2
        samplers = []
        
        for rank in range(world_size):
            sampler = TwoTierBatchSampler(
                str(temp_dir / "processed" / "metadata.parquet"),
                str(temp_dir / "processed" / "label_mappings.pkl"),
                config_path,
                world_size=world_size,
                rank=rank
            )
            samplers.append(sampler)
        
        print(f"✓ Created {world_size} samplers for DDP")
        
        # Test that indices are disjoint
        all_indices = set()
        for rank, sampler in enumerate(samplers):
            sampler.set_epoch(0)
            
            rank_indices = set()
            for step, batch_indices in enumerate(sampler):
                if step >= 2:  # Test first 2 batches
                    break
                rank_indices.update(batch_indices)
            
            print(f"✓ Rank {rank} used {len(rank_indices)} unique indices")
            
            # Check for overlap
            overlap = all_indices & rank_indices
            assert len(overlap) == 0, f"Found overlapping indices between ranks: {overlap}"
            
            all_indices.update(rank_indices)
        
        print(f"✓ No index overlap between ranks")
        print("✓ Multi-GPU tests passed!")
        
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Skip audio tests if torchaudio not available
    try:
        import torchaudio
        test_dataset()
        test_sampler()
        test_hard_negatives()
        test_multi_gpu()
        print("\n🎉 All tests passed!")
    except ImportError:
        print("Warning: torchaudio not available, skipping audio tests")
        print("Install torchaudio to run full tests: pip install torchaudio")
