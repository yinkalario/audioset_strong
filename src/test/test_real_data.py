#!/usr/bin/env python3
"""
Test script for real AudioSet data pipeline.

This script tests the complete pipeline with the actual processed AudioSet data.
"""

import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.sampler import TwoTierBatchSampler


def test_real_data_sampler():
    """Test the sampler with real processed data."""
    print("=== Testing Real Data Sampler ===")
    
    # Check if processed data exists (prefer parquet for efficiency)
    metadata_path = "meta/baby_cry/processed/metadata.parquet"
    mappings_path = "meta/baby_cry/processed/label_mappings.pkl"
    config_path = "configs/baby_cry.yaml"

    if not Path(metadata_path).exists():
        print(f"Error: {metadata_path} not found. Run data processor first:")
        print("PYTHONPATH=. python -m src.data.data_processor --config configs/baby_cry.yaml")
        return False

    if not Path(mappings_path).exists():
        print(f"Error: {mappings_path} not found. Run data processor first.")
        return False
    
    # Load and inspect data
    df = pd.read_parquet(metadata_path)
    config = yaml.safe_load(open(config_path))
    
    print(f"✓ Loaded metadata: {len(df)} total samples")
    
    pos_count = (df['is_positive']).sum()
    strong_neg_count = ((df['data_type'] == 'strong') & (~df['is_positive'])).sum()
    weak_neg_count = (df['data_type'] == 'weak').sum()
    
    print(f"✓ Data split: {pos_count} pos, {strong_neg_count} strong neg, {weak_neg_count} weak neg")
    
    # Test sampler creation
    try:
        sampler = TwoTierBatchSampler(
            metadata_path,
            mappings_path,
            config_path,
            world_size=1,
            rank=0
        )
        print(f"✓ Sampler created successfully")
        print(f"✓ Steps per epoch: {sampler.get_steps_per_epoch()}")
        
    except Exception as e:
        print(f"✗ Sampler creation failed: {e}")
        return False
    
    # Test batch generation
    try:
        sampler.set_epoch(0)
        
        batch_count = 0
        total_pos = 0
        total_strong_neg = 0
        total_weak_neg = 0
        
        for step, batch_indices in enumerate(sampler):
            if step >= 5:  # Test first 5 batches
                break
            
            batch_count += 1
            
            # Count sample types in batch
            pos_in_batch = 0
            strong_neg_in_batch = 0
            weak_neg_in_batch = 0
            
            for idx in batch_indices:
                row = df.iloc[idx]
                if row['is_positive']:
                    pos_in_batch += 1
                elif row['data_type'] == 'strong':
                    strong_neg_in_batch += 1
                else:  # weak
                    weak_neg_in_batch += 1
            
            total_pos += pos_in_batch
            total_strong_neg += strong_neg_in_batch
            total_weak_neg += weak_neg_in_batch
            
            print(f"  Batch {step}: {pos_in_batch} pos, {strong_neg_in_batch} strong neg, {weak_neg_in_batch} weak neg")
            
            # Verify batch size
            expected_batch_size = config["batch_size"]
            actual_batch_size = len(batch_indices)
            assert actual_batch_size == expected_batch_size, f"Wrong batch size: {actual_batch_size} vs {expected_batch_size}"
        
        print(f"✓ Generated {batch_count} batches successfully")
        print(f"✓ Total samples: {total_pos} pos, {total_strong_neg} strong neg, {total_weak_neg} weak neg")
        
        # Check composition ratios
        total_samples = total_pos + total_strong_neg + total_weak_neg
        pos_ratio = total_pos / total_samples
        strong_neg_ratio = total_strong_neg / total_samples
        weak_neg_ratio = total_weak_neg / total_samples
        
        expected_pos_ratio = config["pos_per_batch_frac"]
        expected_strong_ratio = config["strong_neg_per_batch_frac"]
        expected_weak_ratio = config["weak_neg_per_batch_frac"]
        
        print(f"✓ Composition ratios:")
        print(f"  Pos: {pos_ratio:.3f} (expected {expected_pos_ratio:.3f})")
        print(f"  Strong neg: {strong_neg_ratio:.3f} (expected {expected_strong_ratio:.3f})")
        print(f"  Weak neg: {weak_neg_ratio:.3f} (expected {expected_weak_ratio:.3f})")
        
    except Exception as e:
        print(f"✗ Batch generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test hard negative mining
    try:
        print("\n--- Testing Hard Negative Mining ---")
        
        # Add some fake hard negatives
        fake_hard_negs = [1000, 2000, 3000]
        sampler.extend_hard_buffer(fake_hard_negs)
        print(f"✓ Added {len(fake_hard_negs)} hard negatives to buffer")
        
        # Test next epoch
        sampler.set_epoch(1)
        
        # Check if hard negatives are used (simplified check)
        for step, batch_indices in enumerate(sampler):
            if step >= 2:  # Check first 2 batches
                break
            
            hard_negs_used = sum(1 for idx in batch_indices if idx in fake_hard_negs)
            if hard_negs_used > 0:
                print(f"✓ Hard negatives being used: {hard_negs_used} in batch {step}")
                break
        
    except Exception as e:
        print(f"✗ Hard negative mining test failed: {e}")
        return False
    
    print("\n✓ All real data sampler tests passed!")
    return True


def test_multi_gpu_simulation():
    """Test multi-GPU simulation with real data."""
    print("\n=== Testing Multi-GPU Simulation ===")

    metadata_path = "meta/baby_cry/processed/metadata.parquet"
    mappings_path = "meta/baby_cry/processed/label_mappings.pkl"
    config_path = "configs/baby_cry.yaml"
    
    try:
        # Test with 2 "GPUs"
        world_size = 2
        samplers = []
        
        for rank in range(world_size):
            sampler = TwoTierBatchSampler(
                metadata_path,
                mappings_path,
                config_path,
                world_size=world_size,
                rank=rank
            )
            samplers.append(sampler)
        
        print(f"✓ Created {world_size} samplers for multi-GPU simulation")
        
        # Test that indices are disjoint
        all_indices = set()
        for rank, sampler in enumerate(samplers):
            sampler.set_epoch(0)
            
            rank_indices = set()
            for step, batch_indices in enumerate(sampler):
                if step >= 3:  # Test first 3 batches
                    break
                rank_indices.update(batch_indices)
            
            print(f"✓ Rank {rank}: {len(rank_indices)} unique indices")
            
            # Check for overlap
            overlap = all_indices & rank_indices
            if overlap:
                print(f"✗ Found overlapping indices between ranks: {len(overlap)} overlaps")
                return False
            
            all_indices.update(rank_indices)
        
        print(f"✓ No index overlap between {world_size} ranks")
        print(f"✓ Total unique indices across all ranks: {len(all_indices)}")
        
    except Exception as e:
        print(f"✗ Multi-GPU simulation failed: {e}")
        return False
    
    print("✓ Multi-GPU simulation test passed!")
    return True


def main():
    """Run all real data tests."""
    print("AudioSet Real Data Pipeline Tests")
    print("=" * 50)
    
    success = True
    
    # Test sampler with real data
    if not test_real_data_sampler():
        success = False
    
    # Test multi-GPU simulation
    if not test_multi_gpu_simulation():
        success = False
    
    if success:
        print("\n🎉 All real data tests passed!")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
