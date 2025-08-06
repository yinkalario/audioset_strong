#!/usr/bin/env python3
"""
Example script showing how to use the AudioSet training pipeline.

This script demonstrates:
1. Processing AudioSet metadata with audio path validation
2. Creating a dataset that handles missing audio files
3. Setting up the two-tier batch sampler
4. Running a simple training loop

Before running this script:
1. Update the audio_root path in configs/baby_cry.yaml
2. Ensure you have the required metadata files in meta/baby_cry/seg1s/

Author: Yin Cao
"""
import sys
from pathlib import Path

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processor import AudioSetDataProcessor  # noqa: E402
from src.data.dataset import AudioSetDataset  # noqa: E402
from src.data.sampler import TwoTierBatchSampler  # noqa: E402


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length label lists."""
    # Separate the batch elements
    wavs, labels, label_lists, clip_ids = zip(*batch)

    # Debug: Check shapes
    print(f"Debug: Batch size: {len(wavs)}")
    print(f"Debug: Audio shapes: {[w.shape for w in wavs[:3]]}")  # First 3 shapes
    print(f"Debug: Label list lengths: {[len(ll) for ll in label_lists[:3]]}")

    # Stack tensors and convert to tensors
    wav_batch = torch.stack(wavs)
    label_batch = torch.tensor(labels)

    # Keep label_lists and clip_ids as lists (variable length)
    return wav_batch, label_batch, list(label_lists), list(clip_ids)


def simple_model(input_size: int, num_classes: int = 1) -> nn.Module:
    """Create a simple CNN model for demonstration."""
    # Note: input_size is provided for interface compatibility but not used
    # in this simple adaptive model
    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=1024, stride=512),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(32, num_classes),
        nn.Sigmoid()
    )


def main():
    """Main training pipeline demonstration."""
    print("🎵 AudioSet Training Pipeline Example")
    print("=" * 50)

    # Configuration
    config_path = "configs/baby_cry.yaml"

    # Step 1: Process metadata
    print("\n1. Processing AudioSet metadata...")
    processor = AudioSetDataProcessor(config_path)
    # This only needs to be run once. So you can skip this step
    # if you have already run it.
    # processor.process_and_save()

    # Get processed paths
    processed_dir = Path(processor.config["processed_data_dir"])
    metadata_path = str(processed_dir / "metadata.parquet")
    mappings_path = str(processed_dir / "label_mappings.pkl")

    print(f"   ✓ Metadata saved to: {metadata_path}")
    print(f"   ✓ Label mappings saved to: {mappings_path}")

    # Step 2: Create dataset
    print("\n2. Creating AudioSet dataset...")
    dataset = AudioSetDataset(metadata_path, config_path)

    data_info = dataset.get_data_split_info()
    print(f"   ✓ Total samples: {data_info['total_samples']}")
    print(f"   ✓ Positive samples: {data_info['positive_samples']}")
    print(f"   ✓ Strong negative samples: {data_info['strong_negative_samples']}")
    print(f"   ✓ Weak negative samples: {data_info['weak_negative_samples']}")
    print(f"   ✓ Clip length: {data_info['clip_length']}s")

    # Step 3: Create sampler
    print("\n3. Setting up two-tier batch sampler...")

    # Load config to get batch size
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f)

    sampler = TwoTierBatchSampler(
        dataset=dataset,
        batch_size_per_device=config["batch_size"],
        metadata_path=metadata_path,
        mappings_path=mappings_path,
        config_path=config_path,
        num_replicas=1,
        rank=0
    )

    steps_per_epoch = sampler.steps_per_epoch
    print(f"   ✓ Steps per epoch: {steps_per_epoch}")
    print(f"   ✓ Batch size per device: {config['batch_size']}")

    # Step 4: Create DataLoader
    print("\n4. Creating DataLoader...")
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,  # Set to 0 for debugging, increase for real training
        pin_memory=True,
        collate_fn=custom_collate_fn  # Use custom collate function
    )

    # Step 5: Create model
    print("\n5. Creating model...")
    input_size = int(data_info['clip_length'] * data_info['sample_rate'])
    model = simple_model(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    print(f"   ✓ Model created with input size: {input_size}")

    # Step 6: Training loop demonstration
    print("\n6. Running training demonstration...")
    model.train()

    # Get hard negative mining threshold from config
    hard_neg_threshold = config.get("val_threshold", 0.5)
    print(f"   ✓ Hard negative threshold: {hard_neg_threshold}")

    for epoch in range(2):  # Just 2 epochs for demonstration
        print(f"\n   Epoch {epoch + 1}/2")
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0
        epoch_hard_negatives = []  # Collect hard negatives for end-of-epoch update

        for batch_idx, (wav, y, _, clip_ids) in enumerate(dataloader):
            if batch_idx >= 3:  # Only process first 3 batches for demo
                break

            # Forward pass
            wav = wav.unsqueeze(1)  # Add channel dimension [B, 1, T]
            logits = model(wav)
            loss = criterion(logits.squeeze(), y.float())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Collect hard negatives (but don't update buffer yet)
            with torch.no_grad():
                predictions = torch.sigmoid(logits.squeeze())
                # Find negative samples (y=0) with high predictions (false positives)
                for pred, label, clip_id in zip(predictions, y, clip_ids):
                    if label == 0 and pred > hard_neg_threshold:
                        epoch_hard_negatives.append(clip_id)

            # Count positive/negative samples
            pos_count = y.sum().item()
            neg_count = len(y) - pos_count

            print(f"     Batch {batch_idx}: loss={loss.item():.4f}, "
                  f"pos={pos_count}, neg={neg_count}")

        # Update hard negative buffer at end of epoch
        if epoch_hard_negatives:
            sampler.extend_hard_buffer(epoch_hard_negatives)
            print(f"   ✓ Added {len(epoch_hard_negatives)} hard negatives to buffer")

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"   Average loss: {avg_loss:.4f}")

    print("\n🎉 Training demonstration completed!")
    print("\nNext steps for real training:")
    print("- Increase num_workers in DataLoader for faster data loading")
    print("- Use a proper model architecture (e.g., ResNet, EfficientNet)")
    print("- Add validation loop and metrics")
    print(f"- Tune hard negative mining threshold (currently {hard_neg_threshold})")
    print("- Add model checkpointing and logging")


if __name__ == "__main__":
    main()
