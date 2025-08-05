"""
AudioSet Data Pipeline

Advanced data pipeline for AudioSet-based audio event detection with:

Core Features:
- Flexible clip lengths (configurable duration)
- Two-tier negative sampling with √-frequency weighting
- Hard negative mining for adaptive training
- Smart head-class balancing (only on unbalanced data)
- Multi-GPU distributed training support
- Deterministic sampling across epochs

Components:
- AudioSetDataProcessor: Processes raw AudioSet metadata
- AudioSetDataset: Map-style PyTorch dataset for audio loading
- TwoTierBatchSampler: Advanced batch sampler with completion guarantees

Example Usage:
    # Process metadata
    from src.data.data_processor import AudioSetDataProcessor
    processor = AudioSetDataProcessor("configs/baby_cry.yaml")
    processor.process_and_save()

    # Create dataset and sampler
    from src.data import AudioSetDataset, TwoTierBatchSampler
    dataset = AudioSetDataset("artifacts/metadata.parquet", "configs/baby_cry.yaml")
    sampler = TwoTierBatchSampler(
        "artifacts/metadata.parquet",
        "artifacts/label_mappings.pkl",
        "configs/baby_cry.yaml"
    )
"""

# Conditional imports to handle missing dependencies gracefully
__all__ = []

try:
    from .dataset import AudioSetDataset
    __all__.append("AudioSetDataset")
except ImportError as e:
    print(f"Warning: Could not import AudioSetDataset: {e}")

try:
    from .sampler import TwoTierBatchSampler
    __all__.append("TwoTierBatchSampler")
except ImportError as e:
    print(f"Warning: Could not import TwoTierBatchSampler: {e}")

try:
    from .data_processor import AudioSetDataProcessor
    __all__.append("AudioSetDataProcessor")
except ImportError as e:
    print(f"Warning: Could not import AudioSetDataProcessor: {e}")
