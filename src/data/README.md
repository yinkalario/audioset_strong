# AudioSet Data Pipeline

A robust and efficient data processing pipeline for AudioSet training with advanced sampling strategies and missing file handling.

**Author: Yin Cao**

## 🎯 Overview

This pipeline transforms raw AudioSet metadata into training-ready format with intelligent data balancing, missing file handling, and advanced sampling strategies.

### Key Features

- **AudioSet Integration**: Handles real AudioSet directory structure and file naming
- **Smart Missing File Handling**: Automatically filters unavailable YouTube videos
- **Advanced Sampling**: Two-tier batch sampling with hard negative mining
- **Flexible Configuration**: Configurable clip lengths, batch composition, and audio parameters
- **Multi-GPU Support**: Deterministic data partitioning for distributed training
- **Dual Format Output**: CSV for inspection, Parquet for efficient loading

## 📁 AudioSet Directory Structure

The pipeline expects AudioSet audio files in this structure:

```
audio_root/
├── unbalanced_train_segments/
│   ├── unbalanced_train_segments_part00/
│   │   ├── Y-0049eXE2Zc.wav
│   │   └── ...
│   ├── unbalanced_train_segments_part01/
│   └── ... (up to part40)
├── balanced_train_segments/
│   ├── Y00M9FhCet6s.wav
│   └── ...
└── eval_segments/
    ├── Y007P6bFgRCU.wav
    └── ...
```

**Important**: Audio files are prefixed with 'Y' (e.g., `Y-0049eXE2Zc.wav` for YID `-0049eXE2Zc`)

## 🔧 Core Components

### 1. Data Processor (`data_processor.py`)
Transforms raw metadata into training format:
- Loads strong/weak label files
- Applies head-class trimming (unbalanced data only)
- Handles missing audio files automatically
- Generates label mappings and sampling weights
- Outputs dual format (CSV + Parquet)

### 2. Dataset (`dataset.py`)
PyTorch Dataset with intelligent audio loading:
- Configurable clip duration and sample rate
- Smart cropping: random for weak labels, exact for strong labels
- Robust audio loading with automatic resampling
- Returns normalized tensors with comprehensive metadata

### 3. Sampler (`sampler.py`)
Advanced batch sampler with two-tier negative sampling:
- Fixed batch composition ratios
- Hard negative mining with adaptive focus
- Label coverage guarantees
- Multi-GPU support with deterministic partitioning

## 🚀 Quick Start

### 1. Configuration

Update `configs/baby_cry.yaml`:

```yaml
# Audio data path
audio_root: "/path/to/your/audioset/audio"

# Metadata paths
pos_strong_paths: ["meta/baby_cry/seg1s/baby_cry_ov_train.tsv"]
neg_strong_paths: ["meta/baby_cry/seg1s/baby_cry_neg_train.tsv"]
weak_paths: ["meta/baby_cry/seg1s/baby_cry_weak_train.csv"]

# Output
processed_data_dir: "meta/baby_cry/processed"
```

### 2. Process Data

```python
from src.data.data_processor import AudioSetDataProcessor

# Process metadata (handles missing files automatically)
processor = AudioSetDataProcessor("configs/baby_cry.yaml")
processor.process_and_save()
```

### 3. Create Dataset and Sampler

```python
from src.data.dataset import AudioSetDataset
from src.data.sampler import TwoTierBatchSampler
from torch.utils.data import DataLoader

# Create dataset
dataset = AudioSetDataset(
    "meta/baby_cry/processed/metadata.parquet",
    "configs/baby_cry.yaml"
)

# Create sampler with advanced strategies
sampler = TwoTierBatchSampler(
    "meta/baby_cry/processed/metadata.parquet",
    "meta/baby_cry/processed/label_mappings.pkl",
    "configs/baby_cry.yaml"
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_sampler=sampler)
```

## 🎵 Missing File Handling

The pipeline automatically handles missing YouTube videos:

1. **Discovery**: Scans AudioSet directories for available files
2. **Filtering**: Removes samples with missing audio files
3. **Reporting**: Shows how many files were filtered
4. **Continuation**: Training proceeds with available data

**Note**: If audio files are stored remotely, the pipeline will show warnings but continue processing.

## 🧪 Testing

Run the comprehensive test suite:

```bash
PYTHONPATH=. python src/test/run_all_tests.py
```

Tests cover:
- Data processing with missing file handling
- Dataset loading and audio processing
- Batch sampling strategies
- Integration scenarios

## 📊 Batch Composition

Default batch composition (configurable):
- **25%** Positive samples (target sound)
- **25%** Strong negative samples (Tier A + Tier B)
- **50%** Weak negative samples

The sampler ensures balanced training while maintaining label coverage and supporting hard negative mining.

## 🔄 Hard Negative Mining

The pipeline supports dynamic hard negative mining:
- Maintains a buffer of hard negative samples
- Adaptively focuses on challenging examples
- Integrates seamlessly with batch composition
- Improves model performance on difficult cases

## 🌐 Multi-GPU Support

Built-in support for distributed training:
- Deterministic data partitioning across GPUs
- No sample overlap between ranks
- Consistent batch composition per GPU
- Efficient scaling to multiple devices

---

For detailed examples and advanced usage, see the main project README and example scripts.
