# AudioSet Data Pipeline

A robust and efficient data processing pipeline for AudioSet training with advanced sampling strategies, missing file handling, and comprehensive analysis tools.

## 🎯 Overview

This project provides a complete pipeline for working with AudioSet strong labeling data, from raw metadata processing to training-ready datasets with intelligent sampling strategies. It focuses on temporally-precise audio events with exact start and end times, enabling detailed sound event analysis and robust model training.

### Key Features

- **Complete Pipeline**: From raw AudioSet metadata to training-ready datasets
- **Smart Missing File Handling**: Automatically filters unavailable YouTube videos
- **Advanced Sampling**: Two-tier batch sampling with hard negative mining
- **Target Sound Extraction**: Extract specific sound types (baby cry, gunshots, snoring)
- **Overlap Analysis**: Categorize events based on temporal co-occurrence
- **Multi-GPU Support**: Deterministic data partitioning for distributed training
- **Comprehensive Analysis**: Duration-based distribution analysis and visualization

## 🚀 Quickstart

### Prerequisites

1. **Download AudioSet data** (strong labels + audio files)
2. **Set up environment**: `bash scripts/create_env.sh && conda activate audioset_strong`
3. **Update audio path** in `configs/baby_cry.yaml`

### Step 1: Generate Training Metadata

```bash
# Process all sound types and generate training metadata
bash scripts/process_audioset_metadata.sh
```

This creates organized metadata files:
```
meta/
├── baby_cry/
│   ├── raw/pos/          # Raw positive labels
│   ├── raw/neg_strong/   # Raw negative strong labels
│   ├── raw/neg_weak/     # Raw negative weak labels
│   └── seg1s/            # 1-second segmented clips
└── processed/            # Training-ready data
    ├── metadata.parquet  # Efficient format for training
    ├── metadata.csv      # Human-readable format
    └── label_mappings.pkl # Label mappings
```

### Step 2: Train a Model

```python
import yaml
from src.data.data_processor import AudioSetDataProcessor
from src.data.dataset import AudioSetDataset
from src.data.sampler import TwoTierBatchSampler
from torch.utils.data import DataLoader

# 1. Process metadata (handles missing files automatically)
# This only needs to be run once. So you can skip this step
# if you have already run it.
processor = AudioSetDataProcessor("configs/baby_cry.yaml")
processor.process_and_save()

# 2. Create dataset
dataset = AudioSetDataset(
    "meta/baby_cry/processed/metadata.parquet",
    "configs/baby_cry.yaml"
)

# 3. Create advanced sampler with two-tier negative sampling
with open("configs/baby_cry.yaml") as f:
    config = yaml.safe_load(f)

sampler = TwoTierBatchSampler(
    dataset=dataset,
    batch_size_per_device=config["batch_size"],
    metadata_path="meta/baby_cry/processed/metadata.parquet",
    mappings_path="meta/baby_cry/processed/label_mappings.pkl",
    config_path="configs/baby_cry.yaml",
    num_replicas=1,  # Number of GPUs
    rank=0           # Current GPU rank
)

# 4. Create DataLoader
dataloader = DataLoader(dataset, batch_sampler=sampler)

# 5. Training loop
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Important for deterministic sampling

    for batch_idx, (wav, labels, labels_mask, clip_ids) in enumerate(dataloader):
        # wav: [batch_size, audio_length] - Audio waveforms
        # labels: [batch_size] - Binary labels (1=positive, 0=negative)
        # labels_mask: [batch_size, num_labels] - Multi-label mask
        # clip_ids: [batch_size] - Clip identifiers

        # Your training code here

        # Hard negative mining (optional but recommended)
        # Collect hard negatives during epoch, update buffer at epoch end
        epoch_hard_negatives = []

        with torch.no_grad():
            predictions = torch.sigmoid(logits.squeeze())
            # Find negative samples with high predictions (false positives)
            threshold = config["val_threshold"]  # Use config parameter
            for pred, label, clip_id in zip(predictions, labels, clip_ids):
                if label == 0 and pred > threshold:
                    epoch_hard_negatives.append(clip_id)

    # Update sampler's hard negative buffer at end of epoch
    if epoch_hard_negatives:
        sampler.extend_hard_buffer(epoch_hard_negatives)
```

**💡 For a complete working example, see `examples/train_with_audioset.py`**

### Step 3: Run Complete Training Example

```bash
# See examples/train_with_audioset.py for a complete working example
python examples/train_with_audioset.py
```

This demonstrates:
- Complete pipeline from metadata processing to training
- Two-tier batch sampling with hard negative mining
- Proper DataLoader setup and training loop
- Hard negative buffer updates during training

### Step 4: Analyze Datasets (Optional)

```bash
# Generate comprehensive analysis of all AudioSet datasets
bash scripts/analyze_all_datasets.sh
```

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yinkalario/audioset_strong.git
   cd audioset_strong
   ```

2. **Set up environment** (requires conda):
   ```bash
   bash scripts/create_env.sh
   conda activate audioset_strong
   ```

The script will automatically:
- Create a conda environment named `audioset_strong`
- Install all required packages from `requirements.txt`
- Handle environment cleanup if it already exists

## 📊 Dataset Information

The AudioSet strong labeling dataset includes:
- **Training set**: 103,463 clips with 934,821 sound events across 447 unique labels
- **Evaluation set**: 16,996 clips with 139,538 sound events across 416 unique labels
- **Framed evaluation set**: 14,203 clips with both positive and negative labels on 960ms frames

## 🔧 Core Components

### 1. Data Processor (`src/data/data_processor.py`)
Transforms raw AudioSet metadata into training-ready format:
- Loads strong/weak label files with automatic format detection
- Applies head-class trimming (unbalanced data only) to reduce common class dominance
- Handles missing audio files automatically by filtering unavailable YouTube videos
- Generates label mappings and √-frequency sampling weights
- Outputs dual format: CSV for inspection, Parquet for efficient loading

### 2. Dataset (`src/data/dataset.py`)
PyTorch Dataset with intelligent audio loading:
- **Configurable clip duration**: Supports any clip length (0.5s, 1s, 2s, etc.)
- **Smart cropping**: Random cropping for weak labels, exact timing for strong labels
- **Robust audio loading**: Automatic resampling and normalization
- **Missing file handling**: Gracefully handles unavailable audio files
- **Multi-format support**: Loads from CSV or Parquet metadata

### 3. Sampler (`src/data/sampler.py`)
Advanced batch sampler with two-tier negative sampling:
- **Fixed batch composition**: Configurable positive/negative ratios (default: 25%/25%/50%)
- **Two-tier negative sampling**: Balanced (Tier A/B) + hard negatives
- **√-frequency weighting**: Prevents over-sampling of very common labels
- **Label coverage guarantees**: Ensures all labels seen per epoch
- **Hard negative mining**: Adaptive focus on challenging examples
- **Multi-GPU support**: Deterministic data partitioning for distributed training

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

**Important Notes:**
- Audio files are prefixed with 'Y' (e.g., `Y-0049eXE2Zc.wav` for YID `-0049eXE2Zc`)
- Unbalanced training set is split into 41 parts (part00 to part40)
- Some audio files may be missing due to YouTube video removal/restrictions
- The pipeline automatically handles missing files by filtering them from metadata

## Project Structure

```
audioset_strong/
├── meta/                           # Metadata files (ignored by git)
│   ├── audioset_train_strong.tsv   # Training strong labels
│   ├── audioset_eval_strong.tsv    # Evaluation strong labels
│   ├── audioset_eval_strong_framed_posneg.tsv  # Framed pos/neg labels
│   ├── mid_to_display_name.tsv     # Label ID to name mapping
│   ├── baby_cry/                   # Baby cry metadata
│   │   ├── raw/
│   │   │   ├── pos/                # Raw positive labels
│   │   │   ├── neg_strong/         # Raw negative strong labels
│   │   │   └── neg_weak/           # Raw negative weak labels (10-second segments)
│   │   └── seg1s/
│   │       ├── pos/                # 1-second segmented positive labels
│   │       └── neg_strong/         # 1-second segmented negative labels
│   ├── gun/                        # Gun sound metadata (same structure)
│   └── snore/                      # Snore sound metadata (same structure)
├── src/                            # Source code
│   ├── generate_meta/              # Metadata generation scripts
│   │   ├── generate_raw_target_meta.py    # Extract raw positive metadata
│   │   ├── generate_raw_neg_strong_meta.py # Extract raw negative strong metadata
│   │   ├── generate_raw_neg_weak_meta.py  # Extract raw negative weak metadata
│   │   ├── generate_seg_target_meta.py    # Generate segmented positive metadata
│   │   └── generate_seg_neg_strong_meta.py # Generate segmented negative metadata
│   └── analyze_meta/               # Analysis scripts
│       ├── analyze_single_strong_dataset.py # Single strong dataset analysis
│       ├── analyze_strong_label_distribution.py # Strong label distribution analysis
│       └── analyze_weak_label_distribution.py   # Weak label distribution analysis
├── out/                            # Analysis outputs (ignored by git)
│   ├── strong_label_distribution_analysis.png # Strong label analysis
│   ├── weak_label_distribution_analysis.png   # Weak label analysis (combined)
│   ├── weak_label_distribution_analysis_{dataset}.png # Individual dataset analysis
│   ├── top_strong_labels_detailed.png         # Strong label details
│   ├── top_weak_labels_detailed.png           # Weak label details
│   ├── strong_label_distribution_stats.csv    # Strong label statistics
│   ├── weak_label_distribution_stats.csv      # Weak label statistics (combined)
│   └── weak_label_distribution_stats_{dataset}.csv # Individual dataset statistics
├── scripts/                        # Utility scripts
│   ├── create_env.sh               # Simple conda environment setup
│   ├── process_audioset_metadata.sh  # Complete processing pipeline
│   └── analyze_all_datasets.sh    # Complete dataset analysis (all 6 datasets)
└── requirements.txt               # Python dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yinkalario/audioset_strong.git
   cd audioset_strong
   ```

2. **Set up environment** (requires conda):
   ```bash
   bash scripts/create_env.sh
   conda activate audioset_strong
   ```

The script will:
- Create a conda environment named `audioset_strong` with Python 3.13
- Install all required packages from `requirements.txt`
- Handle environment cleanup if it already exists

## Configuration

### Audio Root Path

Before using the training pipeline, configure the path to your AudioSet audio files in `configs/baby_cry.yaml`:

```yaml
# === Audio data path ===
audio_root: "/path/to/your/audioset/audio"  # Update this path
```

The `audio_root` should point to the directory containing the AudioSet audio structure:
- `unbalanced_train_segments/` (with subdirectories part00-part40)
- `balanced_train_segments/`
- `eval_segments/`

**Note**: If audio files are not available locally (e.g., stored on a remote server), the pipeline will still work but will show warnings about missing audio files. Missing files are automatically filtered from the training data.

## ⚙️ Configuration

Key configuration parameters in `configs/baby_cry.yaml`:

```yaml
# Audio data path
audio_root: "/path/to/your/audioset/audio"

# Metadata paths
pos_strong_paths: ["meta/baby_cry/seg1s/baby_cry_ov_train.tsv"]
neg_strong_paths: ["meta/baby_cry/seg1s/baby_cry_neg_train.tsv"]
weak_paths: ["meta/baby_cry/seg1s/baby_cry_weak_train.csv"]
processed_data_dir: "meta/baby_cry/processed"

# Audio parameters
sample_rate: 16000
clip_length: 1.0  # seconds

# Head trimming (unbalanced data only)
head_labels: ["/m/09x0r", "/m/04rlf"]  # Speech, Music
head_keep_frac: 0.2
completion_rare_cutoff: 10

# Batch composition
batch_size: 16
pos_per_batch_frac: 0.25      # 25% positive samples
strong_neg_per_batch_frac: 0.25  # 25% strong negatives
weak_neg_per_batch_frac: 0.5   # 50% weak negatives

# Two-tier sampling
tierA_fraction: 0.6    # 60% Tier A, 40% Tier B
primary_fraction: 0.9  # 90% primary sampling, 10% completion
```

## 🎯 Advanced Features

### Two-Tier Batch Sampling
The sampler implements sophisticated negative sampling strategies:

- **Tier A**: High-frequency labels with √-frequency weighting
- **Tier B**: Low-frequency labels with uniform sampling
- **Completion Sampling**: Ensures all labels are seen per epoch
- **Hard Negative Mining**: Dynamically focuses on challenging examples

### Hard Negative Mining
The sampler supports adaptive hard negative mining to improve model performance:

```python
# Collect hard negatives during training epoch
epoch_hard_negatives = []

for batch in dataloader:
    # ... training code ...

    # Identify false positives (collect, don't update yet)
    with torch.no_grad():
        predictions = torch.sigmoid(logits)
        threshold = config["val_threshold"]  # From config file
        for pred, label, clip_id in zip(predictions, labels, clip_ids):
            if label == 0 and pred > threshold:
                epoch_hard_negatives.append(clip_id)

# Update buffer at end of epoch for efficiency
if epoch_hard_negatives:
    sampler.extend_hard_buffer(epoch_hard_negatives)
```

**Key Features:**
- **End-of-Epoch Updates**: Efficient batch updates instead of per-step
- **Config-Driven**: Threshold controlled by `val_threshold` in config
- **Dynamic Buffer**: Maintains a buffer of challenging negative samples
- **Automatic Integration**: Seamlessly integrates with two-tier sampling
- **Performance Boost**: Focuses training on difficult examples

### Smart Missing File Handling
The pipeline automatically handles missing YouTube videos:

1. **Discovery**: Scans AudioSet directories for available files
2. **Filtering**: Removes samples with missing audio files
3. **Reporting**: Shows how many files were filtered
4. **Continuation**: Training proceeds with available data

### Head-Class Trimming
Applied only to unbalanced training data to reduce over-representation:

- **Multi-label logic**: Keeps samples with ANY head label
- **Configurable**: Adjustable keep fraction and rare label cutoff
- **Preserves balance**: Maintains rare label representation

### Multi-GPU Support
Built-in support for distributed training:

- **Deterministic partitioning**: No sample overlap between ranks
- **Consistent batch composition**: Same ratios across all GPUs
- **Efficient scaling**: Linear scaling to multiple devices

## 📊 Analysis Tools

The pipeline includes comprehensive analysis tools for understanding dataset characteristics:

### Strong Label Analysis
Duration-based analysis of temporally-precise labels:

```bash
# Analyze individual datasets
python src/analyze_meta/analyze_single_strong_dataset.py \
    --file meta/audioset_train_strong.tsv \
    --mid-to-display meta/mid_to_display_name.tsv \
    --output-prefix out/train_analysis

# Analyze all strong datasets
bash scripts/analyze_all_datasets.sh
```

### Weak Label Analysis
Occurrence-based analysis of segment-level labels:

```bash
# Individual dataset analysis
python src/analyze_meta/analyze_weak_label_distribution.py \
    --train-file meta/unbalanced_train_segments.csv \
    --mid-to-display meta/mid_to_display_name.tsv
```

### Output Visualizations
- **6-panel analysis**: Distribution, imbalance, coverage, temporal patterns
- **Top-N detailed plots**: Focus on most/least common labels
- **Statistics export**: CSV files with complete metrics
- **Cross-dataset comparison**: Unified analysis across train/eval splits

**Complete Dataset Analysis** (analyze all 6 datasets separately):
```bash
bash scripts/analyze_all_datasets.sh
```

This analyzes all datasets individually:
- **Strong labels**: audioset_train_strong.tsv, audioset_eval_strong.tsv, audioset_eval_strong_framed_posneg.tsv
- **Weak labels**: balanced_train_segments.csv, unbalanced_train_segments.csv, eval_segments.csv

**Output** (6 separate analyses):

**Strong Label Analyses** (by duration):
- `out/strong_label_distribution_{dataset}_analysis.png` - 6-panel analysis for each dataset
- `out/strong_label_distribution_{dataset}_detailed.png` - Detailed visualization for each dataset
- `out/strong_label_distribution_{dataset}_stats.csv` - Statistics for each dataset

**Weak Label Analyses** (by occurrence count):
- `out/weak_label_distribution_analysis_{dataset}.png` - 4-panel analysis for each dataset
- `out/weak_label_distribution_stats_{dataset}.csv` - Statistics for each dataset

Where `{dataset}` is:
- **Strong**: train, eval, eval_framed (eval_framed filters for PRESENT events only)
- **Weak**: balanced_train, unbalanced_train, eval

## Key Features

### Temporal Overlap Detection

Events are categorized based on temporal overlap with other sound events in the same audio segment:

- **Non-overlapped**: Target sound occurs without any other sounds in the same time frame
- **Overlapped**: Target sound co-occurs with other sounds

### Duration-Based Analysis

Unlike simple event counting, the analysis calculates total duration for each label, providing more meaningful insights into dataset composition:

- **Most duration**: Mechanisms (67.3 hours, 12.58%)
- **Least duration**: Human locomotion (0.45 seconds)
- **Imbalance ratio**: 537,285:1

### Multi-Label Support

The extraction script supports multiple related labels for a single sound type:
```python
# Example: All gun-related sounds
target_labels = ['/m/032s66', '/m/04zjc', '/m/073cg4']
target_name = 'gun'
```

## Automated Processing Pipeline

The `scripts/process_audioset_metadata.sh` script automates the entire metadata processing workflow:

**Features:**
- **Automatic configuration**: Updates target labels for each sound type automatically
- **Error handling**: Stops on any error and provides clear error messages
- **Progress tracking**: Colored output showing progress through each step
- **Validation**: Checks for required files and directories before starting
- **Summary**: Displays final directory structure and statistics

**Usage:**
```bash
# Make sure you're in the audioset_strong root directory
bash scripts/process_audioset_metadata.sh
```

**What it does:**
1. Validates environment and required files
2. Generates raw positive metadata for baby_cry, gun, and snore
3. Generates raw negative strong metadata for all three sound types
4. Generates raw negative weak metadata for all three sound types
5. Creates 1-second segmented positive metadata
6. Creates 1-second segmented negative metadata
7. Generates basic label distribution analyses (train dataset only)
8. Displays summary of generated files

**Note**: For complete analysis of all 6 datasets separately, use `bash scripts/analyze_all_datasets.sh`

## Example Results

### Target Sound Statistics (by Duration)

| Sound Type | Labels | Rank | Duration | Percentage |
|------------|--------|------|----------|------------|
| Gunshot | /m/032s66 | 96 | 2.57 hours | 0.133% |
| Baby cry | /t/dd00002 | 115 | 2.14 hours | 0.109% |
| Machine gun | /m/04zjc | 147 | 1.67 hours | 0.087% |
| Snoring | /m/01d3sd | 165 | 1.44 hours | 0.075% |
| Cap gun | /m/073cg4 | 310 | 0.34 hours | 0.018% |
| Snort | /m/07q0yl5 | 375 | 0.11 hours | 0.006% |

### Complete Output Structure

```
meta/baby_cry/
├── raw/
│   ├── pos/                        # Raw positive labels
│   │   ├── baby_cry_nov_audioset_train_strong.tsv         # 64 events
│   │   ├── baby_cry_ov_audioset_train_strong.tsv          # 1,703 events
│   │   ├── baby_cry_nov_audioset_eval_strong.tsv          # 21 events
│   │   ├── baby_cry_ov_audioset_eval_strong.tsv           # 180 events
│   │   ├── baby_cry_nov_audioset_eval_strong_framed_posneg.tsv  # 64 events
│   │   └── baby_cry_ov_audioset_eval_strong_framed_posneg.tsv   # 259 events
│   ├── neg_strong/                 # Raw negative strong labels
│   │   ├── baby_cry_audioset_train_strong.tsv             # 928,463 events
│   │   ├── baby_cry_audioset_eval_strong.tsv              # 138,753 events
│   │   └── baby_cry_audioset_eval_strong_framed_posneg.tsv # 298,881 events
│   └── neg_weak/                   # Raw negative weak labels (10-second segments)
│       ├── baby_cry_neg_weak_unbalanced_train_segments.csv # 2,039,519 segments
│       ├── baby_cry_neg_weak_balanced_train_segments.csv  # 22,100 segments
│       └── baby_cry_neg_weak_eval_segments.csv            # 20,311 segments
└── seg1s/                          # 1-second segmented data
    ├── pos/                        # Segmented positive labels
    │   ├── baby_cry_nov_audioset_train_strong.tsv         # 108 segments
    │   ├── baby_cry_ov_audioset_train_strong.tsv          # 2,947 segments
    │   ├── baby_cry_nov_audioset_eval_strong.tsv          # 39 segments
    │   ├── baby_cry_ov_audioset_eval_strong.tsv           # 363 segments
    │   ├── baby_cry_nov_audioset_eval_strong_framed_posneg.tsv  # 64 segments
    │   └── baby_cry_ov_audioset_eval_strong_framed_posneg.tsv   # 259 segments
    └── neg_strong/                 # Segmented negative labels
        ├── baby_cry_audioset_train_strong.tsv             # 2,429,966 segments
        ├── baby_cry_audioset_eval_strong.tsv              # 398,338 segments
        └── baby_cry_audioset_eval_strong_framed_posneg.tsv # 298,881 segments
```

## File Formats

### Strong Label TSV Format
```
segment_id	start_time_seconds	end_time_seconds	label
YxlGt805lTA_30000	2.627	7.237	/m/053hz1
```

### Framed Positive/Negative TSV Format
```
segment_id	start_time_seconds	end_time_seconds	label	present
YxlGt805lTA_30000	0.960	1.920	/m/04rlf	PRESENT
YxlGt805lTA_30000	0.960	1.920	/m/07rgkc5	NOT_PRESENT
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{audioset_data_pipeline,
  title={AudioSet Data Pipeline: Advanced Training Pipeline for Audio Event Detection},
  author={Yin Cao},
  year={2025},
  url={https://github.com/yinkalario/audioset_strong}
}
```

## References

- [AudioSet: An ontology and human-labeled dataset for audio events](https://research.google.com/audioset/)
- [AudioSet Strong Labels (ICASSP 2021)](https://research.google.com/audioset/download_strong.html)
- [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
