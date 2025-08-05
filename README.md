# AudioSet Strong Labeling Dataset Processor

This project provides tools for extracting and analyzing specific sound types from the AudioSet strong labeling dataset. It focuses on temporally-strong labeled audio events with precise start and end times, allowing for detailed analysis of sound event distributions and temporal overlaps.

## Overview

AudioSet strong labeling data contains temporally precise annotations for audio events, marking the exact start and end times of sound occurrences within 10-second audio clips. This project enables:

- **Target Sound Extraction**: Extract specific sound types (e.g., baby cry, gunshots, snoring) from the full dataset
- **Overlap Analysis**: Categorize events into overlapped and non-overlapped based on temporal co-occurrence with other sounds
- **Distribution Analysis**: Analyze label distributions by duration rather than simple event counts
- **Visualization**: Generate comprehensive plots showing dataset imbalances and statistics

## Dataset Information

The AudioSet strong labeling dataset includes:
- **Training set**: 103,463 clips with 934,821 sound events across 447 unique labels
- **Evaluation set**: 16,996 clips with 139,538 sound events across 416 unique labels
- **Framed evaluation set**: 14,203 clips with both positive and negative labels on 960ms frames

## AudioSet Audio Directory Structure

The AudioSet audio files are organized in a specific directory structure:

```
audio_root/
├── unbalanced_train_segments/
│   ├── unbalanced_train_segments_part00/
│   │   ├── Y-0049eXE2Zc.wav
│   │   ├── Y-004VvnOKJE.wav
│   │   └── ...
│   ├── unbalanced_train_segments_part01/
│   │   └── ...
│   └── ... (up to part40)
├── balanced_train_segments/
│   ├── Y00M9FhCet6s.wav
│   ├── Y00QiuqBbiI.wav
│   └── ...
└── eval_segments/
    ├── Y007P6bFgRCU.wav
    ├── Y00A8LzmKCE.wav
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

## Usage

### Quick Start (Recommended)

For processing all sound types automatically, use the provided bash script:

```bash
bash scripts/process_audioset_metadata.sh
```

For analyzing label distributions (both strong and weak), use:

```bash
bash scripts/analyze_label_distributions.sh
```

This script will automatically:
1. Generate raw positive metadata for baby_cry, gun, and snore sounds
2. Generate raw negative strong metadata for all three sound types
3. Generate raw negative weak metadata for all three sound types
4. Generate 1-second segmented positive metadata
5. Generate 1-second segmented negative metadata

### Manual Processing (Advanced)

If you need to process individual sound types or customize the workflow:

#### 1. Extract Raw Positive Metadata

The script `generate_raw_target_meta.py` extracts specific sound types and categorizes them into overlapped and non-overlapped events.

**Configuration**: Edit the target labels at the top of the script:

```python
# For baby cry
target_labels = ['/t/dd00002']
target_name = 'baby_cry'

# For gun sounds (multiple related labels)
target_labels = ['/m/032s66', '/m/04zjc', '/m/073cg4']  # Gunshot, Machine gun, Cap gun
target_name = 'gun'

# For snore sounds
target_labels = ['/m/01d3sd', '/m/07q0yl5']  # Snoring, Snort
target_name = 'snore'
```

**Run extraction**:
```bash
python src/generate_raw_target_meta.py --input-dir meta --output-dir meta
```

#### 2. Extract Raw Negative Strong Metadata

Generate negative strong samples (clips without target sounds):

```bash
python src/generate_raw_neg_strong_meta.py --input-dir meta --output-dir meta
```

#### 3. Extract Raw Negative Weak Metadata

Generate negative weak samples (10-second segments without target sounds):

```bash
python src/generate_raw_neg_weak_meta.py --input-dir meta --output-dir meta
```

#### 4. Generate Segmented Positive Metadata

Create 1-second segments from positive samples:

```bash
python src/generate_seg_target_meta.py
```

#### 5. Generate Segmented Negative Metadata

Create 1-second segments from negative samples:

```bash
python src/generate_seg_neg_strong_meta.py
```

#### 6. Analyze Label Distribution (Optional)

**Strong Label Distribution** (by duration):
```bash
python src/analyze_strong_label_distribution.py --train-file meta/audioset_train_strong.tsv --mid-to-display meta/mid_to_display_name.tsv
```

**Weak Label Distribution** (by occurrence count):
```bash
python src/analyze_weak_label_distribution.py --train-file meta/unbalanced_train_segments.csv --mid-to-display meta/mid_to_display_name.tsv
```

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

## Makefile Targets

The project includes convenient Makefile targets for cleaning generated files:

```bash
# Show all available targets
make help

# Clean cache files, temp files, etc. (preserves metadata and analysis)
make clean

# Remove generated metadata files (preserves original AudioSet files)
make clean-metadata

# Remove all analysis output files
make clean-analysis

# Clean everything (cache, metadata, and analysis)
make clean-all

# Sync with main branch
make sync
```

**Target Details:**
- `clean`: Removes Python cache, .DS_Store, pytest cache, etc.
- `clean-metadata`: Removes generated .tsv/.csv files in subdirectories (preserves original AudioSet files)
- `clean-analysis`: Removes the entire `out/` directory
- `clean-all`: Runs all cleaning targets

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{audioset_strong_processor,
  title={AudioSet Strong Labeling Dataset Processor},
  author={Yin Cao},
  year={2025},
  url={https://github.com/yinkalario/audioset_strong}
}
```

## References

- [AudioSet: An ontology and human-labeled dataset for audio events](https://research.google.com/audioset/)
- [AudioSet Strong Labels (ICASSP 2021)](https://research.google.com/audioset/download_strong.html)
- [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
