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

## Project Structure

```
audioset_strong/
├── meta/                           # Metadata files
│   ├── audioset_train_strong.tsv   # Training strong labels
│   ├── audioset_eval_strong.tsv    # Evaluation strong labels
│   ├── audioset_eval_strong_framed_posneg.tsv  # Framed pos/neg labels
│   ├── mid_to_display_name.tsv     # Label ID to name mapping
│   ├── baby_cry/                   # Extracted baby cry metadata
│   ├── gun/                        # Extracted gun sound metadata
│   └── snore/                      # Extracted snore sound metadata
├── src/                            # Source code
│   ├── generate_target_sound_meta.py  # Main extraction script
│   └── analyze_label_distribution.py  # Distribution analysis script
├── out/                            # Analysis outputs
│   ├── label_distribution_analysis.png
│   ├── top_labels_detailed.png
│   └── label_distribution_stats.csv
├── scripts/                        # Utility scripts
├── configs/                        # Configuration files
└── requirements.txt               # Python dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yinkalario/audioset_strong.git
   cd audioset_strong
   ```

2. **Create and activate a virtual environment**:
   ```bash
   bash scripts/create_env.sh
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Extract Target Sound Metadata

The main script `generate_target_sound_meta.py` extracts specific sound types and categorizes them into overlapped and non-overlapped events.

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
python src/generate_target_sound_meta.py --input-dir meta --output-dir meta
```

**Output**: Creates `meta/{target_name}/` directory with 6 files:
- `{target_name}_nov_audioset_train_strong.tsv` - Non-overlapped training events
- `{target_name}_ov_audioset_train_strong.tsv` - Overlapped training events
- `{target_name}_nov_audioset_eval_strong.tsv` - Non-overlapped evaluation events
- `{target_name}_ov_audioset_eval_strong.tsv` - Overlapped evaluation events
- `{target_name}_nov_audioset_eval_strong_framed_posneg.tsv` - Non-overlapped framed events
- `{target_name}_ov_audioset_eval_strong_framed_posneg.tsv` - Overlapped framed events

### 2. Analyze Label Distribution

Analyze the distribution of all labels in the dataset by total duration:

```bash
python src/analyze_label_distribution.py --train-file meta/audioset_train_strong.tsv --mid-to-display meta/mid_to_display_name.tsv
```

**Output**:
- `out/label_distribution_analysis.png` - Comprehensive 6-panel analysis
- `out/top_labels_detailed.png` - Detailed visualization of top labels
- `out/label_distribution_stats.csv` - Complete statistics for all labels

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

### Sample Output Structure

```
meta/baby_cry/
├── baby_cry_nov_audioset_train_strong.tsv     # 64 events
├── baby_cry_ov_audioset_train_strong.tsv      # 1,703 events
├── baby_cry_nov_audioset_eval_strong.tsv      # 21 events
├── baby_cry_ov_audioset_eval_strong.tsv       # 180 events
├── baby_cry_nov_audioset_eval_strong_framed_posneg.tsv  # 64 events
└── baby_cry_ov_audioset_eval_strong_framed_posneg.tsv   # 259 events
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
