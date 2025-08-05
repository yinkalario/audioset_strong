#!/usr/bin/env python3
"""
AudioSet Dataset

PyTorch Dataset for flexible audio loading with configurable clip lengths
and intelligent preprocessing for different label types.

Key Features:
- Configurable clip duration and sample rate
- Smart cropping: random for weak labels, exact for strong labels
- Robust audio loading with automatic resampling
- Supports both CSV and Parquet metadata formats
- Returns normalized tensors with comprehensive metadata

Author: Yin Cao
"""

import random
from pathlib import Path
from typing import Tuple, Set

import pandas as pd
import torch
import torchaudio
import yaml
from torch.nn import functional as F


class AudioSetDataset(torch.utils.data.Dataset):
    """Map-style dataset for AudioSet training."""
    
    def __init__(self, metadata_path: str, config_path: str):
        """Initialize dataset.
        
        Args:
            metadata_path: Path to processed metadata parquet file
            config_path: Path to config YAML file
        """
        # Load configuration
        self.config = yaml.safe_load(open(config_path))
        
        # Audio parameters
        self.sample_rate = self.config["sample_rate"]
        self.clip_length = self.config["clip_length"]
        self.n_samples = int(self.sample_rate * self.clip_length)
        self.audio_root = Path(self.config.get("audio_root", "/path/to/audioset/audio"))
        
        # Load metadata (prefer parquet for efficiency, fallback to CSV)
        if metadata_path.endswith('.parquet'):
            self.df = pd.read_parquet(metadata_path)
        else:
            self.df = pd.read_csv(metadata_path)
            # Parse labels column (stored as string in CSV)
            self.df['labels'] = self.df['labels'].apply(self._parse_labels_from_csv)
        
        # Create index mappings for different data types
        self.pos_indices = self.df[self.df['is_positive'] == True].index.tolist()
        self.strong_neg_indices = self.df[
            (self.df['data_type'] == 'strong') & (self.df['is_positive'] == False)
        ].index.tolist()
        self.weak_neg_indices = self.df[self.df['data_type'] == 'weak'].index.tolist()
        
        print(f"[Dataset] Loaded {len(self.pos_indices)} positive, "
              f"{len(self.strong_neg_indices)} strong negative, "
              f"{len(self.weak_neg_indices)} weak negative samples")
        print(f"[Dataset] Clip length: {self.clip_length}s ({self.n_samples} samples)")
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Set[str], str]:
        """Get a single sample.
        
        Args:
            idx: Index into the dataset
            
        Returns:
            wav: Audio tensor [n_samples] 
            y: Binary label (1 for target sound, 0 for other)
            labels_mask: Set of negative label MIDs in this span
            clip_id: Unique identifier for this clip
        """
        row = self.df.iloc[idx]
        
        # Load audio
        try:
            wav = self._load_audio(row)
        except Exception as e:
            print(f"Warning: Failed to load audio for {row['clip_id']}: {e}")
            # Return silence if loading fails
            wav = torch.zeros(self.n_samples, dtype=torch.float32)
            return wav, int(row['is_positive']), set(row['labels']), row['clip_id']
        
        # Convert labels list to set (handle both list and numpy array)
        labels = row['labels']
        if hasattr(labels, '__len__') and len(labels) > 0:
            labels_set = set(labels)
        else:
            labels_set = set()
        
        return wav, int(row['is_positive']), labels_set, row['clip_id']
    
    def _load_audio(self, row: pd.Series) -> torch.Tensor:
        """Load and process audio for a sample."""
        # Use audio_path from processed metadata (handles AudioSet structure)
        audio_path = Path(row['audio_path'])

        # Load audio
        wav, sr = torchaudio.load(audio_path)
        wav = wav.squeeze(0)  # Remove channel dimension
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wav = resampler(wav)
        
        # Handle different clip types
        if row['data_type'] == 'weak':
            # Weak clip: random crop from 10-second segment
            wav = self._random_crop_weak(wav, row)
        else:
            # Strong clip: exact temporal slice
            wav = self._exact_slice_strong(wav, row)
        
        # Ensure exact length
        wav = self._ensure_length(wav)
        
        return wav.float()

    def _parse_labels_from_csv(self, labels_str: str) -> list:
        """Parse labels from CSV string representation."""
        if pd.isna(labels_str) or labels_str == '' or labels_str == '[]':
            return []

        # Handle string representation of list
        if labels_str.startswith('[') and labels_str.endswith(']'):
            # Remove brackets and quotes, split by comma
            labels_str = labels_str.strip('[]')
            if not labels_str:
                return []
            # Split and clean up quotes
            labels = [label.strip().strip("'\"") for label in labels_str.split(',')]
            return [label for label in labels if label]
        else:
            # Single label or comma-separated
            return [label.strip() for label in str(labels_str).split(',') if label.strip()]

    def _random_crop_weak(self, wav: torch.Tensor, row: pd.Series) -> torch.Tensor:
        """Random crop from weak (10-second) clip."""
        # Weak clips are 10 seconds, we want self.clip_length seconds
        total_duration = row['end_time'] - row['start_time']
        max_start_offset = total_duration - self.clip_length
        
        if max_start_offset <= 0:
            # Clip is shorter than desired length, use the whole clip
            start_sample = int(row['start_time'] * self.sample_rate)
            end_sample = int(row['end_time'] * self.sample_rate)
        else:
            # Random start within the valid range
            random_start_offset = random.uniform(0, max_start_offset)
            start_time = row['start_time'] + random_start_offset
            start_sample = int(start_time * self.sample_rate)
            end_sample = start_sample + self.n_samples
        
        return wav[start_sample:end_sample]
    
    def _exact_slice_strong(self, wav: torch.Tensor, row: pd.Series) -> torch.Tensor:
        """Exact temporal slice for strong clips."""
        start_sample = int(row['start_time'] * self.sample_rate)
        end_sample = int(row['end_time'] * self.sample_rate)
        return wav[start_sample:end_sample]
    
    def _ensure_length(self, wav: torch.Tensor) -> torch.Tensor:
        """Ensure audio is exactly the right length."""
        if wav.shape[0] < self.n_samples:
            # Pad if too short
            wav = F.pad(wav, (0, self.n_samples - wav.shape[0]))
        elif wav.shape[0] > self.n_samples:
            # Trim if too long
            wav = wav[:self.n_samples]
        
        return wav
    
    def get_steps_per_epoch(self, pos_per_batch: int) -> int:
        """Calculate steps per epoch based on positive samples."""
        return (len(self.pos_indices) + pos_per_batch - 1) // pos_per_batch
    
    def get_data_split_info(self) -> dict:
        """Get information about data splits."""
        return {
            'total_samples': len(self.df),
            'positive_samples': len(self.pos_indices),
            'strong_negative_samples': len(self.strong_neg_indices),
            'weak_negative_samples': len(self.weak_neg_indices),
            'clip_length': self.clip_length,
            'sample_rate': self.sample_rate
        }
