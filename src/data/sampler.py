#!/usr/bin/env python3
"""
Two-Tier Batch Sampler

Advanced batch sampler with intelligent negative sampling strategies
for balanced training and hard negative mining.

Key Features:
- Configurable batch composition with fixed ratios
- Two-tier negative sampling: balanced + hard negatives
- Label coverage guarantees (all labels seen per epoch)
- Hard negative mining with adaptive focus
- Multi-GPU support with deterministic data partitioning
- Efficient sampling with √-frequency weighting

Author: Yin Cao
"""
import random
from collections import deque
from typing import List, Iterator, Dict, Optional

import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Sampler


class TwoTierBatchSampler(Sampler):
    """Two-tier batch sampler for AudioSet training."""

    def __init__(self, dataset, batch_size_per_device: int, metadata_path: str,
                 mappings_path: str, config_path: str, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, seed: int = 0, drop_last: bool = False):
        """Initialize the sampler.

        Args:
            dataset: The dataset to sample from
            batch_size_per_device: Batch size per device/GPU
            metadata_path: Path to processed metadata parquet file
            mappings_path: Path to label mappings pickle file
            config_path: Path to config YAML file
            num_replicas: Number of processes participating in distributed training
            rank: Rank of the current process within num_replicas
            seed: Random seed used to shuffle the sampler
            drop_last: If True, drop the last incomplete batch
        """
        super().__init__(dataset)

        if num_replicas is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the range [0, {}]".format(
                    rank, num_replicas - 1))

        self.dataset = dataset
        self.batch_size_per_device = batch_size_per_device
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.drop_last = drop_last
        # Load configuration
        self.config = yaml.safe_load(open(config_path))

        # Load metadata
        if metadata_path.endswith('.parquet'):
            self.df = pd.read_parquet(metadata_path)
        else:
            self.df = pd.read_csv(metadata_path)
            # Parse labels column (stored as string in CSV)
            self.df['labels'] = self.df['labels'].apply(self._parse_labels_from_csv)

        # Load label mappings from pickle
        self.mappings = pd.read_pickle(mappings_path)

        # Extract data type indices
        pos_indices = self.df[self.df['is_positive']].index.tolist()
        strong_neg_indices = self.df[
            (self.df['data_type'] == 'strong') & (~self.df['is_positive'])
        ].index.tolist()
        weak_neg_indices = self.df[self.df['data_type'] == 'weak'].index.tolist()

        # Apply DDP stride partitioning
        self.pos_indices = pos_indices[self.rank::self.num_replicas]
        self.strong_neg_indices = strong_neg_indices[self.rank::self.num_replicas]
        self.weak_neg_indices = weak_neg_indices[self.rank::self.num_replicas]

        # Apply DDP partitioning to label mappings
        self.strong_neg_map = self._partition_label_map(
            self.mappings['strong_neg_map'], self.rank, self.num_replicas
        )
        self.weak_neg_map = self._partition_label_map(
            self.mappings['weak_neg_map'], self.rank, self.num_replicas
        )

        # Sampling weights (same across all ranks)
        self.strong_labels = self.mappings['strong_labels']
        self.strong_probs = self.mappings['strong_probs']
        self.weak_labels = self.mappings['weak_labels']
        self.weak_probs = self.mappings['weak_probs']

        # Batch composition (use batch_size_per_device)
        batch_size = self.batch_size_per_device
        self.pos_per_batch = int(batch_size * self.config["pos_per_batch_frac"])
        self.strong_neg_per_batch = int(batch_size * self.config["strong_neg_per_batch_frac"])
        self.weak_neg_per_batch = int(batch_size * self.config["weak_neg_per_batch_frac"])

        # Ensure batch adds up correctly
        total = self.pos_per_batch + self.strong_neg_per_batch + self.weak_neg_per_batch
        if total != batch_size:
            self.weak_neg_per_batch = batch_size - self.pos_per_batch - self.strong_neg_per_batch

        # Two-tier parameters
        self.tierA_count = int(self.strong_neg_per_batch * self.config["tierA_fraction"])
        self.tierB_count = self.strong_neg_per_batch - self.tierA_count
        self.primary_fraction = self.config["primary_fraction"]

        # Hard negative buffer
        self.hard_buffer = deque(maxlen=self.config["hard_buffer_size"])

        # Calculate steps per epoch based on positive samples
        self.steps_per_epoch = ((len(self.pos_indices) + self.pos_per_batch - 1)
                                // self.pos_per_batch)

        print(f"[Sampler] Rank {self.rank}: {len(self.pos_indices)} pos, "
              f"{len(self.strong_neg_indices)} strong neg, "
              f"{len(self.weak_neg_indices)} weak neg")
        print(f"[Sampler] Batch: {self.pos_per_batch} pos + "
              f"{self.strong_neg_per_batch} strong ({self.tierA_count}A+{self.tierB_count}B) + "
              f"{self.weak_neg_per_batch} weak")
        print(f"[Sampler] Steps per epoch: {self.steps_per_epoch}")

        self.set_epoch(0)

    def __len__(self) -> int:
        """Return the number of samples per epoch (number of batches * batch_size_per_device)."""
        return self.steps_per_epoch * self.batch_size_per_device

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

    def _partition_label_map(self, label_map: Dict, rank: int, world_size: int) -> Dict:
        """Apply DDP partitioning to label mappings."""
        partitioned = {}
        for label, indices in label_map.items():
            partitioned[label] = indices[rank::world_size]
        return partitioned

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch

        # Deterministic but different seed per rank and epoch
        seed = self.seed + self.rank + 31 * epoch
        random.seed(seed)
        np.random.seed(seed)

        # Reset strong negative pointers and unseen labels
        self.strong_pointers = {label: 0 for label in self.strong_labels}
        self.unseen_labels = set(self.strong_labels)

        # Shuffle all label index lists
        for indices in self.strong_neg_map.values():
            random.shuffle(indices)

        # Create shuffled copies for this epoch
        self.pos_indices_epoch = self.pos_indices.copy()
        self.weak_indices_epoch = self.weak_neg_indices.copy()
        random.shuffle(self.pos_indices_epoch)
        random.shuffle(self.weak_indices_epoch)

        # Reset position in positive samples
        self.pos_position = 0

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches for one epoch."""
        for _ in range(self.steps_per_epoch):
            batch_indices = []

            # Draw positive samples
            batch_indices.extend(self._draw_positives())

            # Draw strong negative samples (Tier A + Tier B)
            batch_indices.extend(self._draw_strong_negatives())

            # Draw weak negative samples
            batch_indices.extend(self._draw_weak_negatives())

            # Shuffle batch to mix sample types
            random.shuffle(batch_indices)

            yield batch_indices

    def _draw_positives(self) -> List[int]:
        """Draw positive samples for this batch."""
        start_pos = self.pos_position
        end_pos = min(start_pos + self.pos_per_batch, len(self.pos_indices_epoch))

        batch_pos = self.pos_indices_epoch[start_pos:end_pos]
        self.pos_position = end_pos

        # If we don't have enough positives, cycle through again
        while len(batch_pos) < self.pos_per_batch:
            remaining = self.pos_per_batch - len(batch_pos)
            random.shuffle(self.pos_indices_epoch)
            batch_pos.extend(self.pos_indices_epoch[:remaining])
            self.pos_position = remaining

        return batch_pos

    def _pop_strong_negative(self, label: str) -> int:
        """Pop next strong negative sample for given label."""
        if label not in self.strong_neg_map or not self.strong_neg_map[label]:
            # Fallback to random strong negative if label is empty
            if self.strong_neg_indices:
                return random.choice(self.strong_neg_indices)
            else:
                # Last resort: return a random index (will be handled by dataset)
                return 0

        indices = self.strong_neg_map[label]
        idx = indices[self.strong_pointers[label]]
        self.strong_pointers[label] = (self.strong_pointers[label] + 1) % len(indices)
        return idx

    def _draw_strong_negatives(self) -> List[int]:
        """Draw strong negative samples (Tier A + Tier B)."""
        batch_strong = []

        # Tier A: √freq + completion pass
        primary_count = int(self.tierA_count * self.primary_fraction)

        # Primary pass: sample by √freq
        if self.strong_labels and primary_count > 0:
            labels = np.random.choice(
                self.strong_labels,
                size=primary_count,
                p=self.strong_probs,
                replace=True
            )
            for label in labels:
                batch_strong.append(self._pop_strong_negative(label))
                self.unseen_labels.discard(label)

        # Completion pass: ensure every label appears at least once per epoch
        while len(batch_strong) < self.tierA_count and self.unseen_labels:
            label = self.unseen_labels.pop()
            batch_strong.append(self._pop_strong_negative(label))

        # Fill remaining Tier A slots
        while len(batch_strong) < self.tierA_count:
            if self.strong_labels:
                label = np.random.choice(self.strong_labels, p=self.strong_probs)
                batch_strong.append(self._pop_strong_negative(label))
            else:
                if self.strong_neg_indices:
                    batch_strong.append(random.choice(self.strong_neg_indices))
                else:
                    batch_strong.append(0)  # Fallback

        # Tier B: hard negatives buffer
        hard_count = min(self.tierB_count, len(self.hard_buffer))
        for _ in range(hard_count):
            if self.hard_buffer:
                batch_strong.append(self.hard_buffer.popleft())

        # Fill remaining Tier B slots
        while len(batch_strong) < self.strong_neg_per_batch:
            if self.strong_labels:
                label = np.random.choice(self.strong_labels, p=self.strong_probs)
                batch_strong.append(self._pop_strong_negative(label))
            else:
                if self.strong_neg_indices:
                    batch_strong.append(random.choice(self.strong_neg_indices))
                else:
                    batch_strong.append(0)  # Fallback

        return batch_strong

    def _draw_weak_negatives(self) -> List[int]:
        """Draw weak negative samples by √freq."""
        batch_weak = []

        if self.weak_labels and self.weak_neg_per_batch > 0:
            labels = np.random.choice(
                self.weak_labels,
                size=self.weak_neg_per_batch,
                p=self.weak_probs,
                replace=True
            )

            for label in labels:
                if label in self.weak_neg_map and self.weak_neg_map[label]:
                    idx = random.choice(self.weak_neg_map[label])
                    batch_weak.append(idx)
                else:
                    # Fallback if label has no samples
                    if self.weak_neg_indices:
                        batch_weak.append(random.choice(self.weak_neg_indices))
                    else:
                        batch_weak.append(0)  # Fallback

        # Fill any remaining slots
        while len(batch_weak) < self.weak_neg_per_batch:
            if self.weak_neg_indices:
                batch_weak.append(random.choice(self.weak_neg_indices))
            else:
                batch_weak.append(0)  # Fallback

        return batch_weak

    def extend_hard_buffer(self, new_indices: List[int]) -> None:
        """Add new hard negative indices to the buffer."""
        for idx in new_indices:
            if idx not in self.hard_buffer:
                self.hard_buffer.appendleft(idx)

    def get_steps_per_epoch(self) -> int:
        """Get number of steps per epoch."""
        return self.steps_per_epoch
