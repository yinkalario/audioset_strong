"""dataset.py – Map‑style PyTorch Dataset that returns 1‑s waveforms.
Supports both strong (1‑s exact) and weak (10‑s random‑crop) rows.
"""

import random, torchaudio, torch, pickle, yaml
from torch.nn import functional as F
from pathlib import Path

class AudioSegmentDataset(torch.utils.data.Dataset):
    def __init__(self, index_pkl: str, config_yaml: str):
        meta = pickle.load(open(index_pkl, "rb"))
        self.rows = meta["pos"] + meta["sneg"] + meta["weak"]
        self.sr = yaml.safe_load(open(config_yaml))["sample_rate"]
        self.n_samples = int(self.sr * yaml.safe_load(open(config_yaml))["clip_length"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        wav, _ = torchaudio.load(self._to_path(r.clip_id))
        wav = wav.squeeze(0)

        if r.t1 - r.t0 == 10.0:               # weak clip – random crop
            start = random.uniform(0, 9)
            wav = wav[int(start*self.sr): int((start+1)*self.sr)]
        else:                                 # strong – exact slice
            wav = wav[int(r.t0*self.sr): int(r.t1*self.sr)]

        if wav.shape[0] < self.n_samples:
            wav = F.pad(wav, (0, self.n_samples - wav.shape[0]))
        return wav, r.y, r.labels, r.clip_id

    def _to_path(self, clip_id: str):
        # Implement: map clip_id to actual .wav on disk / yt‑cache path.
        return Path("/path/to/wavs") / f"{clip_id}.wav"
