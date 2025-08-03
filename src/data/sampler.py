"""sampler.py – Two‑tier, two‑pass BatchSampler implementing:
    • Fixed per‑batch ratio (pos, strong‑neg TierA/TierB, weak‑neg)
    • Tier A: √freq + completion guarantee
    • Tier B: hard‑negative buffer
    • Weak‑neg: √freq after head‑trim
Deterministic across epochs & multi‑GPU via stride partition + epoch seed.
"""

import random, numpy as np, pickle, yaml
from collections import deque

class TwoTierBatchSampler:
    def __init__(self, index_pkl: str, cfg_yaml: str, world=1, rank=0):
        meta    = pickle.load(open(index_pkl, "rb"))
        cfg     = yaml.safe_load(open(cfg_yaml))

        # stride‑split
        self.pos   = meta["pos"][rank::world]
        self.sneg  = meta["sneg"][rank::world]
        self.weak  = meta["weak"][rank::world]

        self.lab2s = {k: v[rank::world] for k, v in meta["label2sneg"].items()}
        self.lab2w = {k: v[rank::world] for k, v in meta["label2weak"].items()}
        self.s_labs, self.s_p = meta["sneg_labs"], meta["sneg_p"]
        self.w_labs, self.w_p = meta["weak_labs"], meta["weak_p"]

        # hyper‑params
        self.bs   = cfg["batch_size"]
        self.Kp   = cfg["pos_per_batch"]
        self.Ks   = cfg["strong_neg_per_batch"]
        self.Kw   = cfg["weak_neg_per_batch"]
        self.Ka   = int(self.Ks * cfg["tierA_fraction"])
        self.Kb   = self.Ks - self.Ka
        self.primary_frac = cfg["primary_fraction"]
        self.max_buf = cfg["hard_buffer_size"]

        self.rank, self.world = rank, world
        self.hard_buf = deque(maxlen=self.max_buf)
        self.set_epoch(0)

    # ---------- epoch boundary ----------
    def set_epoch(self, epoch):
        self.epoch = epoch
        seed = 2025 + 31*self.rank + 97*epoch
        random.seed(seed); np.random.seed(seed)
        self.ptr = {lab: 0 for lab in self.s_labs}
        self.unseen = set(self.s_labs)
        for lst in self.lab2s.values(): random.shuffle(lst)
        random.shuffle(self.pos); random.shuffle(self.weak)

    # ---------- iterator ----------
    def __iter__(self):
        while True:
            batch = (self._draw_pos() + self._draw_sneg() + self._draw_weak())
            random.shuffle(batch)
            yield batch

    # ---------- per‑component draws ----------
    def _draw_pos(self):
        draw = [self.pos.pop() for _ in range(self.Kp)]
        if len(self.pos) < self.Kp: random.shuffle(self.pos)
        return draw

    def _pop_clip(self, lab):
        lst = self.lab2s[lab]
        clip = lst[self.ptr[lab]]
        self.ptr[lab] = (self.ptr[lab] + 1) % len(lst)
        return clip

    def _draw_sneg(self):
        out = []
        K_primary = int(self.Ka * self.primary_frac)
        labs = np.random.choice(self.s_labs, K_primary, p=self.s_p, replace=True)
        for l in labs:
            out.append(self._pop_clip(l)); self.unseen.discard(l)
        while len(out) < self.Ka and self.unseen:
            l = self.unseen.pop(); out.append(self._pop_clip(l))
        while len(out) < self.Ka:
            l = np.random.choice(self.s_labs, p=self.s_p)
            out.append(self._pop_clip(l))
        take = min(self.Kb, len(self.hard_buf))
        for _ in range(take): out.append(self.hard_buf.pop())
        while len(out) < self.Ks:
            l = np.random.choice(self.s_labs, p=self.s_p)
            out.append(self._pop_clip(l))
        return out

    def _draw_weak(self):
        labs = np.random.choice(self.w_labs, self.Kw, p=self.w_p, replace=True)
        return [random.choice(self.lab2w[l]) for l in labs]

    # ---------- hard‑neg API ----------
    def extend_hard_buffer(self, new_ids):
        for cid in new_ids:
            if cid not in self.hard_buf:
                self.hard_buf.appendleft(cid)
