"""index_builder.py – Parse TSV/CSV metadata, apply head‑trim & √freq weighting,
then pickle all index structures used by dataset.py and sampler.py.
Run this **once** before training:
    python -m src.data.index_builder --config configs/baby_cry.yaml
"""

import argparse, pickle, random, yaml, csv, os
from collections import defaultdict, namedtuple
from math import sqrt

Row = namedtuple("Row", [
    "clip_id",      # str
    "t0",           # float (sec) – 0 for weak
    "t1",           # float (sec) – 1 for strong, 10 for weak
    "y",            # int 1 | 0
    "labels",       # list[str] – negative labels present in this 1‑s span
])


def load_strong_tsv(path: str, positive: bool):
    rows = []
    with open(path, "r", newline="") as f:
        for ln in csv.reader(f, delimiter="\t"):
            clip, start, end, mid = ln[0], float(ln[1]), float(ln[2]), ln[3]
            rows.append(Row(clip, start, end, int(positive), [] if positive else [mid]))
    return rows


def load_weak_csv(path: str):
    # AudioSet CSV header: YTID, start_seconds, end_seconds, pos_labels
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for ytid, start, end, labels in reader:
            first_label = labels.split(";")[0]
            clip_id = f"{ytid}_{int(float(start)*1000):05d}"
            rows.append(Row(clip_id, 0.0, 10.0, 0, [first_label]))
    return rows


def sqrt_norm(counts):
    labels = list(counts)
    weights = [sqrt(counts[l]) for l in labels]
    s = sum(weights)
    return labels, [w / s for w in weights]


def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))

    rows_pos, rows_sneg, rows_weak = [], [], []

    for p in cfg["pos_strong_paths"]:
        rows_pos.extend(load_strong_tsv(p, positive=True))

    for p in cfg["neg_strong_paths"]:
        rows_sneg.extend(load_strong_tsv(p, positive=False))

    for p in cfg["neg_weak_paths"]:
        rows_weak.extend(load_weak_csv(p))

    # ---------------- head‑class trimming on weak ----------------
    head_set = set(cfg["head_labels"])
    keep_p = cfg["head_keep_fraction"]
    trimmed_weak = [r for r in rows_weak if r.labels[0] not in head_set
                    or random.random() < keep_p]

    # ---------------- build per‑label dicts ----------------------
    lab2sneg = defaultdict(list)
    for idx, r in enumerate(rows_sneg):
        for lab in r.labels:
            lab2sneg[lab].append(idx)

    lab2weak = defaultdict(list)
    for idx, r in enumerate(trimmed_weak):
        lab2weak[r.labels[0]].append(idx)

    # bucket ultra‑rare labels
    cutoff = cfg["completion_rare_cutoff"]
    for d in (lab2sneg, lab2weak):
        rare_bucket = []
        rare_keys = [k for k, v in d.items() if len(v) < cutoff]
        for k in rare_keys:
            rare_bucket.extend(d.pop(k))
        if rare_bucket:
            d["other_rare"] = rare_bucket

    sneg_labels, sneg_p = sqrt_norm({k: len(v) for k, v in lab2sneg.items()})
    weak_labels, weak_p = sqrt_norm({k: len(v) for k, v in lab2weak.items()})

    out = dict(
        pos=rows_pos,
        sneg=rows_sneg,
        weak=trimmed_weak,
        label2sneg=lab2sneg,
        label2weak=lab2weak,
        sneg_labs=sneg_labels,
        sneg_p=sneg_p,
        weak_labs=weak_labels,
        weak_p=weak_p,
    )

    os.makedirs("artifacts", exist_ok=True)
    pickle.dump(out, open("artifacts/index.pkl", "wb"))
    print("[index_builder] wrote artifacts/index.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baby_cry.yaml")
    main(ap.parse_args().config)