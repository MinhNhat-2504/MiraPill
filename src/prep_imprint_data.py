# -*- coding: utf-8 -*-
"""
Experiment A, step 1 — build the fine-tuning dataset for the imprint reader.

Design: TWO-IMAGE input (front + back) -> FULL pill imprint target.
Rationale: ePillID images are single-sided and many faces are blank/score-only,
so we cannot reliably label per-side text. Feeding BOTH faces and asking for the
complete imprint gives clean supervision and matches the both-sides eval.

Training pairs come from REFERENCE (studio) images only (clean, reliable). The
held-out CONSUMER images stay untouched for evaluation (the hard domain).

Splits are by pill type (label_prod_code) -> no leakage between train/val.

    python prep_imprint_data.py --gt_csv "D:\\pillbox_kaggle\\Pillbox.csv" \\
        --data_root_dir "D:/ePillID_data/ePillID_data" --out_dir imprint_ft_data
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from imprint_feasibility import (load_ground_truth_imprints, _load_fold_with_keys,
                                 clean_gt_imprint, normalize_imprint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root_dir", default="/mydata")
    ap.add_argument("--img_dir", default="classification_data")
    ap.add_argument("--all_imgs_csv",
                    default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/"
                            "pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--out_dir", default="imprint_ft_data")
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = _load_fold_with_keys(args)                  # images + label_prod_code + full_path
    gt = load_ground_truth_imprints(args.gt_csv)     # label_prod_code + cleaned splimprint
    gt_map = dict(zip(gt["label_prod_code"], gt["splimprint"]))

    ref = df[df["is_ref"]].copy()
    ref = ref[ref["label_prod_code"].isin(gt_map)]
    has_side = "is_front" in ref.columns

    samples = []
    for lpc, g in ref.groupby("label_prod_code"):
        target = normalize_imprint(clean_gt_imprint(gt_map[lpc]))
        if not target:
            continue
        if has_side:
            g = g.sort_values("is_front", ascending=False)  # front first
        imgs = list(dict.fromkeys(g["full_path"].tolist()))   # unique, keep order
        imgs = imgs[:2]                                       # front + back
        if not imgs:
            continue
        samples.append({"label_prod_code": lpc, "images": imgs, "target": target})

    # split by pill type (samples are already one-per-type)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(samples))
    n_val = int(len(samples) * args.val_frac)
    val_idx, train_idx = set(idx[:n_val]), set(idx[n_val:])
    train = [samples[i] for i in range(len(samples)) if i in train_idx]
    val = [samples[i] for i in range(len(samples)) if i in val_idx]

    def dump(rows, name):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    tp, vp = dump(train, "train.jsonl"), dump(val, "val.jsonl")
    n2 = sum(1 for s in samples if len(s["images"]) == 2)
    print(f"[prep] pill types with imprint GT: {len(samples)}")
    print(f"[prep]   with 2 side images: {n2}  | with 1: {len(samples) - n2}")
    print(f"[prep] train: {len(train)} -> {tp}")
    print(f"[prep] val:   {len(val)} -> {vp}")
    print("[prep] examples:")
    for s in samples[:5]:
        print(f"   {s['label_prod_code']:>10} | {len(s['images'])} img | target='{s['target']}'")


if __name__ == "__main__":
    main()
