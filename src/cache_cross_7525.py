# -*- coding: utf-8 -*-
"""
Regenerate the CURE / OGYEI query-imprint caches with the 75/25 reader.

The eval_predictions_*.csv files embed ABSOLUTE image paths from the machine
that produced them (D:\CURE_crops, D:\ogyeiv2_crops); the data now lives under
D:\Data\... . We remap paths ONLY for reading the file from disk, but store the
ORIGINAL path string as the cache key so that downstream lookups (cure_fusion,
rebuttal_stats2.eval_cross) keyed on the prediction-CSV strings still match.

Fails loudly if more than 5% of reads error or if every imprint comes back
empty (the all-empty-cache failure mode must never be silent again).

    python run_bigstack.py cache_cross_7525 --dataset cure
    python run_bigstack.py cache_cross_7525 --dataset ogyei
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_rerank import load_holdout
from imprint_feasibility import QwenReader, normalize_imprint

CFG = {
    "cure": dict(pred="outputs/eval_predictions_cure_val.csv",
                 old_root="D:/CURE_crops", new_root="D:/Data/CURE_crops",
                 out="fusion_out/cure_imprints_holdout_7525.csv"),
    "ogyei": dict(pred="outputs/eval_predictions_ogyei_val.csv",
                  old_root="D:/ogyeiv2_crops", new_root="D:/Data/ogyeiv2_crops",
                  out="fusion_out/ogyei_imprints_7525.csv"),
}


def main():
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CFG), required=True)
    ap.add_argument("--adapter", default="qwen_imprint_lora_7525")
    args = ap.parse_args()
    cfg = CFG[args.dataset]

    _, S, correct, paths = load_holdout(cfg["pred"])
    uniq = sorted({p for imgs in paths for p in imgs})
    print(f"[{args.dataset}] {len(paths)} queries, {len(uniq)} unique images")

    old, new = cfg["old_root"], cfg["new_root"]

    def on_disk(p):
        q = str(p).replace("\\", "/")
        if q.startswith(old + "/"):
            q = new + q[len(old):]
        return q

    remapped = sum(1 for p in uniq if on_disk(p) != str(p).replace("\\", "/"))
    print(f"[remap] {remapped}/{len(uniq)} paths remapped {old} -> {new}")

    reader = QwenReader("Qwen/Qwen2-VL-2B-Instruct", "cuda", adapter=args.adapter)
    rows, errs = [], 0
    for i, p in enumerate(uniq):
        try:
            imp = normalize_imprint(reader.read(Image.open(on_disk(p))))
        except Exception as e:
            imp = ""; errs += 1
            print(f"  [warn] {p}: {e}")
        rows.append({"image": p, "imprint": imp})
        if (i + 1) % 100 == 0:
            print(f"  [imprint] {i+1}/{len(uniq)}", flush=True)

    nonempty = sum(1 for r in rows if r["imprint"].strip())
    print(f"[{args.dataset}] reads: {nonempty}/{len(rows)} non-empty, {errs} errors")
    if errs > 0.05 * len(rows):
        sys.exit(f"[FATAL] {errs} read errors (> 5%) - refusing to write cache")
    if nonempty == 0:
        sys.exit("[FATAL] every imprint empty - refusing to write cache")
    os.makedirs("fusion_out", exist_ok=True)
    pd.DataFrame(rows).to_csv(cfg["out"], index=False)
    print(f"[cache] -> {cfg['out']}")


if __name__ == "__main__":
    main()
