# -*- coding: utf-8 -*-
"""
CURE step 3: build the per-class imprint DB by OCR-consensus. CURE has no imprint
metadata, so we read each class's clean REFERENCE crops (top+bottom) with the
ePillID-fine-tuned Qwen reader and pool the tokens -> the class's imprint
signature. This replaces Pillbox as the imprint DB for CURE fusion.

Output: <dst>/cure_imprint_db.csv  (columns: pilltype_id, imprint)
"""
import argparse
import os
import pandas as pd
from PIL import Image

from imprint_feasibility import QwenReader, normalize_imprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"D:\CURE_crops\manifest.csv")
    ap.add_argument("--crops_root", default=r"D:\CURE_crops")
    ap.add_argument("--adapter", default="qwen_imprint_lora")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=r"D:\CURE_crops\cure_imprint_db.csv")
    ap.add_argument("--max_ref_per_class", type=int, default=0,
                    help="cap reference reads per class (0=all); OGYEIv2 has ~20")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df["is_ref"] = df["is_ref"].astype(bool)
    ref = df[df["is_ref"]]
    if args.max_ref_per_class > 0:
        ref = ref.groupby("pilltype_id", group_keys=False).head(args.max_ref_per_class)
    print(f"[db] reference crops: {len(ref)} over {ref['pilltype_id'].nunique()} classes")

    reader = QwenReader(args.qwen_model, args.device, adapter=args.adapter)
    rows = []
    for cls, g in ref.groupby("pilltype_id"):
        toks = []
        for _, r in g.iterrows():
            p = os.path.join(args.crops_root, r["image_path"].replace("/", os.sep))
            try:
                toks.append(normalize_imprint(reader.read(Image.open(p))))
            except Exception as e:
                print(f"  [warn] {p}: {e}")
        pooled = normalize_imprint(" ".join(t for t in toks if t))
        # dedup tokens, keep order
        pooled = " ".join(dict.fromkeys(pooled.split()))
        rows.append({"pilltype_id": cls, "imprint": pooled})
        if len(rows) % 40 == 0:
            print(f"  [db] {len(rows)} classes...", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    n_have = int((out["imprint"].str.strip() != "").sum())
    print(f"\n[db] DONE. {len(out)} classes, {n_have} with non-empty imprint "
          f"({100*n_have/len(out):.1f}%)")
    print(f"[db] saved -> {args.out}")
    print("[db] samples:")
    for _, r in out.head(8).iterrows():
        print(f"   class {r['pilltype_id']}: '{r['imprint']}'")


if __name__ == "__main__":
    main()
