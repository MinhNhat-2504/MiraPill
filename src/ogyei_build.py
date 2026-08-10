# -*- coding: utf-8 -*-
"""
OGYEIv2 prep (no-harm / Prop-1 validation dataset — imprint-LESS Hungarian pills).
Crops each single-pill image to 224 and writes an ePillID-schema manifest.
Structure: <root>/{train,valid,test}/images/{drug}_{s|u}_NNN.jpg
  class = drug name; domain s -> is_ref(gallery), u -> query; no top/bottom sides.
Output: D:\\ogyeiv2_crops\\... + manifest.csv  (+ folds via ogyei_make_folds-style here)
"""
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd

from cure_crop import crop_pill


def drug_of(fn):
    return re.sub(r"_[su]_\d+$", "", os.path.splitext(os.path.basename(fn))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\ogyeiv2\ogyeiv2\ogyeiv2")
    ap.add_argument("--dst", default=r"D:\ogyeiv2_crops")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n_test", type=int, default=24)
    ap.add_argument("--n_val", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    imgs = [f for f in glob.glob(os.path.join(args.src, "**", "images", "*.*"), recursive=True)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[ogyei] {len(imgs)} images")
    # encode drug -> integer class id (stable, sorted)
    drugs = sorted({drug_of(f) for f in imgs})
    d2id = {d: i for i, d in enumerate(drugs)}
    print(f"[ogyei] {len(drugs)} drug classes")

    rows, nfail = [], 0
    for i, f in enumerate(imgs):
        d = drug_of(f); cid = d2id[d]
        is_ref = "_s_" in os.path.basename(f)
        rel = os.path.join(str(cid), "s" if is_ref else "u",
                           os.path.splitext(os.path.basename(f))[0] + ".png")
        out = os.path.join(args.dst, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        try:
            img, method, found = crop_pill(f, size=args.size)
            img.save(out)
            if not found:
                nfail += 1
        except Exception as e:
            nfail += 1; print(f"  [warn] {f}: {e}"); continue
        rows.append({"images": rel.replace(os.sep, "/"), "pilltype_id": cid,
                     "label_code_id": cid, "prod_code_id": 0,
                     "is_ref": bool(is_ref), "is_front": True, "is_new": False,
                     "image_path": rel.replace(os.sep, "/"), "label": str(cid),
                     "drug": d})
        if (i + 1) % 500 == 0:
            print(f"  [ogyei] {i+1}/{len(imgs)} (fallback {nfail})", flush=True)

    df = pd.DataFrame(rows)
    # folds: hold out test/val CLASSES (their u-images = unseen queries); s stays gallery
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(sorted(df["pilltype_id"].unique()))
    test_cls = set(perm[:args.n_test].tolist())
    val_cls = set(perm[args.n_test:args.n_test + args.n_val].tolist())
    u = df[~df["is_ref"]]
    out_base = os.path.join(args.dst, "folds", "ogyei", "base")
    os.makedirs(out_base, exist_ok=True)
    cols = ["images", "pilltype_id", "label_code_id", "prod_code_id",
            "is_ref", "is_front", "is_new", "image_path", "label"]
    df[cols].to_csv(os.path.join(out_base, "ogyei_all.csv"), index=False)
    u[u["pilltype_id"].isin(val_cls)][cols].to_csv(os.path.join(out_base, "ogyei_val.csv"), index=False)
    u[u["pilltype_id"].isin(test_cls)][cols].to_csv(os.path.join(out_base, "ogyei_test.csv"), index=False)
    df.to_csv(os.path.join(args.dst, "manifest.csv"), index=False)

    from collections import Counter
    print(f"\n[ogyei] DONE crops={len(df)} fallback={nfail} methods={dict(Counter(df.get('crop_method',[])))}")
    print(f"[ogyei] classes={df['pilltype_id'].nunique()} ref(s)={int(df['is_ref'].sum())} "
          f"query(u)={int((~df['is_ref']).sum())}")
    print(f"[ogyei] test classes={len(test_cls)} val={len(val_cls)} -> {out_base}")


if __name__ == "__main__":
    main()
