# -*- coding: utf-8 -*-
"""Generate OGYEIv2 SEEN-class folds (image-split, all classes seen) from the
existing crop manifest — OGYEIv2's native within-dataset protocol."""
import os, numpy as np, pandas as pd
M = r"D:\ogyeiv2_crops\manifest.csv"
df = pd.read_csv(M); df["is_ref"] = df["is_ref"].astype(bool)
cust = df[~df["is_ref"]]; rng = np.random.RandomState(42)
te, va = [], []
for cls, g in cust.groupby("pilltype_id"):
    idx = rng.permutation(g.index.values)
    nt = max(1, int(len(idx) * 0.2)); nv = max(1, int(len(idx) * 0.1))
    te += list(idx[:nt]); va += list(idx[nt:nt + nv])
out = r"D:\ogyeiv2_crops\folds\ogyei_seen\base"; os.makedirs(out, exist_ok=True)
cols = ["images", "pilltype_id", "label_code_id", "prod_code_id", "is_ref",
        "is_front", "is_new", "image_path", "label"]
df[cols].to_csv(os.path.join(out, "ogyei_all.csv"), index=False)
cust.loc[va][cols].to_csv(os.path.join(out, "ogyei_val.csv"), index=False)
cust.loc[te][cols].to_csv(os.path.join(out, "ogyei_test.csv"), index=False)
print(f"SEEN folds: all={len(df)} test_q={len(te)} val_q={len(va)} classes={df['pilltype_id'].nunique()} -> {out}")
