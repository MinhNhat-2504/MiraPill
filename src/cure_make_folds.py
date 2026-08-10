# -*- coding: utf-8 -*-
"""
CURE step 2: from the crop manifest, write ePillID-schema CSVs so the existing
train_nocv pipeline runs unchanged. Mirrors ePillID's low-shot protocol:
held-out TEST classes' Customer images are queries (never trained); ALL Reference
images stay in the gallery; remaining Customer images train the embedding.

Outputs under <dst>/folds/cure/base/:  cure_all.csv, cure_val.csv, cure_test.csv
"""
import argparse
import os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=r"D:\CURE_crops\manifest.csv")
    ap.add_argument("--dst", default=r"D:\CURE_crops")
    ap.add_argument("--n_test", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seen_split", action="store_true",
                    help="within-dataset (deployment) eval: hold out CUSTOMER IMAGES "
                         "per class instead of whole classes (all classes seen)")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--val_frac", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df["is_ref"] = df["is_ref"].astype(bool)
    rng = np.random.RandomState(args.seed)

    if args.seen_split:
        # Drop classes lacking a Reference from the WHOLE dataset so the gallery
        # covers every (re-encoded) class -> consistent label indexing in eval.
        ref_classes = set(df[df["is_ref"]]["pilltype_id"])
        df = df[df["pilltype_id"].isin(ref_classes)].copy()  # keep original ids
        cust = df[~df["is_ref"]]
        te, va = [], []
        for cls, g in cust.groupby("pilltype_id"):
            idx = rng.permutation(g.index.values)
            nt = max(1, int(len(idx) * args.test_frac))
            nv = max(1, int(len(idx) * args.val_frac))
            te += list(idx[:nt]); va += list(idx[nt:nt + nv])
        test_df = cust.loc[te]; val_df = cust.loc[va]
        classes = sorted(df["pilltype_id"].unique())
        test_cls = set(classes); val_cls = set(classes)
        print(f"[folds] SEEN-CLASS split; {len(ref_classes)} ref'd classes "
              f"(dropped classes without reference)")
    else:
        classes = sorted(df["pilltype_id"].unique(), key=lambda x: int(x))
        cust = df[~df["is_ref"]]
        perm = rng.permutation(classes)
        test_cls = set(perm[:args.n_test].tolist())
        val_cls = set(perm[args.n_test:args.n_test + args.n_val].tolist())
        test_df = cust[cust["pilltype_id"].isin(test_cls)]
        val_df = cust[cust["pilltype_id"].isin(val_cls)]

    sub = "cure_seen" if args.seen_split else "cure"
    out = os.path.join(args.dst, "folds", sub, "base")
    os.makedirs(out, exist_ok=True)
    cols = ["images", "pilltype_id", "label_code_id", "prod_code_id",
            "is_ref", "is_front", "is_new", "image_path", "label"]
    df[cols].to_csv(os.path.join(out, "cure_all.csv"), index=False)
    val_df[cols].to_csv(os.path.join(out, "cure_val.csv"), index=False)
    test_df[cols].to_csv(os.path.join(out, "cure_test.csv"), index=False)

    print(f"[folds] all={len(df)}  (ref={int(df['is_ref'].sum())}, "
          f"cust={int((~df['is_ref']).sum())})")
    print(f"[folds] classes total={len(classes)}  test={len(test_cls)}  val={len(val_cls)}  "
          f"train={len(classes)-len(test_cls)-len(val_cls)}")
    print(f"[folds] val queries={len(val_df)}  test queries={len(test_df)}")
    print(f"[folds] written to {out}")


if __name__ == "__main__":
    main()
