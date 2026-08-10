# -*- coding: utf-8 -*-
"""
E3 — failure-case study. Quantifies and illustrates the cases the imprint branch
rescues: queries the visual model ranks wrong but MIRA-Pill ranks right (the
near-identical-imprint failures Usuyama 2020 Fig.4 / IEEEtran flagged).

Aggregates rescue/hurt counts over all 4 folds; dumps qualitative examples
(read imprint vs the wrong visual candidate's imprint vs the true imprint).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from fusion_rerank import (load_holdout, build_class_imprints, imprint_match)

BASE = ("D:/ePillID_data/ePillID_data/folds/"
        "pilltypeid_nih_sidelbls0.01_metric_5folds/base")


def fuse_scores(S, q_imprint, class_imp, lam, topk):
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(S)):
        tq = q_imprint[i]
        if not tq:
            continue
        for c in np.argsort(-S[i])[:topk]:
            Sf[i, c] += lam * imprint_match(tq, class_imp[c])
    return Sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_encoder", default=f"{BASE}/label_encoder_local.pickle")
    ap.add_argument("--gt_csv", default="D:/pillbox_kaggle/Pillbox.csv")
    ap.add_argument("--all_imgs_csv",
                    default=f"{BASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    ap.add_argument("--imprint_cache", default="fusion_out/imprints_holdout.csv")
    ap.add_argument("--example_fold",
                    default="outputs/eval_predictions_pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--n_examples", type=int, default=12)
    args = ap.parse_args()

    classes, class_imp = build_class_imprints(args.label_encoder, args.gt_csv,
                                              args.all_imgs_csv)
    cache = pd.read_csv(args.imprint_cache).sort_values("query_idx")
    q_imprint = cache["imprint"].fillna("").astype(str).tolist()

    fold_files = sorted(f for f in glob.glob("outputs/eval_predictions_*_5folds_*.csv")
                        if f.rstrip(".csv")[-1] in "0123")

    # ---- aggregate rescue/hurt over folds ----
    tot = dict(rescued=0, hurt=0, both_right=0, both_wrong=0, n=0)
    for f in fold_files:
        _, S, correct, _ = load_holdout(f)
        vp = np.argmax(S, 1)
        Sf = fuse_scores(S, q_imprint, class_imp, args.lam, args.topk)
        mp = np.argmax(Sf, 1)
        for i, c in enumerate(correct):
            vok, mok = vp[i] == c, mp[i] == c
            tot["n"] += 1
            if not vok and mok:
                tot["rescued"] += 1
            elif vok and not mok:
                tot["hurt"] += 1
            elif vok and mok:
                tot["both_right"] += 1
            else:
                tot["both_wrong"] += 1

    print("=" * 60)
    print("E3 — imprint rescue analysis (4 folds pooled)")
    print("=" * 60)
    print(f"  queries total              : {tot['n']}")
    print(f"  RESCUED (visual wrong->MIRA right): {tot['rescued']}  "
          f"({100*tot['rescued']/tot['n']:.1f}%)")
    print(f"  hurt    (visual right->MIRA wrong): {tot['hurt']}  "
          f"({100*tot['hurt']/tot['n']:.1f}%)")
    print(f"  net rescued            : {tot['rescued']-tot['hurt']}")
    print(f"  both right / both wrong: {tot['both_right']} / {tot['both_wrong']}")
    print("=" * 60)

    # ---- qualitative examples from one fold ----
    _, S, correct, paths = load_holdout(args.example_fold)
    vp = np.argmax(S, 1)
    Sf = fuse_scores(S, q_imprint, class_imp, args.lam, args.topk)
    mp = np.argmax(Sf, 1)
    ex = []
    for i, c in enumerate(correct):
        if vp[i] != c and mp[i] == c:   # rescued
            ex.append({
                "query_img": os.path.basename(str(paths[i][0])) if paths[i] else "",
                "read_imprint": q_imprint[i],
                "visual_top1_class": classes[vp[i]],
                "visual_top1_imprint": class_imp[vp[i]],
                "true_class": classes[c],
                "true_imprint": class_imp[c],
            })
        if len(ex) >= args.n_examples:
            break
    exdf = pd.DataFrame(ex)
    os.makedirs("fusion_out", exist_ok=True)
    exdf.to_csv("fusion_out/e3_rescued_examples.csv", index=False)
    print(f"\nQualitative RESCUED examples (fold _3), saved to fusion_out/e3_rescued_examples.csv:")
    for _, r in exdf.iterrows():
        print(f"  read='{r['read_imprint']}' | visual✗ '{r['visual_top1_imprint']}' "
              f"({r['visual_top1_class'][:18]}) -> true '{r['true_imprint']}' "
              f"({r['true_class'][:18]})")


if __name__ == "__main__":
    main()
