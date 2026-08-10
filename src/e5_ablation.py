# -*- coding: utf-8 -*-
"""
E5 — ablations.
 (a) Branch ablation: visual-only / imprint-only / MIRA-Pill (fusion).
 (b) Sensitivity to fusion weight lambda and re-rank window top-K.
Reader ablation (off-the-shelf vs fine-tuned reader inside fusion) is done
separately (needs a GPU re-cache); see e5_reader_ablation in fusion_rerank usage.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from fusion_rerank import (load_holdout, build_class_imprints, imprint_match,
                           normalize_imprint, topk_acc, mean_ap_at_rank)

BASE = ("D:/ePillID_data/ePillID_data/folds/"
        "pilltypeid_nih_sidelbls0.01_metric_5folds/base")


def fuse(S, q_imprint, class_imp, lam, topk):
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(S)):
        tq = q_imprint[i]
        if not tq:
            continue
        for c in np.argsort(-S[i])[:topk]:
            Sf[i, c] += lam * imprint_match(tq, class_imp[c])
    return Sf


def imprint_only_scores(q_imprint, class_tok, S_shape):
    """Rank ALL classes by imprint match alone (no visual)."""
    n_q, n_c = S_shape
    M = np.zeros((n_q, n_c), dtype=np.float32)
    for i, tq in enumerate(q_imprint):
        qt = set(normalize_imprint(tq).split())
        if not qt:
            continue
        nq = len(qt)
        for c in range(n_c):
            ct = class_tok[c]
            if ct:
                inter = len(qt & ct)
                if inter:
                    M[i, c] = inter / nq
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label_encoder", default=f"{BASE}/label_encoder_local.pickle")
    ap.add_argument("--gt_csv", default="D:/pillbox_kaggle/Pillbox.csv")
    ap.add_argument("--all_imgs_csv",
                    default=f"{BASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    ap.add_argument("--imprint_cache", default="fusion_out/imprints_holdout.csv")
    ap.add_argument("--fold",
                    default="outputs/eval_predictions_pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    classes, class_imp = build_class_imprints(args.label_encoder, args.gt_csv,
                                              args.all_imgs_csv)
    class_tok = [set(x.split()) if x else set() for x in class_imp]
    cache = pd.read_csv(args.imprint_cache).sort_values("query_idx")
    q_imprint = cache["imprint"].fillna("").astype(str).tolist()

    _, S, correct, _ = load_holdout(args.fold)

    print("=" * 60)
    print(f"E5(a) Branch ablation (fold _3, both-sides, λ={args.lam})")
    print("=" * 60)
    print(f"{'Branch':16s} {'top-1':>8s} {'top-5':>8s} {'MAP':>8s}")
    # visual-only
    print(f"{'Visual-only':16s} {topk_acc(S,correct,1):8.4f} {topk_acc(S,correct,5):8.4f} {mean_ap_at_rank(S,correct):8.4f}")
    # imprint-only
    Mi = imprint_only_scores(q_imprint, class_tok, S.shape)
    print(f"{'Imprint-only':16s} {topk_acc(Mi,correct,1):8.4f} {topk_acc(Mi,correct,5):8.4f} {mean_ap_at_rank(Mi,correct):8.4f}")
    # fusion
    Sf = fuse(S, q_imprint, class_imp, args.lam, args.topk)
    print(f"{'MIRA-Pill':16s} {topk_acc(Sf,correct,1):8.4f} {topk_acc(Sf,correct,5):8.4f} {mean_ap_at_rank(Sf,correct):8.4f}")

    print("\n" + "=" * 60)
    print("E5(b) Sensitivity to lambda (fold _3, topK=20)")
    print("=" * 60)
    for lam in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]:
        Sf = fuse(S, q_imprint, class_imp, lam, args.topk)
        print(f"  λ={lam:.1f}: top1={topk_acc(Sf,correct,1):.4f}  MAP={mean_ap_at_rank(Sf,correct):.4f}")
    print("\nE5(b) Sensitivity to top-K (fold _3, λ=0.2)")
    for tk in [5, 10, 20, 50]:
        Sf = fuse(S, q_imprint, class_imp, 0.2, tk)
        print(f"  K={tk:2d}: top1={topk_acc(Sf,correct,1):.4f}  MAP={mean_ap_at_rank(Sf,correct):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
