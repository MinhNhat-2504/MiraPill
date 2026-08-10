# -*- coding: utf-8 -*-
"""
E1 aggregation: fuse visual + imprint for all 4 CV folds (val=_0.._3, test=_4),
report visual-only vs MIRA-Pill as mean +/- std (the paper's 4-fold protocol).

Holdout test set (_4) is identical across folds, so the imprint cache (read once
on _4 queries) is reused; we assert the query order matches across folds.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

from fusion_rerank import (load_holdout, build_class_imprints, imprint_match,
                           topk_acc, mean_ap_at_rank)

BASE = ("D:/ePillID_data/ePillID_data/folds/"
        "pilltypeid_nih_sidelbls0.01_metric_5folds/base")


def fold_metrics(S, correct, q_imprint, class_imp, lam, topk):
    # visual-only
    v = (topk_acc(S, correct, 1), topk_acc(S, correct, 5), mean_ap_at_rank(S, correct))
    # fusion
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(correct)):
        tq = q_imprint[i]
        if not tq:
            continue
        for c in np.argsort(-S[i])[:topk]:
            Sf[i, c] += lam * imprint_match(tq, class_imp[c])
    m = (topk_acc(Sf, correct, 1), topk_acc(Sf, correct, 5), mean_ap_at_rank(Sf, correct))
    return v, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_glob", default="outputs/eval_predictions_*_5folds_*.csv")
    ap.add_argument("--label_encoder", default=f"{BASE}/label_encoder_local.pickle")
    ap.add_argument("--gt_csv", default="D:/pillbox_kaggle/Pillbox.csv")
    ap.add_argument("--all_imgs_csv",
                    default=f"{BASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    ap.add_argument("--imprint_cache", default="fusion_out/imprints_holdout.csv")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    fold_files = sorted(f for f in glob.glob(args.pred_glob)
                        if f.rstrip(".csv")[-1] in "0123")
    print("[e1] fold prediction files:")
    for f in fold_files:
        print("   ", os.path.basename(f))

    classes, class_imp = build_class_imprints(args.label_encoder, args.gt_csv,
                                              args.all_imgs_csv)
    cache = pd.read_csv(args.imprint_cache).sort_values("query_idx")
    q_imprint = cache["imprint"].fillna("").astype(str).tolist()

    rows = []
    ref_correct = None
    for f in fold_files:
        _, S, correct, _ = load_holdout(f)
        # all folds share the same _4 holdout -> identical query order/labels
        if ref_correct is None:
            ref_correct = correct
        else:
            assert np.array_equal(correct, ref_correct), \
                f"holdout order differs in {f} -> imprint cache misaligned!"
        assert len(correct) == len(q_imprint), \
            f"cache len {len(q_imprint)} != queries {len(correct)}"
        v, m = fold_metrics(S, correct, q_imprint, class_imp, args.lam, args.topk)
        rows.append({"fold": os.path.basename(f).rstrip(".csv")[-1],
                     "v_top1": v[0], "v_top5": v[1], "v_map": v[2],
                     "m_top1": m[0], "m_top5": m[1], "m_map": m[2]})
        print(f"  fold _{rows[-1]['fold']}: visual top1={v[0]:.4f} MAP={v[2]:.4f} | "
              f"MIRA top1={m[0]:.4f} MAP={m[2]:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs("fusion_out", exist_ok=True)
    df.to_csv("fusion_out/e1_per_fold.csv", index=False)

    def ms(col):
        return 100 * df[col].mean(), 100 * df[col].std()

    print("\n" + "=" * 60)
    print(f"E1 — 4-fold CV (both-sides, test=_4), lambda={args.lam}")
    print("=" * 60)
    print(f"{'Model':16s} {'top-1':>14s} {'top-5':>14s} {'MAP':>14s}")
    for name, p in [("Visual-only", "v"), ("MIRA-Pill", "m")]:
        t1 = ms(f"{p}_top1"); t5 = ms(f"{p}_top5"); mp = ms(f"{p}_map")
        print(f"{name:16s} {t1[0]:6.2f}±{t1[1]:4.2f}  {t5[0]:6.2f}±{t5[1]:4.2f}  "
              f"{mp[0]:6.2f}±{mp[1]:4.2f}")
    d1 = 100 * (df["m_top1"].mean() - df["v_top1"].mean())
    dm = 100 * (df["m_map"].mean() - df["v_map"].mean())
    print("-" * 60)
    print(f"Improvement: top-1 +{d1:.2f}pp   MAP +{dm:.2f}pp")
    print("=" * 60)


if __name__ == "__main__":
    main()
