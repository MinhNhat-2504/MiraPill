# -*- coding: utf-8 -*-
"""
E4 — Open-set recognition (OSR) + zero-shot capability (new evaluation the
original ePillID benchmark never ran; carries the generalization argument since
CURE is deferred).

Protocol: split the pill types appearing in the holdout into ENROLLED (known, in
gallery) and NOT-ENROLLED (unknown). A query whose true type is unknown must be
REJECTED. Confidence c(q) = max over enrolled classes of the score; we measure
how well c(q) separates known vs unknown (AUROC, FPR@95%TPR), comparing
visual-only vs MIRA-Pill. Hypothesis: imprint matching sharpens the separation
(an unknown pill's imprint matches no enrolled class well).

Averaged over R random enroll/unknown splits (50% unknown).
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fusion_rerank import (load_holdout, build_class_imprints, imprint_match)

BASE = ("D:/ePillID_data/ePillID_data/folds/"
        "pilltypeid_nih_sidelbls0.01_metric_5folds/base")


def full_fusion(S, q_imprint, class_imp, lam, topk):
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(S)):
        tq = q_imprint[i]
        if not tq:
            continue
        for c in np.argsort(-S[i])[:topk]:
            Sf[i, c] += lam * imprint_match(tq, class_imp[c])
    return Sv, Sf


def fpr_at_tpr(labels, scores, tpr_target=0.95):
    # labels: 1=known, 0=unknown; scores: higher=more "known"
    pos = np.sort(scores[labels == 1])
    thr = pos[int((1 - tpr_target) * len(pos))]  # threshold giving ~95% TPR
    neg = scores[labels == 0]
    return float(np.mean(neg >= thr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold",
                    default="outputs/eval_predictions_pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv")
    ap.add_argument("--label_encoder", default=f"{BASE}/label_encoder_local.pickle")
    ap.add_argument("--gt_csv", default="D:/pillbox_kaggle/Pillbox.csv")
    ap.add_argument("--all_imgs_csv",
                    default=f"{BASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    ap.add_argument("--imprint_cache", default="fusion_out/imprints_holdout.csv")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--unknown_frac", type=float, default=0.5)
    ap.add_argument("--rounds", type=int, default=5)
    args = ap.parse_args()

    classes, class_imp = build_class_imprints(args.label_encoder, args.gt_csv, args.all_imgs_csv)
    cache = pd.read_csv(args.imprint_cache).sort_values("query_idx")
    q_imprint = cache["imprint"].fillna("").astype(str).tolist()
    _, S, correct, _ = load_holdout(args.fold)
    Sv, Sf = full_fusion(S, q_imprint, class_imp, args.lam, args.topk)

    present = np.unique(correct)             # classes that have a query in holdout
    NEG = -1e9
    res = {"visual": [], "mira": []}
    cs = {"visual": [], "mira": []}          # closed-set acc on known queries
    fp = {"visual": [], "mira": []}
    for r in range(args.rounds):
        rng = np.random.RandomState(100 + r)
        perm = rng.permutation(present)
        n_unk = int(len(present) * args.unknown_frac)
        unknown = set(perm[:n_unk].tolist())
        enrolled = np.array(sorted(set(present.tolist()) - unknown))
        is_known = np.array([0 if c in unknown else 1 for c in correct])

        for name, M in [("visual", Sv), ("mira", Sf)]:
            mask = np.full(M.shape[1], NEG)
            mask[enrolled] = 0.0
            Mk = M + mask                    # only enrolled classes are candidates
            conf = Mk.max(1)
            pred = Mk.argmax(1)
            res[name].append(roc_auc_score(is_known, conf))
            fp[name].append(fpr_at_tpr(is_known, conf, 0.95))
            kn = is_known == 1
            cs[name].append(float(np.mean(pred[kn] == correct[kn])))

    print("=" * 60)
    print(f"E4 — Open-set recognition (fold _3, {args.rounds} splits, "
          f"{int(100*args.unknown_frac)}% unknown)")
    print("=" * 60)
    print(f"{'Method':12s} {'AUROC':>14s} {'FPR@95TPR':>14s} {'closed-set acc':>16s}")
    for name in ["visual", "mira"]:
        a = np.array(res[name]); f = np.array(fp[name]); c = np.array(cs[name])
        print(f"{name:12s} {100*a.mean():6.2f}±{100*a.std():4.2f}  "
              f"{100*f.mean():6.2f}±{100*f.std():4.2f}    "
              f"{100*c.mean():6.2f}±{100*c.std():4.2f}")
    da = 100 * (np.mean(res['mira']) - np.mean(res['visual']))
    print("-" * 60)
    print(f"Imprint improves open-set AUROC by +{da:.2f} pp")
    print("=" * 60)


if __name__ == "__main__":
    main()
