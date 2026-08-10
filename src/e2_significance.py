# -*- coding: utf-8 -*-
"""
E2 — statistical significance of MIRA-Pill vs visual-only, tested on the SAME
holdout (the methodologically correct way to compare two models on one test set):
  • McNemar's test on paired top-1 correctness (discordant pairs).
  • Paired t-test + Wilcoxon on per-query AP (1/rank), pooled over 4 folds (n large).
  • Bootstrap 95% CI for Δtop-1 and ΔMAP.
  • Fold-level paired t-test + Cohen's d (n=4) as a secondary check.
No retraining needed. (Multi-seed training tests a different question — training
variance — and can be added later.)
"""
import glob
import numpy as np
import pandas as pd
from scipy import stats

from fusion_rerank import load_holdout, build_class_imprints, imprint_match

BASE = ("D:/ePillID_data/ePillID_data/folds/"
        "pilltypeid_nih_sidelbls0.01_metric_5folds/base")


def fuse(S, q_imprint, class_imp, lam=0.2, topk=20):
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(S)):
        tq = q_imprint[i]
        if not tq:
            continue
        for c in np.argsort(-S[i])[:topk]:
            Sf[i, c] += lam * imprint_match(tq, class_imp[c])
    return Sf


def rank_of_true(M, correct):
    return np.array([1 + int(np.sum(M[i] > M[i, correct[i]])) for i in range(len(correct))])


def main():
    classes, class_imp = build_class_imprints(
        f"{BASE}/label_encoder_local.pickle", "D:/Data/pillbox_kaggle/Pillbox.csv",
        f"{BASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    cache = pd.read_csv("fusion_out/imprints_holdout.csv").sort_values("query_idx")
    q_imprint = cache["imprint"].fillna("").astype(str).tolist()

    fold_files = sorted(f for f in glob.glob("outputs/eval_predictions_*_5folds_*.csv")
                        if f.rstrip(".csv")[-1] in "0123")

    v_ok, m_ok, v_ap, m_ap = [], [], [], []          # pooled per-query
    fold_v1, fold_m1 = [], []                          # fold-level top1
    for f in fold_files:
        _, S, correct, _ = load_holdout(f)
        Sf = fuse(S, q_imprint, class_imp)
        vr, mr = rank_of_true(S, correct), rank_of_true(Sf, correct)
        v_ok += list((vr == 1).astype(int)); m_ok += list((mr == 1).astype(int))
        v_ap += list(1.0 / vr); m_ap += list(1.0 / mr)
        fold_v1.append(np.mean(vr == 1)); fold_m1.append(np.mean(mr == 1))

    v_ok, m_ok = np.array(v_ok), np.array(m_ok)
    v_ap, m_ap = np.array(v_ap), np.array(m_ap)
    n = len(v_ok)

    # --- McNemar on top-1 correctness ---
    b = int(np.sum((v_ok == 0) & (m_ok == 1)))   # visual wrong, MIRA right
    c = int(np.sum((v_ok == 1) & (m_ok == 0)))   # visual right, MIRA wrong
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_mcnemar = stats.chi2.sf(chi2, df=1)

    # --- paired tests on per-query AP ---
    t_stat, p_t = stats.ttest_rel(m_ap, v_ap)
    try:
        w_stat, p_w = stats.wilcoxon(m_ap, v_ap)
    except ValueError:
        p_w = float("nan")
    d_ap = (m_ap - v_ap).mean() / (m_ap - v_ap).std(ddof=1)   # Cohen's d (paired)

    # --- bootstrap 95% CI on Δtop1 and ΔMAP ---
    rng = np.random.RandomState(0)
    B = 10000
    d_top1, d_map = [], []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        d_top1.append(m_ok[idx].mean() - v_ok[idx].mean())
        d_map.append(m_ap[idx].mean() - v_ap[idx].mean())
    ci_top1 = np.percentile(d_top1, [2.5, 97.5]) * 100
    ci_map = np.percentile(d_map, [2.5, 97.5]) * 100

    # --- fold-level (n=4) ---
    fv, fm = np.array(fold_v1), np.array(fold_m1)
    t4, p4 = stats.ttest_rel(fm, fv)
    d4 = (fm - fv).mean() / (fm - fv).std(ddof=1)

    print("=" * 64)
    print(f"E2 — Statistical significance (pooled holdout, n={n} queries, 4 folds)")
    print("=" * 64)
    print(f"top-1: visual {100*v_ok.mean():.2f}%  MIRA {100*m_ok.mean():.2f}%  "
          f"(Δ {100*(m_ok.mean()-v_ok.mean()):+.2f}pp)")
    print(f"MAP  : visual {100*v_ap.mean():.2f}%  MIRA {100*m_ap.mean():.2f}%  "
          f"(Δ {100*(m_ap.mean()-v_ap.mean()):+.2f}pp)")
    print("-" * 64)
    print(f"McNemar (top-1): discordant b={b} (vis✗/MIRA✓), c={c} (vis✓/MIRA✗)")
    print(f"   χ²={chi2:.1f},  p={p_mcnemar:.3e}")
    print(f"Paired t-test on per-query AP: t={t_stat:.1f}, p={p_t:.3e}")
    print(f"Wilcoxon on per-query AP:      p={p_w:.3e}")
    print(f"Cohen's d (per-query AP):      d={d_ap:.3f}")
    print(f"Bootstrap 95% CI  Δtop-1: [{ci_top1[0]:.2f}, {ci_top1[1]:.2f}] pp")
    print(f"Bootstrap 95% CI  ΔMAP  : [{ci_map[0]:.2f}, {ci_map[1]:.2f}] pp")
    print("-" * 64)
    print(f"Fold-level (n=4) paired t-test: t={t4:.2f}, p={p4:.4f}, Cohen's d={d4:.2f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
