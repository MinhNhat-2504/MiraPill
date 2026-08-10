# -*- coding: utf-8 -*-
"""Identify and categorize the 6 hurt cases (visual-right -> MIRA-wrong) pooled
over the 4 ePillID folds, for the gate-activity discussion."""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_rerank import (load_holdout, build_class_imprints, imprint_match)
from imprint_feasibility import normalize_imprint

EBASE = "D:/Data/ePillID_data/ePillID_data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base"
GT = "D:/Data/pillbox_kaggle/Pillbox.csv"
ALL = f"{EBASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv"
LE = f"{EBASE}/label_encoder_local.pickle"
LAM, TOPK = 0.2, 20

classes, class_imp = build_class_imprints(LE, GT, ALL)
q_imp = pd.read_csv("fusion_out/imprints_holdout.csv").sort_values("query_idx")["imprint"].fillna("").astype(str).tolist()
files = sorted(f for f in glob.glob("outputs/eval_predictions_*_5folds_*.csv") if f.rstrip(".csv")[-1] in "0123")

hall, shared = 0, 0
rows = []
for f in files:
    _, S, correct, _ = load_holdout(f)
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(correct)):
        if not q_imp[i]:
            continue
        for c in np.argsort(-S[i])[:TOPK]:
            Sf[i, c] += LAM * imprint_match(q_imp[i], class_imp[c])
    vp, mp = np.argmax(Sv, 1), np.argmax(Sf, 1)
    for i in range(len(correct)):
        if vp[i] == correct[i] and mp[i] != correct[i]:   # hurt
            tq = q_imp[i]
            true_imp = class_imp[correct[i]]
            promoted_imp = class_imp[mp[i]]
            m_true = imprint_match(tq, true_imp)
            m_prom = imprint_match(tq, promoted_imp)
            # hallucinated/misread: read matches promoted (wrong) better than true
            cat = "hallucinated_or_misread" if m_prom > m_true else "correct_read_shared"
            if cat == "hallucinated_or_misread": hall += 1
            else: shared += 1
            rows.append(dict(fold=os.path.basename(f)[-5], read=tq, true=true_imp,
                             promoted=promoted_imp, m_true=round(m_true,2),
                             m_prom=round(m_prom,2), cat=cat))
print(f"TOTAL hurt={len(rows)}  hallucinated_or_misread={hall}  correct_read_shared={shared}")
for r in rows:
    print(r)
