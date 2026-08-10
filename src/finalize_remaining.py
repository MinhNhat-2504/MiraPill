# -*- coding: utf-8 -*-
"""Aggregate the queued experiments (multi-seed, ConvNeXt backbone) into
fusion_out/remaining_results.json. Reads whatever prediction dirs exist; missing
experiments are simply omitted so the filler can run incrementally."""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_rerank import (load_holdout, build_class_imprints, imprint_match,
                           topk_acc, mean_ap_at_rank)

EBASE = "D:/Data/ePillID_data/ePillID_data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base"
GT = "D:/Data/pillbox_kaggle/Pillbox.csv"
ALL = f"{EBASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv"
LE = f"{EBASE}/label_encoder_local.pickle"
LAM, TOPK = 0.2, 20

_classes = _class_imp = None
def cimp():
    global _classes, _class_imp
    if _class_imp is None:
        _classes, _class_imp = build_class_imprints(LE, GT, ALL)
    return _class_imp

def fuse_metrics(pred_csv, imprint_cache="fusion_out/imprints_holdout.csv"):
    _, S, correct, _ = load_holdout(pred_csv)
    q = pd.read_csv(imprint_cache).sort_values("query_idx")["imprint"].fillna("").astype(str).tolist()
    ci = cimp()
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(correct)):
        if i < len(q) and q[i]:
            for c in np.argsort(-S[i])[:TOPK]:
                Sf[i, c] += LAM * imprint_match(q[i], ci[c])
    return dict(v_top1=100*topk_acc(S, correct, 1), v_map=100*mean_ap_at_rank(S, correct),
                m_top1=100*topk_acc(Sf, correct, 1), m_map=100*mean_ap_at_rank(Sf, correct))

out = {}

# ---- multi-seed (fold 3): canonical + seed dirs ----
seed_preds = []
canon = "_canonical_backup/eval_predictions_pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv"
if os.path.exists(canon):
    seed_preds.append(("canonical", canon))
for sd in ("202", "303"):
    g = sorted(glob.glob(f"visual_results_seed{sd}/*/eval_predictions_*_5folds_3.csv"))
    if g:
        seed_preds.append((f"seed{sd}", g[-1]))
if len(seed_preds) >= 2:
    vt, mt = [], []
    for name, p in seed_preds:
        try:
            m = fuse_metrics(p); vt.append(m["v_top1"]); mt.append(m["m_top1"])
            print(f"  multiseed {name}: visual={m['v_top1']:.2f} MIRA={m['m_top1']:.2f}")
        except Exception as e:
            print(f"  multiseed {name} FAIL: {e}")
    if len(mt) >= 2:
        out["multiseed"] = dict(n_seeds=len(mt), mira_mean=float(np.mean(mt)),
                                mira_std=float(np.std(mt, ddof=1)),
                                visual_mean=float(np.mean(vt)), visual_std=float(np.std(vt, ddof=1)),
                                mira_vals=[round(x,2) for x in mt])

# ---- ConvNeXt backbone (fold 3) ----
g = sorted(glob.glob("visual_results_convnext/*/eval_predictions_*_5folds_3.csv"))
if g:
    try:
        m = fuse_metrics(g[-1])
        out["convnext"] = dict(v_top1=round(m["v_top1"],2), v_map=round(m["v_map"],2),
                               m_top1=round(m["m_top1"],2), m_map=round(m["m_map"],2),
                               gain_top1=round(m["m_top1"]-m["v_top1"],2),
                               gain_map=round(m["m_map"]-m["v_map"],2))
        print(f"  convnext: {out['convnext']}")
    except Exception as e:
        print(f"  convnext FAIL: {e}")

# ---- soft-gate + OCR results (produced by their own scripts) ----
for key, path in [("softgate", "fusion_out/softgate_result.json"),
                  ("ocr", "fusion_out/ocr_result.json")]:
    if os.path.exists(path):
        out[key] = json.load(open(path))

json.dump(out, open("fusion_out/remaining_results.json", "w"), indent=2)
print("\n[saved] fusion_out/remaining_results.json")
print(json.dumps(out, indent=2))
