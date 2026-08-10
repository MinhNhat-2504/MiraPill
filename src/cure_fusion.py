# -*- coding: utf-8 -*-
"""
CURE cross-dataset fusion + E7/E8/E9. Mirrors the ePillID fusion but uses the
OCR-consensus imprint DB (cure_imprint_db.csv, keyed by class id) instead of
Pillbox. S_visual comes from the CURE eval predictions; imprints of test queries
are read with the ePillID-fine-tuned Qwen reader.

Phase 1 (GPU): --cache_imprints  -> read each holdout query's side crops.
Phase 2 (CPU): fuse -> E7 (visual vs MIRA), E8 (rescued), E9 (branch ablation).
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

from fusion_rerank import (load_holdout, imprint_match, topk_acc, mean_ap_at_rank,
                           parse_paths)
from imprint_feasibility import normalize_imprint, QwenReader

try:
    import sklearn.preprocessing._label as _lbl
    sys.modules.setdefault("sklearn.preprocessing.label", _lbl)
except Exception:
    pass

BASE = r"D:\CURE_crops\folds\cure\base"


def build_class_imprints_cure(label_encoder_path, db_csv):
    import math
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)
    classes = [str(c) for c in le.classes_]
    db = pd.read_csv(db_csv, dtype=str)
    m = {str(r["pilltype_id"]): normalize_imprint(r.get("imprint", "") or "")
         for _, r in db.iterrows()}
    class_imp = [m.get(c, "") for c in classes]
    n = sum(1 for x in class_imp if x)
    print(f"[db] CURE classes={len(classes)} with imprint={n} ({100*n/len(classes):.1f}%)")
    # IDF over the class DB: down-weight non-discriminative tokens (e.g. a 'T'
    # the reader hallucinates on blank faces of many classes)
    N = len(class_imp); dfreq = {}
    for x in class_imp:
        for w in set(x.split()):
            dfreq[w] = dfreq.get(w, 0) + 1
    idf = {w: math.log(N / c) for w, c in dfreq.items()}
    idf["__default__"] = math.log(N)
    return classes, class_imp, idf


def idf_match(tq, tc, idf):
    """IDF-weighted asymmetric imprint match (rare tokens count more)."""
    qt = normalize_imprint(tq).split()
    ct = set(normalize_imprint(tc).split())
    if not qt or not ct:
        return 0.0
    d = idf["__default__"]
    num = sum(idf.get(w, d) for w in qt if w in ct)
    den = sum(idf.get(w, d) for w in qt) or 1.0
    return num / den


def cache(paths, args):
    """Read each UNIQUE test image once (20k query-pairs reuse ~1.8k images)."""
    from PIL import Image
    uniq = sorted({p for imgs in paths for p in imgs})
    print(f"[cache] {len(uniq)} unique images (from {len(paths)} query pairs)")
    reader = QwenReader(args.qwen_model, args.device, adapter=args.adapter)
    rows = []
    for i, p in enumerate(uniq):
        try:
            imp = normalize_imprint(reader.read(Image.open(p)))
        except Exception as e:
            imp = ""; print(f"  [warn] {p}: {e}")
        rows.append({"image": p, "imprint": imp})
        if (i + 1) % 100 == 0:
            print(f"  [imprint] {i+1}/{len(uniq)}", flush=True)
    os.makedirs(os.path.dirname(args.imprint_cache) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.imprint_cache, index=False)
    print(f"[cache] -> {args.imprint_cache}")


def fuse(S, q_imprint, class_imp, lam, topk, idf=None):
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(S)):
        tq = q_imprint[i]
        if not tq:
            continue
        for c in np.argsort(-S[i])[:topk]:
            mt = idf_match(tq, class_imp[c], idf) if idf else imprint_match(tq, class_imp[c])
            Sf[i, c] += lam * mt
    return Sv, Sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", default="outputs/eval_predictions_cure_val.csv")
    ap.add_argument("--label_encoder", default=f"{BASE}/label_encoder_local.pickle")
    ap.add_argument("--db_csv", default=r"D:\CURE_crops\cure_imprint_db.csv")
    ap.add_argument("--imprint_cache", default="fusion_out/cure_imprints_holdout.csv")
    ap.add_argument("--adapter", default="qwen_imprint_lora")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache_imprints", action="store_true")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    h, S, correct, paths = load_holdout(args.pred_csv)
    print(f"[load] CURE holdout queries={len(correct)} S={S.shape}")
    if args.cache_imprints:
        cache(paths, args)
        return

    classes, class_imp, idf = build_class_imprints_cure(args.label_encoder, args.db_csv)
    cdf = pd.read_csv(args.imprint_cache)
    path2imp = {r["image"]: str(r.get("imprint", "") or "") for _, r in cdf.iterrows()}
    # pool the two side-images' imprints per query pair
    q_imprint = [normalize_imprint(" ".join(path2imp.get(p, "") for p in imgs[:2]))
                 for imgs in paths]
    Sv, Sf = fuse(S, q_imprint, class_imp, args.lam, args.topk, idf=idf)

    # imprint-only (rank all classes by match) for E9
    tok = [set(x.split()) if x else set() for x in class_imp]
    Mi = np.zeros_like(S)
    for i, t in enumerate(q_imprint):
        qt = set(normalize_imprint(t).split())
        if qt:
            for c in range(S.shape[1]):
                if tok[c]:
                    inter = len(qt & tok[c])
                    if inter:
                        Mi[i, c] = inter / len(qt)

    vp, mp = np.argmax(S, 1), np.argmax(Sf, 1)
    rescued = int(np.sum((vp != correct) & (mp == correct)))
    hurt = int(np.sum((vp == correct) & (mp != correct)))

    print("\n" + "=" * 60)
    print(f"CURE cross-dataset (holdout, {len(correct)} queries, λ={args.lam})")
    print("=" * 60)
    print("E7/E9 branch table:")
    print(f"  {'Visual-only':16s} top1={topk_acc(S,correct,1):.4f} top5={topk_acc(S,correct,5):.4f} MAP={mean_ap_at_rank(S,correct):.4f}")
    print(f"  {'Imprint-only':16s} top1={topk_acc(Mi,correct,1):.4f} top5={topk_acc(Mi,correct,5):.4f} MAP={mean_ap_at_rank(Mi,correct):.4f}")
    print(f"  {'MIRA-Pill':16s} top1={topk_acc(Sf,correct,1):.4f} top5={topk_acc(Sf,correct,5):.4f} MAP={mean_ap_at_rank(Sf,correct):.4f}")
    print(f"E8: rescued (visual wrong->MIRA right)={rescued}  hurt={hurt}  net={rescued-hurt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
