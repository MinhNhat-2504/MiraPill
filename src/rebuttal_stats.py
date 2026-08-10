# -*- coding: utf-8 -*-
"""
Compute all reviewer-requested statistics for the CMC-87689 revision from the
CACHED predictions + imprint reads (no retraining). Reuses fusion_rerank logic.
Outputs a single JSON with every number needed to fill the manuscript placeholders.
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_rerank import (load_holdout, build_class_imprints, imprint_match,
                           topk_acc, mean_ap_at_rank)

BASE = ("D:/Data/ePillID_data/ePillID_data/folds/"
        "pilltypeid_nih_sidelbls0.01_metric_5folds/base")
GT = "D:/Data/pillbox_kaggle/Pillbox.csv"
ALL = f"{BASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv"
LE = f"{BASE}/label_encoder_local.pickle"
LAM, TOPK = 0.2, 20
RNG = np.random.RandomState(12345)
NBOOT = 10000

def fuse(S, correct, q_imp, class_imp, lam=LAM, topk=TOPK):
    """Return per-query dict arrays: visual/MIRA top1,top5 correctness, AP, gated."""
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    gated = np.zeros(len(correct), dtype=bool)
    for i in range(len(correct)):
        tq = q_imp[i] if i < len(q_imp) else ""
        if not tq:
            continue
        gated[i] = True
        for c in np.argsort(-S[i])[:topk]:
            Sf[i, c] += lam * imprint_match(tq, class_imp[c])
    def per_q(M):
        order = np.argsort(-M, axis=1)
        t1 = np.array([correct[i] == order[i, 0] for i in range(len(correct))])
        t5 = np.array([correct[i] in order[i, :5] for i in range(len(correct))])
        ap = np.array([1.0 / (1 + int(np.sum(M[i] > M[i, correct[i]]))) for i in range(len(correct))])
        return t1, t5, ap
    v1, v5, vap = per_q(S)
    m1, m5, map_ = per_q(Sf)
    return dict(v1=v1, v5=v5, vap=vap, m1=m1, m5=m5, map=map_, gated=gated)

def clustered_boot(delta_by_cluster_sizes, clusters, values, nboot=NBOOT):
    """Bootstrap over clusters (list of arrays of per-unit deltas). Resample
    clusters with replacement, weight by size, return 95% CI of mean delta (pp)."""
    keys = list(clusters.keys())
    means = []
    for _ in range(nboot):
        pick = RNG.choice(len(keys), len(keys), replace=True)
        num = 0.0; den = 0
        for j in pick:
            arr = clusters[keys[j]]
            num += arr.sum(); den += len(arr)
        means.append(100.0 * num / den)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)

def main():
    out = {}
    # ---- load 4 ePillID folds (shared _4 holdout) ----
    files = sorted(f for f in glob.glob("outputs/eval_predictions_*_5folds_*.csv")
                   if f.rstrip(".csv")[-1] in "0123")
    classes, class_imp = build_class_imprints(LE, GT, ALL)
    cache = pd.read_csv("fusion_out/imprints_holdout.csv").sort_values("query_idx")
    q_imp = cache["imprint"].fillna("").astype(str).tolist()

    per_fold = []
    correct_ref = None
    for f in files:
        _, S, correct, paths = load_holdout(f)
        if correct_ref is None:
            correct_ref = correct
        r = fuse(S, correct, q_imp, class_imp)
        r["correct"] = correct
        per_fold.append(r)
    N = len(correct_ref)
    out["holdout_N"] = int(N)
    out["pooled_N"] = int(N * len(files))
    out["n_folds"] = len(files)

    # pooled per-query deltas with cluster = true class, and (fold,class)
    d1 = np.concatenate([pf["m1"].astype(int) - pf["v1"].astype(int) for pf in per_fold])
    d5 = np.concatenate([pf["m5"].astype(int) - pf["v5"].astype(int) for pf in per_fold])
    dap = np.concatenate([pf["map"] - pf["vap"] for pf in per_fold])
    cls = np.concatenate([pf["correct"] for pf in per_fold])
    foldid = np.concatenate([[k] * N for k in range(len(files))])

    def cluster_by(keyarr, dvals):
        c = {}
        for k, dv in zip(keyarr, dvals):
            c.setdefault(k, []).append(dv)
        return {k: np.array(v) for k, v in c.items()}

    out["boot_class_top1"] = clustered_boot(None, cluster_by(cls, d1), None)
    out["boot_class_top5"] = clustered_boot(None, cluster_by(cls, d5), None)
    out["boot_class_map"] = clustered_boot(None, cluster_by(cls, dap), None)
    fc_keys = np.array([f"{f}_{c}" for f, c in zip(foldid, cls)])
    out["boot_foldclass_top1"] = clustered_boot(None, cluster_by(fc_keys, d1), None)

    # point deltas (pp)
    out["delta_top1_pp"] = float(100 * d1.mean())
    out["delta_top5_pp"] = float(100 * d5.mean())
    out["delta_map_pp"] = float(100 * dap.mean())

    # ---- gate activity + rescue/hurt (pooled) ----
    gated = np.concatenate([pf["gated"] for pf in per_fold])
    v1all = np.concatenate([pf["v1"] for pf in per_fold])
    m1all = np.concatenate([pf["m1"] for pf in per_fold])
    rescued = int(np.sum((~v1all) & m1all))
    hurt = int(np.sum(v1all & (~m1all)))
    out["gate_active"] = int(gated.sum()); out["gate_inactive"] = int((~gated).sum())
    out["gate_active_share"] = float(100 * gated.mean())
    out["gate_active_top1correct"] = int(np.sum(gated & m1all))
    out["gate_active_top1correct_share"] = float(100 * np.sum(gated & m1all) / max(gated.sum(),1))
    out["rescued"] = rescued; out["hurt"] = hurt
    out["rescued_share"] = float(100*rescued/len(v1all)); out["hurt_share"] = float(100*hurt/len(v1all))

    # ---- stratified by imprint availability (pooled) ----
    # imprint present in metadata = class has non-empty DB imprint
    dbimp = np.array([1 if class_imp[c] else 0 for c in cls])
    readnonempty = np.concatenate([[1 if (q_imp[i] if i<len(q_imp) else "") else 0
                                    for i in range(N)] for _ in files])
    strata = {}
    for name, mask in [("present_readnonempty", (dbimp==1)&(readnonempty==1)),
                       ("present_readempty", (dbimp==1)&(readnonempty==0)),
                       ("absent_metadata", (dbimp==0))]:
        if mask.sum()==0:
            strata[name]=dict(n=0,vtop1=0,dtop1=0); continue
        strata[name]=dict(n=int(mask.sum()),
                          vtop1=float(100*v1all[mask].mean()),
                          dtop1=float(100*(m1all[mask].astype(int)-v1all[mask].astype(int)).mean()))
    out["strata"] = strata

    # ---- Table 2 data-flow (per fold: train/val/test types, images) ----
    allc = pd.read_csv(ALL)
    folds_types = {}
    for k in range(5):
        fk = pd.read_csv(f"{BASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_{k}.csv")
        folds_types[k] = set(fk["label"].unique())
    test_types = folds_types[4]
    df2 = []
    for k in range(4):  # val fold k, test=4
        val_types = folds_types[k]
        train_types = set(allc["label"].unique()) - val_types - test_types
        ref_imgs = int(allc[(allc["label"].isin(train_types)) & (allc["is_ref"])].shape[0])
        test_cons = int(allc[(allc["label"].isin(test_types)) & (~allc["is_ref"])].shape[0])
        df2.append(dict(fold=k+1, train_types=len(train_types), test_types=len(test_types),
                        ref_gallery=ref_imgs, consumer_test=test_cons,
                        bothsides_queries=int(N), final_queries=int(N)))
    out["dataflow"] = df2

    print(json.dumps(out, indent=2))
    with open("fusion_out/rebuttal_stats.json", "w") as fp:
        json.dump(out, fp, indent=2)
    print("\n[saved] fusion_out/rebuttal_stats.json")

if __name__ == "__main__":
    main()
