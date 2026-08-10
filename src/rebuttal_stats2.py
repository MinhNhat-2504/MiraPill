# -*- coding: utf-8 -*-
"""Second batch: FT-vs-OTS McNemar + OTS Top-5 (ePillID fold3), and clustered
paired bootstrap CI of dTop-1 for CURE and OGYEIv2. Reuses cached predictions."""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_rerank import (load_holdout, build_class_imprints, imprint_match,
                           topk_acc, mean_ap_at_rank)
from cure_fusion import build_class_imprints_cure, idf_match
from imprint_feasibility import normalize_imprint

RNG = np.random.RandomState(2024); NBOOT = 10000
EBASE = "D:/Data/ePillID_data/ePillID_data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base"
GT = "D:/Data/pillbox_kaggle/Pillbox.csv"
ALL = f"{EBASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv"
ELE = f"{EBASE}/label_encoder_local.pickle"

def fuse_plain(S, correct, q_imp, class_imp, lam, topk=20):
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(correct)):
        if not q_imp[i]: continue
        for c in np.argsort(-S[i])[:topk]:
            Sf[i, c] += lam * imprint_match(q_imp[i], class_imp[c])
    return Sf

def fuse_idf(S, q_imp, class_imp, idf, lam, topk=20):
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(q_imp)):
        if not q_imp[i]: continue
        for c in np.argsort(-S[i])[:topk]:
            Sf[i, c] += lam * idf_match(q_imp[i], class_imp[c], idf)
    return Sf

def top1_correct(M, correct):
    p = np.argmax(M, 1)
    return (p == np.asarray(correct)).astype(int)

def boot_ci_delta(vcorr, mcorr, clusters):
    keys = list(clusters.keys()); means = []
    for _ in range(NBOOT):
        pick = RNG.choice(len(keys), len(keys), replace=True)
        num = 0; den = 0
        for j in pick:
            idx = clusters[keys[j]]
            num += (mcorr[idx] - vcorr[idx]).sum(); den += len(idx)
        means.append(100.0 * num / den)
    return [float(x) for x in np.percentile(means, [2.5, 97.5])]

def clusters_by_class(correct):
    c = {}
    for i, k in enumerate(correct):
        c.setdefault(int(k), []).append(i)
    return {k: np.array(v) for k, v in c.items()}

out = {}

# ---- ePillID fold 3: FT vs OTS ----
pred3 = "outputs/eval_predictions_pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv"
_, S3, corr3, _ = load_holdout(pred3)
classes, class_imp = build_class_imprints(ELE, GT, ALL)
ft = pd.read_csv("fusion_out/imprints_holdout.csv").sort_values("query_idx")["imprint"].fillna("").astype(str).tolist()
ots = pd.read_csv("fusion_out/imprints_holdout_OTS.csv").sort_values("query_idx")["imprint"].fillna("").astype(str).tolist()
Sft = fuse_plain(S3, corr3, ft, class_imp, lam=0.2)
Sots = fuse_plain(S3, corr3, ots, class_imp, lam=0.2)
ftc = top1_correct(Sft, corr3); otc = top1_correct(Sots, corr3)
b = int(np.sum((ftc == 1) & (otc == 0)))   # FT right, OTS wrong
c = int(np.sum((ftc == 0) & (otc == 1)))   # FT wrong, OTS right
chi = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
p = float(chi2.sf(chi, 1)) if (b + c) > 0 else 1.0
out["ft_vs_ots"] = dict(ft_top1=float(100*ftc.mean()), ots_top1=float(100*otc.mean()),
                        ots_top5=float(100*topk_acc(Sots, corr3, 5)),
                        ft_top5=float(100*topk_acc(Sft, corr3, 5)),
                        b=b, c=c, chi2=float(chi), mcnemar_p=p)

# ---- CURE ----
def eval_cross(pred, le, db, cache_csv, lam):
    _, S, correct, paths = load_holdout(pred)
    classes, class_imp, idf = build_class_imprints_cure(le, db)
    cdf = pd.read_csv(cache_csv)
    col = "image" if "image" in cdf.columns else cdf.columns[0]
    path2imp = {r[col]: str(r.get("imprint", "") or "") for _, r in cdf.iterrows()}
    q_imp = [normalize_imprint(" ".join(path2imp.get(p, "") for p in imgs[:2])) for imgs in paths]
    Sf = fuse_idf(S, q_imp, class_imp, idf, lam=lam)
    vcorr = top1_correct(S, correct); mcorr = top1_correct(Sf, correct)
    ci = boot_ci_delta(vcorr, mcorr, clusters_by_class(correct))
    return dict(n=len(correct), v_top1=float(100*vcorr.mean()), m_top1=float(100*mcorr.mean()),
                delta=float(100*(mcorr.mean()-vcorr.mean())), boot_ci=ci)

out["cure"] = eval_cross("outputs/eval_predictions_cure_val.csv",
                         "D:/Data/CURE_crops/folds/cure/base/label_encoder_local.pickle",
                         "D:/Data/CURE_crops/cure_imprint_db.csv",
                         "fusion_out/cure_imprints_holdout.csv", lam=0.3)
out["ogyei"] = eval_cross("outputs/eval_predictions_ogyei_val.csv",
                          "D:/Data/ogyeiv2_crops/folds/ogyei/base/label_encoder_local.pickle",
                          "D:/Data/ogyeiv2_crops/ogyei_imprint_db.csv",
                          "fusion_out/ogyei_imprints.csv", lam=0.3)

print(json.dumps(out, indent=2))
json.dump(out, open("fusion_out/rebuttal_stats2.json", "w"), indent=2)
print("[saved] fusion_out/rebuttal_stats2.json")
