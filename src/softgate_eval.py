# -*- coding: utf-8 -*-
"""Table 24 (Appendix B): hard vs soft confidence gate on ePillID fold 3.
Soft gate g(q)=sigmoid(alpha*mean_logprob + beta), (alpha,beta) calibrated on the
distribution of reader confidences (NOT on labels): beta=-median, alpha=4/IQR.
Caches per-query mean log-prob so the GPU pass runs once."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_rerank import load_holdout, build_class_imprints, imprint_match

EBASE = "D:/Data/ePillID_data/ePillID_data/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base"
GT = "D:/Data/pillbox_kaggle/Pillbox.csv"
ALL = f"{EBASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv"
LE = f"{EBASE}/label_encoder_local.pickle"
PRED3 = "outputs/eval_predictions_pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv"
CACHE = "fusion_out/softgate_logprob.csv"
LAM, TOPK = 0.2, 20

def cache_logprobs(paths):
    from imprint_feasibility import QwenReader, normalize_imprint
    from PIL import Image
    reader = QwenReader("Qwen/Qwen2-VL-2B-Instruct", "cuda", adapter="qwen_imprint_lora")
    def fixp(p):
        # data was relocated: D:\ePillID_data -> D:\Data\ePillID_data
        p = str(p).replace("D:\\ePillID_data", "D:\\Data\\ePillID_data").replace(
            "D:/ePillID_data", "D:/Data/ePillID_data")
        return p
    rows = []
    for i, imgs in enumerate(paths):
        reads, lps = [], []
        for p in imgs[:2]:
            p = fixp(p)
            try:
                imp, lp = reader.read_with_logprob(Image.open(p))
                if imp:
                    reads.append(imp); lps.append(lp)
            except Exception as e:
                print(f"  [warn] {p}: {e}")
        pooled = normalize_imprint(" ".join(reads))
        mlp = float(np.mean(lps)) if lps else -10.0
        rows.append({"query_idx": i, "imprint": pooled, "mean_logprob": mlp,
                     "nonempty": int(bool(pooled))})
        if (i + 1) % 50 == 0:
            print(f"  [softgate] {i+1}/{len(paths)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs("fusion_out", exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df

def fuse(S, correct, gates, q_imp, class_imp):
    Sv = (S - S.min(1, keepdims=True)) / (S.max(1, keepdims=True) - S.min(1, keepdims=True) + 1e-8)
    Sf = Sv.copy()
    for i in range(len(correct)):
        if not q_imp[i] or gates[i] <= 0:
            continue
        for c in np.argsort(-S[i])[:TOPK]:
            Sf[i, c] += LAM * gates[i] * imprint_match(q_imp[i], class_imp[c])
    p = np.argmax(Sf, 1); v = np.argmax(Sv, 1)
    top1 = float(100 * np.mean(p == correct))
    rescued = int(np.sum((v != correct) & (p == correct)))
    hurt = int(np.sum((v == correct) & (p != correct)))
    return top1, rescued, hurt

def main():
    _, S, correct, paths = load_holdout(PRED3)
    if not os.path.exists(CACHE):
        print("[softgate] caching reader logprobs (GPU)...")
        cache_logprobs(paths)
    df = pd.read_csv(CACHE).sort_values("query_idx")
    q_imp = df["imprint"].fillna("").astype(str).tolist()
    mlp = df["mean_logprob"].to_numpy()
    _, class_imp = build_class_imprints(LE, GT, ALL)

    # hard gate
    hard_g = (df["nonempty"].to_numpy() > 0).astype(float)
    h_top1, h_res, h_hurt = fuse(S, correct, hard_g, q_imp, class_imp)
    # soft gate: distribution-calibrated (no labels)
    med = float(np.median(mlp)); q1, q3 = np.percentile(mlp, [25, 75]); iqr = max(q3 - q1, 1e-3)
    alpha, beta = 4.0 / iqr, -4.0 / iqr * med
    soft_g = 1.0 / (1.0 + np.exp(-(alpha * mlp + beta)))
    s_top1, s_res, s_hurt = fuse(S, correct, soft_g, q_imp, class_imp)

    out = {"hard": {"top1": h_top1, "rescued": h_res, "hurt": h_hurt},
           "soft": {"top1": s_top1, "rescued": s_res, "hurt": s_hurt},
           "alpha": alpha, "beta": beta, "median_lp": med}
    print(json.dumps(out, indent=2))
    json.dump(out, open("fusion_out/softgate_result.json", "w"), indent=2)
    print("[saved] fusion_out/softgate_result.json")

if __name__ == "__main__":
    main()
