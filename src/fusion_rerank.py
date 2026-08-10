# -*- coding: utf-8 -*-
"""
MIRA-Pill fusion (visual + imprint) re-ranker — core contribution #2.

S_visual comes straight from the author evaluator's saved predictions
(`eval_predictions_*.csv`, column `similarity` = cosine to all 4902 classes,
the exact both-sides protocol). The fine-tuned imprint reader reads each query
pill's imprint; we match it against every candidate class's Pillbox imprint to
get S_imprint, then fuse with confidence gating:

    S_final = norm(S_visual) + lambda * g(conf) * S_imprint

g(conf)=1 when the reader returns a non-empty imprint, else 0 -> when imprint is
unreadable (blank/score-only faces) fusion == visual baseline (no harm); when
readable it disambiguates visually-identical pills.

Two phases (decoupled so GPU runs once, tuning is CPU-only):
  1) --cache_imprints : run the (fine-tuned) Qwen reader on each holdout query's
     side images, save imprint+flag to a cache CSV.   [GPU]
  2) default          : load S_visual + imprint cache + Pillbox DB, fuse, report
     visual-only vs fusion metrics.                    [CPU, fast; tune lambda]

Examples:
  python fusion_rerank.py --pred_csv outputs/eval_predictions_..._3.csv \\
      --label_encoder <fresh label_encoder_local.pickle> \\
      --gt_csv "D:\\pillbox_kaggle\\Pillbox.csv" \\
      --adapter qwen_imprint_lora --data_root_dir "D:/ePillID_data/ePillID_data" \\
      --cache_imprints --imprint_cache fusion_out/imprints.csv
  python fusion_rerank.py --pred_csv ... --label_encoder ... --gt_csv ... \\
      --imprint_cache fusion_out/imprints.csv --lam 0.5
"""
import argparse
import ast
import os
import pickle
import sys

import numpy as np
import pandas as pd

from imprint_feasibility import (normalize_imprint, clean_gt_imprint,
                                 load_ground_truth_imprints,
                                 product_code_to_label_prod_code)

# old-sklearn pickle safety net
try:
    import sklearn.preprocessing._label as _lbl
    sys.modules.setdefault("sklearn.preprocessing.label", _lbl)
except Exception:
    pass


def parse_vec(s):
    return np.fromstring(str(s).replace("[", "").replace("]", "").replace("\n", " "),
                         sep=" ")


def parse_paths(s):
    """img_path field is a python-list-as-string of the query's side images."""
    try:
        v = ast.literal_eval(str(s))
        return v if isinstance(v, (list, tuple)) else [v]
    except Exception:
        return [str(s)]


def label_to_lpc(label):
    """class label (e.g. '51285-0092-87_BE305F72') -> label_prod_code '51285-92'."""
    ndc = str(label).split("_")[0]
    return product_code_to_label_prod_code(ndc)


def imprint_match(tq, tc):
    """Asymmetric match of a (possibly partial) read imprint tq against a
    candidate's full DB imprint tc. Returns [0,1]: fraction of read tokens found
    in the candidate, with a char-overlap tie-breaker."""
    tq, tc = normalize_imprint(tq), normalize_imprint(tc)
    if not tq or not tc:
        return 0.0
    qtok, ctok = tq.split(), set(tc.split())
    if not qtok:
        return 0.0
    tok_frac = sum(t in ctok for t in qtok) / len(qtok)
    # char-level bonus: read string contained in candidate (ignoring spaces)
    contains = 1.0 if tq.replace(" ", "") in tc.replace(" ", "") else 0.0
    return 0.85 * tok_frac + 0.15 * contains


# ---------------- metrics ----------------
def topk_acc(scores, correct, k):
    order = np.argsort(-scores, axis=1)[:, :k]
    return np.mean([c in order[i] for i, c in enumerate(correct)])


def mean_ap_at_rank(scores, correct):
    """MAP where each query has exactly one correct class: AP = 1/rank."""
    ranks = []
    for i, c in enumerate(correct):
        rank = 1 + int(np.sum(scores[i] > scores[i, c]))
        ranks.append(1.0 / rank)
    return float(np.mean(ranks))


def report(name, scores, correct):
    print(f"  {name:14s} | top1={topk_acc(scores, correct, 1):.4f}  "
          f"top5={topk_acc(scores, correct, 5):.4f}  "
          f"MAP={mean_ap_at_rank(scores, correct):.4f}")
    return topk_acc(scores, correct, 1), mean_ap_at_rank(scores, correct)


def load_holdout(pred_csv):
    df = pd.read_csv(pred_csv)
    h = df[df["dataset"] == "holdout"].reset_index(drop=True)
    S = np.stack([parse_vec(s) for s in h["similarity"]])
    correct = h["correct_index"].astype(int).to_numpy()
    paths = [parse_paths(p) for p in h["img_path"]]
    return h, S, correct, paths


def build_class_imprints(label_encoder_path, gt_csv, all_imgs_csv):
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)
    classes = list(le.classes_)
    gt = load_ground_truth_imprints(gt_csv)
    lpc2imp = dict(zip(gt["label_prod_code"], gt["splimprint"]))

    # class labels are a mix of NDC strings AND sha224 hashes -> cannot parse the
    # hashes to an NDC. The fold CSV carries label_code_id/prod_code_id for EVERY
    # label, so build label -> label_prod_code from the columns (fall back to NDC
    # parse if a label is missing from the CSV).
    fold = pd.read_csv(all_imgs_csv)
    fold_lpc = (fold["label_code_id"].astype(str) + "-"
                + fold["prod_code_id"].astype(str))
    lab2lpc = dict(zip(fold["label"].astype(str), fold_lpc))

    class_imp = []
    for lab in classes:
        lpc = lab2lpc.get(str(lab)) or label_to_lpc(lab)
        class_imp.append(normalize_imprint(lpc2imp.get(lpc, "")))
    n_have = sum(1 for x in class_imp if x)
    print(f"[db] classes={len(classes)}  with imprint={n_have} "
          f"({100*n_have/len(classes):.1f}%)")
    return classes, class_imp


def cache_imprints(paths, args):
    """GPU phase: read each query pill's imprint with the (fine-tuned) reader."""
    from imprint_feasibility import QwenReader, preprocess_image
    from PIL import Image
    reader = QwenReader(args.qwen_model, args.device, adapter=args.adapter)
    rows = []
    for i, imgs in enumerate(paths):
        reads = []
        for p in imgs[:2]:
            try:
                reads.append(reader.read(Image.open(p)))
            except Exception as e:
                print(f"  [warn] {p}: {e}")
        pooled = normalize_imprint(" ".join(r for r in reads if r))
        rows.append({"query_idx": i, "imprint": pooled,
                     "nonempty": int(bool(pooled))})
        if (i + 1) % 25 == 0 or (i + 1) == len(paths):
            print(f"  [imprint] {i+1}/{len(paths)}")
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.imprint_cache) or ".", exist_ok=True)
    out.to_csv(args.imprint_cache, index=False)
    print(f"[cache] saved imprint readings -> {args.imprint_cache}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--label_encoder", required=True)
    ap.add_argument("--gt_csv", required=True)
    ap.add_argument("--adapter", default="qwen_imprint_lora")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data_root_dir", default="D:/ePillID_data/ePillID_data")
    ap.add_argument("--all_imgs_csv",
                    default="D:/ePillID_data/ePillID_data/folds/"
                            "pilltypeid_nih_sidelbls0.01_metric_5folds/base/"
                            "pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv",
                    help="fold all.csv (provides label -> label_prod_code)")
    ap.add_argument("--imprint_cache", default="fusion_out/imprints.csv")
    ap.add_argument("--cache_imprints", action="store_true")
    ap.add_argument("--lam", type=float, default=0.5, help="imprint fusion weight")
    ap.add_argument("--topk", type=int, default=20, help="re-rank within top-K visual")
    args = ap.parse_args()

    h, S, correct, paths = load_holdout(args.pred_csv)
    print(f"[load] holdout queries={len(correct)}  S_visual shape={S.shape}")

    if args.cache_imprints:
        cache_imprints(paths, args)
        return

    classes, class_imp = build_class_imprints(args.label_encoder, args.gt_csv,
                                              args.all_imgs_csv)
    if not os.path.exists(args.imprint_cache):
        sys.exit(f"[ERROR] imprint cache not found: {args.imprint_cache}\n"
                 f"        run once with --cache_imprints first.")
    imp = pd.read_csv(args.imprint_cache).sort_values("query_idx")
    q_imprint = imp["imprint"].fillna("").astype(str).tolist()

    # normalize S_visual per query to [0,1]-ish for fusion stability
    Sv = S.copy()
    Sv = (Sv - Sv.min(axis=1, keepdims=True)) / (
        Sv.max(axis=1, keepdims=True) - Sv.min(axis=1, keepdims=True) + 1e-8)

    # build S_imprint only over each query's top-K visual candidates (efficiency);
    # confidence gate g=1 iff the reader returned a non-empty imprint
    Sf = Sv.copy()
    class_imp_arr = class_imp
    n_boosted = 0
    for i in range(len(correct)):
        tq = q_imprint[i] if i < len(q_imprint) else ""
        if not tq:
            continue
        n_boosted += 1
        cand = np.argsort(-S[i])[:args.topk]
        for c in cand:
            Sf[i, c] += args.lam * imprint_match(tq, class_imp_arr[c])

    print(f"[fuse] queries with a readable imprint (gated on): {n_boosted}/{len(correct)}  "
          f"lambda={args.lam}  topK={args.topk}\n")
    print("RESULTS (holdout, both-sides):")
    report("visual-only", S, correct)
    report("MIRA-Pill", Sf, correct)


if __name__ == "__main__":
    main()
