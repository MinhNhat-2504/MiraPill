# -*- coding: utf-8 -*-
"""
Local runner for the ORIGINAL ePillID visual model (ResNet50 + CBP multi-head
metric learning) — no Azure ML, runs on torch 2.5.

Reuses the author's training/eval pipeline (multihead_trainer.train via
train_nocv.run) unchanged. The azureml shim (src/azureml/) makes it import; we
point --label_encoder at a fresh path so train_nocv FITS a new LabelEncoder from
the data (sidesteps the old-sklearn pickle). Trains one fold, saves the model
.pth and per-query prediction scores (used later as S_visual for fusion).

Smoke (1 epoch, small batch):
    python run_visual_train.py --max_epochs 1 --batch_size 12 --smoke

Full:
    python run_visual_train.py --max_epochs 40 --batch_size 16
"""
import argparse
import os
import sys

# old-sklearn pickle safety net (only needed if a legacy pickle is ever loaded)
try:
    import sklearn.preprocessing._label as _lbl
    sys.modules.setdefault("sklearn.preprocessing.label", _lbl)
except Exception:
    pass

import arguments
import train_nocv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root_dir", default="D:/ePillID_data/ePillID_data")
    ap.add_argument("--fold_base",
                    default="folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base")
    ap.add_argument("--val_fold", default="pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv")
    ap.add_argument("--test_fold", default="pilltypeid_nih_sidelbls0.01_metric_5folds_4.csv")
    ap.add_argument("--max_epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--results_dir", default=os.path.abspath("visual_results"))
    ap.add_argument("--img_dir", default="classification_data",
                    help="subdir under data_root_dir holding images ('.' for CURE crops)")
    ap.add_argument("--all_csv", default=None, help="override all_imgs_csv (rel to data_root)")
    ap.add_argument("--no_sides", action="store_true",
                    help="dataset has no front/back sides (e.g. OGYEIv2) -> disable "
                         "side labels and both-sides pairing")
    ap.add_argument("--network", default="resnet50", help="backbone (resnet50/resnet152)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="set torch/numpy/random seed for multi-seed runs")
    cli = ap.parse_args()

    if cli.seed is not None:
        import random
        import numpy as _np
        random.seed(cli.seed)
        _np.random.seed(cli.seed)
        import torch as _torch
        _torch.manual_seed(cli.seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(cli.seed)
        print(f"[run_visual_train] SEED set to {cli.seed}")

    # build the full arg namespace from the author's parser defaults, then override
    args = arguments.nocv_parser().parse_args([])
    args.data_root_dir = os.path.abspath(cli.data_root_dir)
    args.img_dir = cli.img_dir
    args.appearance_network = cli.network
    args.pooling = "CBP"
    args.metric_evaluator_type = "cosine"
    args.batch_size = cli.batch_size
    args.max_epochs = cli.max_epochs
    args.results_dir = cli.results_dir
    args.supress_warnings = True
    if cli.no_sides:
        args.train_with_side_labels = 0
        args.metric_simul_sidepairs_eval = 0

    base = cli.fold_base
    args.all_imgs_csv = cli.all_csv or f"{base}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv"
    args.val_imgs_csv = f"{base}/{cli.val_fold}"
    args.test_imgs_csv = f"{base}/{cli.test_fold}"
    # fresh path -> train_nocv fits a new LabelEncoder from the data (no old pickle)
    args.label_encoder = f"{base}/label_encoder_local.pickle"

    if cli.smoke:
        print("[smoke] 1-fold, few epochs, small batch — sanity only")

    print(f"[run_visual_train] data_root={args.data_root_dir}")
    print(f"[run_visual_train] net={args.appearance_network}+{args.pooling} "
          f"bs={args.batch_size} epochs={args.max_epochs}")
    print(f"[run_visual_train] results_dir={args.results_dir}")

    metrics_df, predictions_df = train_nocv.run(args)

    print("\n[run_visual_train] DONE.")
    print("[run_visual_train] metrics (holdout):")
    try:
        m = metrics_df[metrics_df["name"].str.contains("global_ap|map|accuracy",
                                                       case=False, na=False)]
        print(m.to_string(index=False))
    except Exception as e:
        print("  (could not filter metrics:", e, ")")
    print(f"[run_visual_train] model + predictions saved under {args.results_dir}")


if __name__ == "__main__":
    main()
