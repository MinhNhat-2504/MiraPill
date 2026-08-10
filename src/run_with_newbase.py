# -*- coding: utf-8 -*-
"""
CPU-phase launcher: run a module's main() with its hardcoded old data roots
(D:/ePillID_data, D:/pillbox_kaggle) redirected to D:/Data/... .

    python run_with_newbase.py fuse_4fold --lam 0.2
"""
import importlib
import sys

NEW_BASE = ("D:/Data/ePillID_data/ePillID_data/folds/"
            "pilltypeid_nih_sidelbls0.01_metric_5folds/base")
NEW_GT = "D:/Data/pillbox_kaggle/Pillbox.csv"


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python run_with_newbase.py <module> [args...]")
    mod_name = sys.argv[1]
    extra = sys.argv[2:]
    mod = importlib.import_module(mod_name)
    if hasattr(mod, "BASE"):
        mod.BASE = NEW_BASE
    # argparse defaults with literal old gt path: override via CLI when the
    # module accepts --gt_csv and the caller did not pass one
    argv = [mod_name + ".py"] + extra
    src = open(mod.__file__, encoding="utf-8", errors="ignore").read()
    if "--gt_csv" in src and not any(a.startswith("--gt_csv") for a in extra):
        argv += ["--gt_csv", NEW_GT]
    sys.argv = argv
    mod.main()


if __name__ == "__main__":
    main()
