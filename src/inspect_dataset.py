# -*- coding: utf-8 -*-
"""
Generic dataset inspector — point it at an extracted dataset folder (e.g. the
Kaggle OGYEIv2 download) and it reports the structure needed to adapt the
MIRA-Pill pipeline: directory tree, image counts per top-level folder, and any
data files (CSV/JSON/TXT/XLSX) with columns + a sample row, flagging text/imprint
/appearance/NDC fields.

    python inspect_dataset.py --local "D:\\ogyeiv2"
"""
import argparse
import glob
import json
import os

import pandas as pd

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
DATA_EXT = (".csv", ".tsv", ".tab", ".json", ".txt", ".xlsx", ".jsonl")
TEXT_HINTS = ("imprint", "text", "desc", "appearance", "ndc", "name", "label",
              "color", "shape", "class", "drug", "leaflet", "caption")


def tree(root, max_depth=3):
    root = os.path.abspath(root)
    print(f"\n[tree] {root} (depth {max_depth})")
    base_depth = root.rstrip(os.sep).count(os.sep)
    for cur, dirs, files in os.walk(root):
        depth = cur.count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        n_img = sum(1 for f in files if f.lower().endswith(IMG_EXT))
        n_dat = sum(1 for f in files if f.lower().endswith(DATA_EXT))
        indent = "  " * depth
        tag = []
        if n_img:
            tag.append(f"{n_img} imgs")
        if n_dat:
            tag.append(f"{n_dat} data-files")
        print(f"{indent}{os.path.basename(cur) or cur}/  {'| '.join(tag)}")
        dirs.sort()


def inspect_data_files(root):
    print("\n[data files] columns + sample (flag text/imprint fields):")
    files = [f for f in glob.glob(os.path.join(root, "**", "*"), recursive=True)
             if f.lower().endswith(DATA_EXT)]
    for f in sorted(files)[:40]:
        size = round(os.path.getsize(f) / 1e6, 2)
        print(f"\n  >>> {os.path.relpath(f, root)} ({size} MB)")
        try:
            if f.lower().endswith((".json", ".jsonl")):
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    first = fh.readline().strip()
                print(f"      json head: {first[:300]}")
                continue
            sep = "\t" if f.lower().endswith((".tab", ".tsv")) else None
            df = None
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(f, sep=sep, engine="python", dtype=str,
                                     nrows=3, encoding=enc, on_bad_lines="skip")
                    break
                except Exception:
                    continue
            if df is None:
                print("      (could not read)")
                continue
            cols = list(df.columns)
            flagged = [c for c in cols if any(h in c.lower() for h in TEXT_HINTS)]
            print(f"      columns ({len(cols)}): {cols[:30]}")
            if flagged:
                print(f"      *** text/imprint-like: {flagged}")
            if len(df):
                row = {c: str(df.iloc[0][c])[:40] for c in cols[:8]}
                print(f"      sample row: {row}")
        except Exception as e:
            print(f"      (error: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", required=True, help="extracted dataset folder")
    ap.add_argument("--max_depth", type=int, default=3)
    args = ap.parse_args()
    if not os.path.isdir(args.local):
        raise SystemExit(f"folder not found: {args.local}")

    all_files = glob.glob(os.path.join(args.local, "**", "*"), recursive=True)
    n_img = sum(1 for f in all_files if f.lower().endswith(IMG_EXT))
    n_dat = sum(1 for f in all_files if f.lower().endswith(DATA_EXT))
    print(f"[summary] {len(all_files)} entries | {n_img} images | {n_dat} data files")
    tree(args.local, args.max_depth)
    inspect_data_files(args.local)
    print("\n[done] Paste this output back so the pipeline can be adapted.")


if __name__ == "__main__":
    main()
