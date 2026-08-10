"""
Download the Kaggle 'dhuh137/pillbox' dataset and auto-detect which file holds
the Pillbox master data (imprint + color + shape + NDC/product_code).

Setup (one time):
    pip install kagglehub pandas
    # Kaggle auth: kaggle.com -> Settings -> API -> "Create New Token"
    #   saves kaggle.json -> put it at  C:\\Users\\<you>\\.kaggle\\kaggle.json
    # (or set env vars KAGGLE_USERNAME / KAGGLE_KEY)

Run:
    python fetch_pillbox_kaggle.py
"""

import os
import glob


IMPRINT_HINTS = ("splimprint", "imprint")
KEY_HINTS = ("product_code", "ndc", "label_prod_code", "product_ndc", "ndc9")


def inspect_tabular(path):
    """Return (has_imprint, has_key, columns) for a CSV/TSV/tab file, or None."""
    import pandas as pd
    sep = "\t" if path.lower().endswith((".tab", ".tsv")) else None
    for enc in ("cp1252", "utf-8", "latin-1"):
        try:
            head = pd.read_csv(path, sep=sep, dtype=str, encoding=enc,
                               engine="python", nrows=5, on_bad_lines="skip")
            cols = [c.lower().strip() for c in head.columns]
            has_imp = any(any(h in c for h in IMPRINT_HINTS) for c in cols)
            has_key = any(any(h in c for h in KEY_HINTS) for c in cols)
            return has_imp, has_key, list(head.columns)
        except Exception:
            continue
    return None


def inspect_sql(path):
    """Scan a .sql dump for table/column names hinting at imprint + ndc."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            blob = f.read(2_000_000).lower()  # first ~2MB is enough for schema
    except Exception:
        return None
    has_imp = any(h in blob for h in IMPRINT_HINTS)
    has_key = any(h in blob for h in KEY_HINTS)
    return has_imp, has_key, None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=None,
                    help="path to a folder you already extracted (skips API download)")
    args = ap.parse_args()

    if args.local:
        path = args.local
        if not os.path.isdir(path):
            raise SystemExit(f"folder not found: {path}")
        print("[local] scanning extracted folder:", path)
    else:
        try:
            import kagglehub
        except ImportError:
            raise SystemExit("kagglehub not installed -> pip install kagglehub pandas\n"
                             "(or download the zip manually and run with "
                             "--local <extracted_folder>)")
        print("[download] dhuh137/pillbox (~1.16 GB, first run only)...")
        path = kagglehub.dataset_download("dhuh137/pillbox")
        print("[download] dataset files at:", path)

    all_files = glob.glob(os.path.join(path, "**", "*"), recursive=True)
    data_files = [f for f in all_files
                  if f.lower().endswith((".csv", ".tsv", ".tab", ".sql", ".txt"))]
    print(f"\n[scan] {len(all_files)} files total; "
          f"{len(data_files)} candidate data files:\n")

    winners = []
    for f in sorted(data_files):
        size_mb = round(os.path.getsize(f) / 1e6, 2)
        if f.lower().endswith(".sql"):
            res = inspect_sql(f)
        else:
            res = inspect_tabular(f)
        if res is None:
            print(f"  [skip] {os.path.basename(f)} ({size_mb} MB) - unreadable")
            continue
        has_imp, has_key, cols = res
        tag = "  <-- HAS IMPRINT + NDC KEY" if (has_imp and has_key) else ""
        print(f"  {os.path.basename(f)} ({size_mb} MB) "
              f"imprint={has_imp} key={has_key}{tag}")
        if cols:
            print(f"        columns: {cols[:20]}")
        if has_imp and has_key:
            winners.append(f)

    print("\n" + "=" * 60)
    if winners:
        print("FOUND the master-data file(s):")
        for w in winners:
            print("   ", w)
        print("\nNext: send this path to Claude to build the imprint GT CSV, or run\n"
              "      the feasibility scorer directly:\n"
              f"      python imprint_feasibility.py --readers both \\\n"
              f"        --gt_csv \"{winners[0]}\" \\\n"
              f"        --data_root_dir \"D:/ePillID_data/ePillID_data\"")
    else:
        print("No file with BOTH imprint and NDC columns was found.")
        print("Paste the file list above to Claude to decide next steps.")
    print("=" * 60)


if __name__ == "__main__":
    main()
