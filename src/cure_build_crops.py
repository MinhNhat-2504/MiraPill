# -*- coding: utf-8 -*-
"""
CURE step 1: crop every image to a tight 224x224 pill and write a manifest in
the ePillID CSV schema, so the existing training/eval pipeline can be reused.

Input : D:\\CURE\\Pill_Images\\{class}\\{top|bottom}\\{Customer|Reference}\\img
Output: D:\\CURE_crops\\<same-relpath>.png  +  manifest.csv
manifest columns mirror ePillID: images, pilltype_id, label_code_id,
prod_code_id, is_ref, is_front, is_new, image_path, label, crop_method
"""
import argparse
import glob
import os
import pandas as pd

from cure_crop import crop_pill


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=r"D:\CURE\Pill_Images")
    ap.add_argument("--dst", default=r"D:\CURE_crops")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    imgs = [f for f in glob.glob(os.path.join(args.src, "**", "*.*"), recursive=True)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    print(f"[crops] {len(imgs)} images to process")
    rows, n_fail = [], 0
    for i, f in enumerate(imgs):
        relparts = f.split("Pill_Images" + os.sep)[1].split(os.sep)
        cls = relparts[0]
        side = relparts[1]                       # top / bottom
        domain = relparts[2]                     # Customer / Reference
        is_ref = int(domain == "Reference")
        is_front = int(side == "top")
        rel_png = os.path.join(cls, side, domain,
                               os.path.splitext(os.path.basename(f))[0] + ".png")
        out_path = os.path.join(args.dst, rel_png)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        try:
            img, method, found = crop_pill(f, size=args.size)
            img.save(out_path)
            if not found:
                n_fail += 1
        except Exception as e:
            method = "error"; n_fail += 1
            print(f"  [warn] {f}: {e}")
            continue
        rows.append({
            "images": rel_png.replace(os.sep, "/"),
            "pilltype_id": cls, "label_code_id": int(cls), "prod_code_id": 0,
            "is_ref": bool(is_ref), "is_front": bool(is_front), "is_new": False,
            "image_path": rel_png.replace(os.sep, "/"),
            "label": cls, "crop_method": method,
        })
        if (i + 1) % 500 == 0:
            print(f"  [crops] {i+1}/{len(imgs)}  (fallback/fail so far: {n_fail})", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(args.dst, exist_ok=True)
    mpath = os.path.join(args.dst, "manifest.csv")
    df.to_csv(mpath, index=False)
    from collections import Counter
    print(f"\n[crops] DONE. cropped={len(df)}  fallback/center={n_fail}")
    print("[crops] methods:", dict(Counter(df["crop_method"])))
    print(f"[crops] manifest -> {mpath}")
    print(f"[crops] classes={df['pilltype_id'].nunique()}  "
          f"ref={int(df['is_ref'].sum())}  cust={int((~df['is_ref']).sum())}")


if __name__ == "__main__":
    main()
