# -*- coding: utf-8 -*-
"""Extract the 9 real pill images for Figure 8 (qualitative rescued examples)
from the ePillID dataset -> TSP_CMC_MIRAPILL/image/qual_row{1,2,3}_{query,wrong,correct}.png
Uses the actual rescued cases from fusion_out/e3_rescued_examples.csv."""
import os, shutil
import pandas as pd
from PIL import Image

ROOT = "D:/Data/ePillID_data/ePillID_data"
IMGDIR = os.path.join(ROOT, "classification_data")
ALL = f"{ROOT}/folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv"
OUT = r"d:/Project/AI Engineer/ePillID-benchmark-master/_review_extract/TSP_CMC_MIRAPILL/image"

# (row, query_img, wrong_class, correct_class, ref_side)
# ref_side = the face that actually carries the DISCRIMINATIVE imprint: the two
# classes of a rescued pair share the prefix, so we show the face where they
# differ. TV/TEVA carry the digits on the back; PLIVA prints "PLIVA <n>" on the
# front (its back is a plain score line). Verified visually.
# The query column uses the face of the SAME both-sides consumer query that shows
# the imprint the reader actually transcribed (its partner face carries only the
# prefix or is blank), so the figure shows what the VLM read.
CASES = [
    (1, "2410.jpg", "00093-7295-01_31281890", "00093-7296-01_F727FBEF", "back"),
    (2, "770.jpg",  "00093-7239-98_A71DD3FE", "00093-7238-98_731DB9BD", "back"),
    (3, "4901.jpg", "50111-0467-01_54302A61", "50111-0328-01_55212A99", "front"),
]

df = pd.read_csv(ALL)

def ref_image_for(label, side):
    """Reference image of the face carrying the discriminative imprint."""
    g = df[(df["label"].astype(str) == label) & (df["is_ref"])]
    if len(g) == 0:
        return None
    want = g[g["is_front"] == (side == "front")]
    row = (want if len(want) else g).iloc[0]
    return os.path.join(IMGDIR, str(row["image_path"]).replace("/", os.sep))

def query_image(fname):
    """consumer query image lives under fcn_mix_weight/dc_224/."""
    g = df[(~df["is_ref"]) & (df["image_path"].astype(str).str.endswith("/" + fname))]
    if len(g):
        return os.path.join(IMGDIR, str(g.iloc[0]["image_path"]).replace("/", os.sep))
    p = os.path.join(IMGDIR, "fcn_mix_weight", "dc_224", fname)
    return p if os.path.exists(p) else None

def save(src, dst):
    if not src or not os.path.exists(src):
        print(f"  !! MISSING {src}")
        return False
    im = Image.open(src).convert("RGB")
    im.save(dst)
    print(f"  ok {os.path.basename(dst):26} <- {os.path.basename(src)}  {im.size}")
    return True

os.makedirs(OUT, exist_ok=True)
ok = 0
for row, qimg, wrong, correct, side in CASES:
    print(f"[row {row}] discriminative side = {side}")
    ok += save(query_image(qimg), f"{OUT}/qual_row{row}_query.png")
    ok += save(ref_image_for(wrong, side), f"{OUT}/qual_row{row}_wrong.png")
    ok += save(ref_image_for(correct, side), f"{OUT}/qual_row{row}_correct.png")
print(f"\n{ok}/9 images written to {OUT}")
