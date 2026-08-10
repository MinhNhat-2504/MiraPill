# -*- coding: utf-8 -*-
"""
E6 — CLIP zero-shot baseline (positioning). Off-the-shelf CLIP matches each pill
image to a per-class attribute text ("a photo of a {color} {shape} pill imprinted
with {imprint}") built from Pillbox. Shows a generic vision-language model is
insufficient for fine-grained pill ID, motivating MIRA-Pill (fine-tuned imprint
reader + fusion).
"""
import argparse
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fusion_rerank import load_holdout, label_to_lpc, parse_paths
from imprint_feasibility import normalize_imprint, product_code_to_label_prod_code

BASE = ("D:/ePillID_data/ePillID_data/folds/"
        "pilltypeid_nih_sidelbls0.01_metric_5folds/base")

try:
    import sklearn.preprocessing._label as _lbl
    sys.modules.setdefault("sklearn.preprocessing.label", _lbl)
except Exception:
    pass


def class_attribute_texts(label_encoder_path, gt_csv, all_imgs_csv):
    with open(label_encoder_path, "rb") as f:
        le = pickle.load(f)
    classes = list(le.classes_)

    df = pd.read_csv(gt_csv, dtype=str, encoding="latin-1", on_bad_lines="skip")
    cols = {c.upper().strip(): c for c in df.columns}
    pc, imp = cols.get("PRODUCT_CODE"), cols.get("SPLIMPRINT")
    col = cols.get("SPLCOLOR_TEXT") or cols.get("SPLCOLOR")
    shp = cols.get("SPLSHAPE_TEXT") or cols.get("SPLSHAPE")
    df = df[[pc, imp, col, shp]].dropna(subset=[pc])
    df["lpc"] = df[pc].apply(product_code_to_label_prod_code)
    df = df.dropna(subset=["lpc"]).drop_duplicates("lpc", keep="first").set_index("lpc")

    fold = pd.read_csv(all_imgs_csv)
    lab2lpc = dict(zip(fold["label"].astype(str),
                       fold["label_code_id"].astype(str) + "-" + fold["prod_code_id"].astype(str)))

    texts = []
    for lab in classes:
        lpc = lab2lpc.get(str(lab)) or label_to_lpc(lab)
        if lpc in df.index:
            r = df.loc[lpc]
            color = str(r[col]).lower().replace(";", " ") if pd.notna(r[col]) else ""
            shape = str(r[shp]).lower() if pd.notna(r[shp]) else ""
            imprint = normalize_imprint(r[imp]) if pd.notna(r[imp]) else ""
        else:
            color = shape = imprint = ""
        t = f"a photo of a {color} {shape} pill"
        if imprint:
            t += f" imprinted with {imprint}"
        texts.append(t.replace("  ", " ").strip())
    return classes, texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv",
                    default="outputs/eval_predictions_pilltypeid_nih_sidelbls0.01_metric_5folds_3.csv")
    ap.add_argument("--label_encoder", default=f"{BASE}/label_encoder_local.pickle")
    ap.add_argument("--gt_csv", default="D:/pillbox_kaggle/Pillbox.csv")
    ap.add_argument("--all_imgs_csv",
                    default=f"{BASE}/pilltypeid_nih_sidelbls0.01_metric_5folds_all.csv")
    ap.add_argument("--clip_model", default="laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
                    help="must have safetensors weights (transformers 5.5 blocks .bin)")
    args = ap.parse_args()

    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(dev).eval()
    except Exception:
        # fallback to a repo guaranteed to ship safetensors
        args.clip_model = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True).to(dev).eval()
    proc = CLIPProcessor.from_pretrained(args.clip_model)
    print(f"[e6] CLIP model: {args.clip_model}")

    classes, texts = class_attribute_texts(args.label_encoder, args.gt_csv, args.all_imgs_csv)
    print(f"[e6] classes={len(classes)}  sample text: '{texts[0]}'")

    def norm(x):
        return x / x.norm(dim=-1, keepdim=True)

    # encode class texts (explicit text_model -> projection: version-robust)
    tfeat = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            b = proc(text=texts[i:i+256], return_tensors="pt", padding=True,
                     truncation=True).to(dev)
            out = model.text_model(**b)
            f = model.text_projection(out.pooler_output)
            tfeat.append(norm(f).cpu())
    tfeat = torch.cat(tfeat)  # [C, d]

    h, S, correct, paths = load_holdout(args.pred_csv)
    # encode query images (mean of both sides)
    qfeat = []
    with torch.no_grad():
        for imgs in paths:
            ims = [Image.open(p).convert("RGB") for p in imgs[:2]]
            b = proc(images=ims, return_tensors="pt").to(dev)
            out = model.vision_model(pixel_values=b["pixel_values"])
            f = norm(model.visual_projection(out.pooler_output))
            qfeat.append(f.mean(0, keepdim=True).cpu())
    qfeat = torch.cat(qfeat)  # [Q, d]

    sim = (qfeat @ tfeat.T).numpy()  # [Q, C]
    order = np.argsort(-sim, axis=1)
    top1 = np.mean([correct[i] == order[i, 0] for i in range(len(correct))])
    top5 = np.mean([correct[i] in order[i, :5] for i in range(len(correct))])
    ranks = [1 + int(np.sum(sim[i] > sim[i, correct[i]])) for i in range(len(correct))]
    mapr = float(np.mean([1.0 / r for r in ranks]))

    print("=" * 56)
    print("E6 — CLIP zero-shot (off-the-shelf) on ePillID holdout")
    print("=" * 56)
    print(f"  CLIP zero-shot : top1={top1:.4f}  top5={top5:.4f}  MAP={mapr:.4f}")
    print(f"  (vs Visual-only 0.9089 / MIRA-Pill 0.9633 top-1)")
    print("=" * 56)


if __name__ == "__main__":
    main()
