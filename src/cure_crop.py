# -*- coding: utf-8 -*-
"""
CURE pill localizer/cropper. CURE images are 2448x2448 with a small pill on a
(often textured) background. We crop to the pill so the visual model and imprint
reader see a tight pill (like ePillID's pre-segmented 224 crops).

Multi-strategy detector (first that yields a confident pill wins):
  1) saturation/brightness blob: pill differs from cardboard in HSV
  2) Otsu on |gray - bg_median| + circularity filter
  3) Hough circles
  4) fallback: center square crop
Returns (PIL 224x224, method, found_flag).
"""
import numpy as np
from PIL import Image
import cv2


def _pick_contour(mask, H, W):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, bestscore = None, 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.0015 * H * W or a > 0.55 * H * W:
            continue
        p = cv2.arcLength(c, True)
        if p == 0:
            continue
        circ = 4 * np.pi * a / (p * p)
        x, y, w, h = cv2.boundingRect(c)
        ar = w / max(h, 1)
        if ar < 0.4 or ar > 2.5 or circ < 0.45:
            continue
        if circ * a > bestscore:
            best, bestscore = (x, y, w, h), circ * a
    return best


def detect(arr):
    H, W = arr.shape[:2]
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    g = cv2.GaussianBlur(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), (9, 9), 0)

    # 1) HSV: pill is either much brighter or more saturated than the bg median
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    sat = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    bright = (v > int(np.median(v)) + 35).astype(np.uint8) * 255
    for mask in (cv2.bitwise_or(sat, bright), bright, sat):
        b = _pick_contour(mask, H, W)
        if b:
            return b, "hsv"

    # 2) Otsu on abs-diff from bg
    diff = np.abs(g.astype(np.int16) - np.median(g)).astype(np.uint8)
    th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    b = _pick_contour(th, H, W)
    if b:
        return b, "otsu"

    # 3) Hough circles
    circ = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=H,
                            param1=120, param2=40,
                            minRadius=int(0.03 * H), maxRadius=int(0.35 * H))
    if circ is not None:
        cx, cy, r = np.uint16(np.around(circ))[0][0]
        return (int(cx - r), int(cy - r), int(2 * r), int(2 * r)), "hough"
    return None, "fail"


def crop_pill(path, size=224, margin=0.30):
    arr = np.array(Image.open(path).convert("RGB"))
    H, W = arr.shape[:2]
    box, method = detect(arr)
    if box is None:
        s = int(min(H, W) * 0.4)                       # fallback: center crop
        x0, y0, method = (W - s) // 2, (H - s) // 2, "center"
    else:
        x, y, w, h = box
        m = int(max(w, h) * margin); s = max(w, h) + 2 * m
        cx, cy = x + w // 2, y + h // 2
        x0 = max(0, min(cx - s // 2, W - s)); y0 = max(0, min(cy - s // 2, H - s))
    crop = arr[y0:y0 + s, x0:x0 + s]
    return Image.fromarray(crop).resize((size, size)), method, box is not None


if __name__ == "__main__":
    import argparse, glob, os, random
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"D:\CURE\Pill_Images")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", default="feasibility_out/cure_crops")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cust = [f for f in glob.glob(os.path.join(args.root, "**", "Customer", "*.*"),
                                 recursive=True) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    random.seed(0); sample = random.sample(cust, min(args.n, len(cust)))
    from collections import Counter
    methods = Counter()
    for i, f in enumerate(sample):
        img, method, found = crop_pill(f)
        methods[method] += 1
        img.save(os.path.join(args.out, f"{i:02d}_{method}_{os.path.basename(f)}.png"))
    print("detection methods over", len(sample), "Customer images:", dict(methods))
    print("found (non-fallback):", sum(v for k, v in methods.items() if k != "center"),
          "/", len(sample))
    print("crops saved to", args.out)
