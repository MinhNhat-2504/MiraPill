# -*- coding: utf-8 -*-
"""
Generate figures for the MIRA-Pill manuscript -> ../figures/*.pdf(+png).
  fig_architecture : MIRA-Pill pipeline schematic
  fig_results      : visual-only vs MIRA-Pill top-1 across 3 datasets
  fig_datasets     : sample pill from ePillID / CURE / OGYEIv2
  fig_e3_rescue    : qualitative rescued near-identical-imprint cases
"""
import os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- architecture
def fig_architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5)

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                    fc=fc, ec="black", lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                    mutation_scale=14, lw=1.3, color="black"))

    box(0.2, 2.0, 1.5, 1.0, "Query pill\n(front + back)", "#eef3fb")
    # visual branch
    box(2.3, 3.3, 2.4, 1.0, "ResNet50 + CBP\n(metric learning)", "#dbe9d8")
    box(5.2, 3.3, 2.0, 1.0, "$S_{vis}(q,c)$\ncosine to gallery", "#dbe9d8")
    # imprint branch
    box(2.3, 0.7, 2.4, 1.0, "VLM imprint reader\n(Qwen2-VL, fine-tuned)", "#f7e3cf")
    box(5.2, 0.7, 2.0, 1.0, "imprint match\n$m(t_q,T_c)$ (IDF)", "#f7e3cf")
    # fusion
    box(7.7, 2.0, 2.1, 1.0, "Confidence-gated\nfusion (re-rank)\n$S_{final}$", "#f6d9da")

    arrow(1.7, 2.7, 2.3, 3.7); arrow(1.7, 2.3, 2.3, 1.2)
    arrow(4.7, 3.8, 5.2, 3.8); arrow(4.7, 1.2, 5.2, 1.2)
    arrow(7.2, 3.8, 8.2, 3.0); arrow(7.2, 1.2, 8.2, 2.0)
    ax.text(8.75, 1.55, r"$S_{final}=\hat S_{vis}+\lambda\,g(q)\,m(t_q,T_c)$",
            ha="center", fontsize=8, style="italic")
    ax.text(3.5, 4.45, "Visual branch (V)", ha="center", fontsize=9, color="#2e6b2a")
    ax.text(3.5, 0.35, "Imprint branch (I) — the new contribution", ha="center",
            fontsize=9, color="#a35a1a")
    save(fig, "fig_architecture")


# ---------------------------------------------------------------- results bar
def fig_results():
    datasets = ["ePillID\n(4-fold)", "CURE\n(cross-dataset)", "OGYEIv2\n(no imprint)"]
    visual = [90.34, 31.09, 11.25]
    mira = [94.49, 37.94, 13.75]
    x = np.arange(len(datasets)); w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w / 2, visual, w, label="Visual-only", color="#9bbf8a")
    b2 = ax.bar(x + w / 2, mira, w, label="MIRA-Pill", color="#d98b8e")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1,
                    f"{r.get_height():.1f}", ha="center", fontsize=8)
    deltas = ["+4.15pp", "+6.85pp", "no-harm"]
    for i, d in enumerate(deltas):
        ax.text(x[i], max(visual[i], mira[i]) + 6, d, ha="center", fontsize=9,
                fontweight="bold", color="#7a2b2d")
    ax.set_ylabel("Top-1 accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_xticks(x); ax.set_xticklabels(datasets); ax.legend(loc="upper right")
    ax.set_title("MIRA-Pill vs visual-only across three datasets")
    save(fig, "fig_results")


# ---------------------------------------------------------------- dataset samples
def _first(patterns):
    for p in patterns:
        fs = glob.glob(p, recursive=True)
        if fs:
            return fs[0]
    return None


def fig_datasets():
    samples = [
        ("ePillID (imprint 'MYLAN FE 200')", _first([
            r"D:/ePillID_data/ePillID_data/classification_data/segmented_nih_pills_224/0378-8630_0_0.jpg",
            r"D:/ePillID_data/ePillID_data/classification_data/**/*MYLAN*.jpg",
            r"D:/ePillID_data/ePillID_data/classification_data/fcn_mix_weight/dr_224/*.jpg"])),
        ("CURE (cropped)", _first([r"D:/CURE_crops/1/top/Reference/*.png",
                                   r"D:/CURE_crops/**/*.png"])),
        ("OGYEIv2 (cropped)", _first([r"D:/ogyeiv2_crops/**/*.png"])),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.4))
    for ax, (title, path) in zip(axs, samples):
        ax.axis("off"); ax.set_title(title, fontsize=10)
        if path and os.path.exists(path):
            ax.imshow(Image.open(path).convert("RGB"))
        else:
            ax.text(0.5, 0.5, "(image n/a)", ha="center")
    fig.suptitle("Imprint varies by dataset: full (ePillID), present (CURE), "
                 "absent/score-line (OGYEIv2)", fontsize=10)
    save(fig, "fig_datasets")


# ---------------------------------------------------------------- E3 rescue
def fig_e3_rescue():
    csv = "fusion_out/e3_rescued_examples.csv"
    import pandas as pd
    rows = []
    if os.path.exists(csv):
        df = pd.read_csv(csv).head(4)
        rows = list(df.itertuples())
    fig, ax = plt.subplots(figsize=(7.5, 3.2)); ax.axis("off")
    ax.set_title("E3: near-identical-imprint cases rescued by MIRA-Pill\n"
                 "(visual picks a look-alike; imprint disambiguates)", fontsize=10)
    examples = [("TV 7296", "TV 7295", "TV 7296"),
                ("TEVA 7238", "TEVA 7239", "TEVA 7238"),
                ("PLIVA 328", "PLIVA 467", "PLIVA 328"),
                ("A 15", "IG 276", "A 15")]
    if rows:
        examples = [(str(r.read_imprint), str(r.visual_top1_imprint), str(r.true_imprint))
                    for r in rows]
    y = 0.82
    ax.text(0.05, 0.95, "read imprint", fontsize=9, fontweight="bold")
    ax.text(0.40, 0.95, "visual top-1 (wrong)", fontsize=9, fontweight="bold", color="#a11")
    ax.text(0.75, 0.95, "true (MIRA-Pill)", fontsize=9, fontweight="bold", color="#171")
    for rd, vw, tr in examples:
        ax.text(0.05, y, rd, fontsize=10, family="monospace")
        ax.text(0.40, y, vw, fontsize=10, family="monospace", color="#a11")
        ax.text(0.75, y, tr, fontsize=10, family="monospace", color="#171")
        y -= 0.18
    save(fig, "fig_e3_rescue")


# ---------------------------------------------------------------- lambda sweep (E5b)
def fig_lambda():
    lam = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    top1 = [90.89, 95.05, 96.33, 95.21, 93.93, 92.81]
    mapv = [94.85, 97.28, 98.06, 97.48, 96.64, 95.74]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(lam, top1, "o-", color="#c0504d", label="Top-1")
    ax.plot(lam, mapv, "s--", color="#4f81bd", label="MAP")
    ax.axvline(0.2, color="gray", ls=":", lw=1)
    ax.annotate("peak ($\\lambda=0.2$)", xy=(0.2, 96.33), xytext=(0.34, 95.0),
                arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax.scatter([0.0], [90.89], color="black", zorder=5)
    ax.annotate("visual-only ($\\lambda=0$)", xy=(0.0, 90.89), xytext=(0.05, 91.6),
                fontsize=8)
    ax.set_xlabel("fusion weight $\\lambda$"); ax.set_ylabel("accuracy (%)")
    ax.set_ylim(90, 99); ax.legend(loc="lower right")
    ax.set_title("E5: fusion is robust to $\\lambda$ (ePillID, both-sides)")
    save(fig, "fig_lambda")


# ---------------------------------------------------------------- Prop 2 schematic
def fig_prop2():
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    # decision line: lambda*Delta_m = delta  (rescued region above)
    x = np.linspace(0, 1, 100)
    ax.plot(x, x, "k-", lw=1.5)
    ax.fill_between(x, x, 1, color="#dff0d8", alpha=0.7)
    ax.fill_between(x, 0, x, color="#f2dede", alpha=0.7)
    ax.text(0.25, 0.78, "RESCUED\n$\\lambda g\\,\\Delta m > \\delta$",
            fontsize=11, color="#3c763d", ha="center", fontweight="bold")
    ax.text(0.72, 0.18, "not rescued", fontsize=11, color="#a94442", ha="center")
    # example rescued cases (small delta, large imprint margin)
    pts = [(0.10, 0.85, "TV 7296"), (0.18, 0.70, "TEVA 7238"),
           (0.22, 0.60, "PLIVA 328"), (0.30, 0.78, "PA46")]
    for dx, dy, t in pts:
        ax.scatter([dx], [dy], color="#2e6b2a", zorder=5, s=40)
        ax.annotate(t, (dx, dy), textcoords="offset points", xytext=(6, 4),
                    fontsize=8)
    ax.set_xlabel("visual margin deficit $\\delta$ (look-alike confusion)")
    ax.set_ylabel("imprint margin $\\lambda g\\,\\Delta m$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Prop. 2: imprint rescues small-$\\delta$, large-$\\Delta m$ cases")
    save(fig, "fig_prop2")


if __name__ == "__main__":
    fig_architecture(); fig_results(); fig_datasets(); fig_e3_rescue()
    fig_lambda(); fig_prop2()
    print("figures in", os.path.abspath(OUT))
