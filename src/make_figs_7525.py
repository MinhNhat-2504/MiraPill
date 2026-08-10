# -*- coding: utf-8 -*-
"""Regenerate the three number-bearing manuscript figures for the 75/25 switch,
matching the style of the existing image/ pngs. Writes to figures_7525/."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_7525")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE = "#5B8DB8", "#DE7E33"


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("saved", name)


# ---------------- fig_folds: per-fold Top-1, visual vs MIRA ----------------
df = pd.read_csv("fusion_out/e1_per_fold.csv")
vis = (100 * df["v_top1"]).round(2).tolist()
mira = (100 * df["m_top1"]).round(2).tolist()
x = np.arange(4)
w = 0.38
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.bar(x - w / 2, vis, w, label="Visual-only", color=BLUE, edgecolor="black", lw=0.6)
ax.bar(x + w / 2, mira, w, label="MIRA-Pill", color=ORANGE, edgecolor="black", lw=0.6)
for i in range(4):
    ax.text(x[i] + w / 2, mira[i] + 0.35, f"+{mira[i]-vis[i]:.2f}",
            ha="center", fontsize=9, fontweight="bold", color=ORANGE)
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_ylim(85, 98.5)
ax.set_xticks(x)
ax.set_xticklabels([f"fold_{i}" for i in range(4)])
ax.legend(loc="upper left", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_folds")

# ---------------- fig_main_bar: 3 datasets ----------------
datasets = ["ePillID\n(imprint full)", "CURE\n(imprint, built DB)", "OGYEIv2\n(no imprint)"]
visual = [90.34, 31.09, 11.25]
mira3 = [94.33, 38.38, 14.17]
deltas = ["+3.99pp", "+7.29pp", "+2.92pp"]
x = np.arange(3)
w = 0.36
fig, ax = plt.subplots(figsize=(7.0, 3.8))
b1 = ax.bar(x - w / 2, visual, w, label="Visual-only", color=BLUE, edgecolor="black", lw=0.6)
b2 = ax.bar(x + w / 2, mira3, w, label="MIRA-Pill", color=ORANGE, edgecolor="black", lw=0.6)
for i in range(3):
    ax.text(x[i] - w / 2, visual[i] + 1.2, f"{visual[i]:.1f}", ha="center", fontsize=9)
    ax.text(x[i] + w / 2, mira3[i] + 1.2, f"{mira3[i]:.1f}", ha="center", fontsize=9,
            fontweight="bold")
    ax.text(x[i], max(visual[i], mira3[i]) + 6.5, deltas[i], ha="center", fontsize=10,
            fontweight="bold", color=ORANGE)
ax.set_ylabel("Top-1 accuracy (%)")
ax.set_ylim(0, 108)
ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.legend(loc="upper right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_main_bar")

# ---------------- fig_lambda: sensitivity sweep (fold 3) ----------------
lam = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
top1 = [90.89, 94.57, 95.85, 94.57, 93.93, 92.01]
mapv = [94.85, 97.04, 97.80, 97.13, 96.73, 95.39]
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(lam, top1, "o-", color=ORANGE, lw=2, ms=7, label="Top-1")
ax.plot(lam, mapv, "s-", color=BLUE, lw=2, ms=7, label="MAP")
ax.axvline(0.2, color="gray", ls="--", lw=1)
ax.text(0.215, 91.6, "selected $\\lambda$=0.2", fontsize=9, color="gray")
ax.set_xlabel("Fusion weight $\\lambda$")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(90, 98.8)
ax.legend(loc="lower right", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "fig_lambda")
