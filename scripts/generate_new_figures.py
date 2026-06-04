"""
Generate missing figures for LaTeX chapter 3:
  1. class_distribution_sidebyside.png  → unfiltered vs filtered side by side
  2. trayecto_distribution_per_min.png  → events per minute per trayecto (time-weighted)

Output: TFG_LateX/figs/
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

REPO   = Path(__file__).parent.parent
FIGS   = REPO / "TFG_LateX" / "figs"
DATA   = REPO / "data" / "processed"

CLASS_NAMES = {
    0: "Horn", 1: "Siren", 2: "Pets", 3: "Physiological",
    4: "Speech", 5: "Ring Tone", 6: "Vibrating", 7: "Notifications", 8: "Cry",
}
# Consistent color palette matching existing notebooks
CLASS_COLORS = {
    "Horn": "#3B4BC8", "Siren": "#F07F0E", "Pets": "#2CA02C",
    "Physiological": "#D62728", "Speech": "#825B2F", "Ring Tone": "#E377C2",
    "Vibrating": "#7F7F7F", "Notifications": "#BCBD22", "Cry": "#17BECF",
}
CLASS_ORDER = ["Horn", "Siren", "Pets", "Physiological", "Speech",
               "Ring Tone", "Vibrating", "Notifications", "Cry"]

# ─── load data ───────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA / "predictions_geo.parquet")
df["class_name"] = df["class"].map(CLASS_NAMES)

tracks = pd.read_parquet(DATA / "tracks.parquet")

# ─── dynamic confidence threshold (mean + 1 std per class) ──────────────────
thresholds = df.groupby("class_name")["confidence"].agg(["mean", "std"])
thresholds["threshold"] = thresholds["mean"] + thresholds["std"]
df["threshold"] = df["class_name"].map(thresholds["threshold"])
df_filt = df[df["confidence"] >= df["threshold"]].copy()

# ─── 1. Side-by-side class distribution (unfiltered | filtered) ─────────────
counts_all  = df["class_name"].value_counts().reindex(CLASS_ORDER, fill_value=0)
counts_filt = df_filt["class_name"].value_counts().reindex(CLASS_ORDER, fill_value=0)
colors = [CLASS_COLORS[c] for c in CLASS_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.subplots_adjust(wspace=0.35)

for ax, counts, title in [
    (axes[0], counts_all,  f"Sin filtrar  (n = {len(df):,})"),
    (axes[1], counts_filt, f"Filtradas por confianza  (umbral = media + 1·DE,  n = {len(df_filt):,})"),
]:
    bars = ax.bar(CLASS_ORDER, counts.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("Clase", fontsize=10)
    ax.set_ylabel("Número de detecciones", fontsize=10)
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=40, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, counts.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts.values) * 0.01,
                    f"{int(val):,}", ha="center", va="bottom", fontsize=8)

out = FIGS / "class_distribution_sidebyside.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ─── 2. Time-weighted trayecto distribution (events / min) ──────────────────
# Compute duration per trayecto from GPS tracks
dur = (
    tracks.groupby("trayecto")["time"]
    .agg(lambda x: (x.max() - x.min()).total_seconds() / 60.0)
    .reset_index()
    .rename(columns={"time": "duration_min"})
)

# Event count per trayecto (unfiltered)
cnt = df.groupby("trayecto").size().reset_index(name="count")
merged = cnt.merge(dur, on="trayecto", how="inner")
merged["events_per_min"] = merged["count"] / merged["duration_min"].clip(lower=1)
merged = merged.sort_values("events_per_min", ascending=False)

# Keep top 30 to avoid overcrowding
top = merged.head(30)

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.bar(range(len(top)), top["events_per_min"], color="#3B82B8", edgecolor="white", linewidth=0.5)
ax2.set_xticks(range(len(top)))
ax2.set_xticklabels(top["trayecto"], rotation=50, ha="right", fontsize=8)
ax2.set_ylabel("Eventos / minuto", fontsize=10)
ax2.set_title("Tasa de detección por trayecto (eventos por minuto de grabación)", fontsize=11)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
ax2.grid(axis="y", alpha=0.3, linewidth=0.6)
ax2.spines[["top", "right"]].set_visible(False)

out2 = FIGS / "trayecto_distribution_per_min.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

print("All figures generated.")
