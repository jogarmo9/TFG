"""
Generate and update figures for LaTeX chapter 4 (Resultados):

  Fig 4.1: trayecto_distribution.png         — event count by trayecto, colored by source
  Fig 4.2: trayecto_distribution_per_min.png — events/min, same order & legend as 4.1
  Split:   class_distribution_before.png     — class distribution before threshold
           class_distribution_after.png      — class distribution after threshold
  Split:   source_comparison_rate.png        — detection rate by source
           source_comparison_confidence.png  — mean confidence by class & source
           source_comparison_classes.png     — % class distribution by source

Output: Detección_de_Áreas_Acústicamente_Peligrosas_mediante_Deep_Learning/figs/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path(__file__).parent.parent
FIGS = (REPO / "Detección_de_Áreas_Acústicamente_Peligrosas_mediante_Deep_Learning" / "figs")
if not FIGS.exists():
    # fallback: look for any dir with 'figs' under a Spanish-name folder
    candidates = list(REPO.glob("*/figs"))
    if candidates:
        FIGS = candidates[0]
    else:
        FIGS = REPO / "outputs"
DATA = REPO / "data" / "processed"

CLASS_NAMES = {
    0: "Horn", 1: "Siren", 2: "Pets", 3: "Physiological",
    4: "Speech", 5: "Ring Tone", 6: "Vibrating", 7: "Notifications", 8: "Cry",
}
CLASS_ORDER = ["Horn", "Siren", "Pets", "Physiological", "Speech",
               "Ring Tone", "Vibrating", "Notifications", "Cry"]

_pal = sns.color_palette("tab10", 9)
CLASS_PALETTE = {CLASS_NAMES[i]: _pal[i] for i in range(9)}

SRC_COLORS = {"mic": "steelblue", "mobile": "coral"}

# ─── load data ───────────────────────────────────────────────────────────────
df = pd.read_parquet(DATA / "predictions_geo.parquet")
df["class_name"] = df["class"].map(CLASS_NAMES)

tracks = pd.read_parquet(DATA / "tracks.parquet")

# dynamic confidence threshold (mean + 1 std per class)
thr = df.groupby("class_name")["confidence"].transform(lambda x: x.mean() + x.std())
df_filt = df[df["confidence"] >= thr].copy()

# source map per trayecto
src_map = df.groupby("trayecto")["source"].first()

# ─── duration per trayecto (GPS) ────────────────────────────────────────────
dur_gps = (
    tracks.groupby("trayecto")["time"]
    .agg(lambda x: (x.max() - x.min()).total_seconds() / 60.0)
    .rename("duration_min")
)

# ─── 1. trayecto_distribution.png  (count, sorted desc) ─────────────────────
cnt = df.groupby("trayecto").size().rename("count").reset_index()
cnt["source"] = cnt["trayecto"].map(src_map)
cnt = cnt.sort_values("count", ascending=False).reset_index(drop=True)

TRAYECTO_ORDER = cnt["trayecto"].tolist()  # shared order for fig 4.1 & 4.2

fig, ax = plt.subplots(figsize=(16, 6))
bar_colors = [SRC_COLORS.get(str(s), "gray") for s in cnt["source"]]
ax.bar(range(len(cnt)), cnt["count"], color=bar_colors, edgecolor="white", linewidth=0.4)
ax.set_xticks(range(len(cnt)))
ax.set_xticklabels(cnt["trayecto"], rotation=50, ha="right", fontsize=7.5)
ax.set_ylabel("Número de eventos detectados", fontsize=11)
ax.set_title("Distribución de eventos detectados por trayecto y fuente", fontsize=12)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(axis="y", alpha=0.3, linewidth=0.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[
    mpatches.Patch(color=SRC_COLORS["mic"],    label="Prototipo (mic)"),
    mpatches.Patch(color=SRC_COLORS["mobile"], label="Móvil"),
], fontsize=10)
plt.tight_layout()
out = FIGS / "trayecto_distribution.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ─── 2. trayecto_distribution_per_min.png  (events/min, same order as fig 4.1) ─
merged = cnt.merge(dur_gps.reset_index(), on="trayecto", how="inner")
merged["events_per_min"] = merged["count"] / merged["duration_min"].clip(lower=1)

# reindex to same order as fig 4.1
merged = merged.set_index("trayecto").reindex(TRAYECTO_ORDER).reset_index()
merged["source"] = merged["trayecto"].map(src_map)

fig2, ax2 = plt.subplots(figsize=(16, 6))
bar_colors2 = [SRC_COLORS.get(str(s), "gray") for s in merged["source"]]
ax2.bar(range(len(merged)), merged["events_per_min"].fillna(0),
        color=bar_colors2, edgecolor="white", linewidth=0.4)
ax2.set_xticks(range(len(merged)))
ax2.set_xticklabels(merged["trayecto"], rotation=50, ha="right", fontsize=7.5)
ax2.set_ylabel("Eventos / minuto de grabación", fontsize=11)
ax2.set_title("Tasa de detección por trayecto (eventos por minuto de grabación)", fontsize=12)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
ax2.grid(axis="y", alpha=0.3, linewidth=0.6)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(handles=[
    mpatches.Patch(color=SRC_COLORS["mic"],    label="Prototipo (mic)"),
    mpatches.Patch(color=SRC_COLORS["mobile"], label="Móvil"),
], fontsize=10)
plt.tight_layout()
out2 = FIGS / "trayecto_distribution_per_min.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ─── 3a. class_distribution_before.png ───────────────────────────────────────
counts_all = df["class_name"].value_counts().reindex(CLASS_ORDER, fill_value=0)
colors_bar = [CLASS_PALETTE[c] for c in CLASS_ORDER]

fig3, ax3 = plt.subplots(figsize=(10, 5))
bars = ax3.bar(CLASS_ORDER, counts_all.values, color=colors_bar,
               edgecolor="white", linewidth=0.5)
ax3.set_title(f"Distribución de clases sin filtrar  (n = {len(df):,})", fontsize=12, pad=8)
ax3.set_xlabel("Clase", fontsize=11)
ax3.set_ylabel("Número de detecciones", fontsize=11)
ax3.set_xticks(range(len(CLASS_ORDER)))
ax3.set_xticklabels(CLASS_ORDER, rotation=40, ha="right", fontsize=9)
ax3.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax3.grid(axis="y", alpha=0.3, linewidth=0.6)
ax3.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars, counts_all.values):
    if val > 0:
        ax3.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + counts_all.values.max() * 0.012,
                 f"{int(val):,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
out3 = FIGS / "class_distribution_before.png"
fig3.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")

# ─── 3b. class_distribution_after.png ────────────────────────────────────────
counts_filt = df_filt["class_name"].value_counts().reindex(CLASS_ORDER, fill_value=0)

fig4, ax4 = plt.subplots(figsize=(10, 5))
bars4 = ax4.bar(CLASS_ORDER, counts_filt.values, color=colors_bar,
                edgecolor="white", linewidth=0.5)
ax4.set_title(f"Distribución de clases tras filtrado  (umbral = $\\mu_c + 1\\cdot\\sigma_c$,  n = {len(df_filt):,})",
              fontsize=12, pad=8)
ax4.set_xlabel("Clase", fontsize=11)
ax4.set_ylabel("Número de detecciones", fontsize=11)
ax4.set_xticks(range(len(CLASS_ORDER)))
ax4.set_xticklabels(CLASS_ORDER, rotation=40, ha="right", fontsize=9)
ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax4.grid(axis="y", alpha=0.3, linewidth=0.6)
ax4.spines[["top", "right"]].set_visible(False)
for bar, val in zip(bars4, counts_filt.values):
    if val > 0:
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + counts_filt.values.max() * 0.012,
                 f"{int(val):,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
out4 = FIGS / "class_distribution_after.png"
fig4.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out4}")

# ─── 4. Source comparison figures (3 separate) ───────────────────────────────
# 4a. source_comparison_rate.png — detection rate (events/min) by source
dur_src = (
    df.groupby(["source", "trayecto"])
    .apply(lambda g: (g["t_start"].max() - g["t_start"].min()).total_seconds() / 60.0)
    .groupby("source").sum()
)
n_src = df.groupby("source").size()
rate_src = (n_src / dur_src).rename("events_per_min").reset_index()
rate_src.columns = ["source", "events_per_min"]
rate_src["source_label"] = rate_src["source"].map({"mic": "Prototipo (mic)", "mobile": "Móvil"})

fig5, ax5 = plt.subplots(figsize=(6, 5))
bar_s = ax5.bar(rate_src["source_label"], rate_src["events_per_min"],
                color=[SRC_COLORS[s] for s in rate_src["source"]],
                edgecolor="white", linewidth=0.5, width=0.5)
for bar, val in zip(bar_s, rate_src["events_per_min"]):
    ax5.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + rate_src["events_per_min"].max() * 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontsize=11)
ax5.set_ylabel("Eventos / minuto", fontsize=11)
ax5.set_title("Tasa de detección por fuente", fontsize=12)
ax5.grid(axis="y", alpha=0.3, linewidth=0.6)
ax5.spines[["top", "right"]].set_visible(False)
ax5.set_ylim(0, rate_src["events_per_min"].max() * 1.2)
plt.tight_layout()
out5 = FIGS / "source_comparison_rate.png"
fig5.savefig(out5, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out5}")

# 4b. source_comparison_confidence.png — mean confidence by class & source
conf_mean = (
    df_filt.groupby(["source", "class_name"])["confidence"]
    .mean().reset_index()
)
conf_mean["source_label"] = conf_mean["source"].map({"mic": "Prototipo (mic)", "mobile": "Móvil"})

fig6, ax6 = plt.subplots(figsize=(11, 5))
x = np.arange(len(CLASS_ORDER))
width = 0.35
for i, (src, label) in enumerate([("mic", "Prototipo (mic)"), ("mobile", "Móvil")]):
    vals = conf_mean[conf_mean["source"] == src].set_index("class_name")["confidence"]
    vals = vals.reindex(CLASS_ORDER, fill_value=0)
    ax6.bar(x + (i - 0.5) * width, vals.values, width=width,
            color=SRC_COLORS[src], edgecolor="white", linewidth=0.4, label=label, alpha=0.85)
ax6.set_xticks(x)
ax6.set_xticklabels(CLASS_ORDER, rotation=40, ha="right", fontsize=9)
ax6.set_ylabel("Confianza media", fontsize=11)
ax6.set_title("Confianza media por clase tras filtrado dinámico", fontsize=12)
ax6.set_ylim(0, 0.9)
ax6.grid(axis="y", alpha=0.3, linewidth=0.6)
ax6.spines[["top", "right"]].set_visible(False)
ax6.legend(fontsize=10)
plt.tight_layout()
out6 = FIGS / "source_comparison_confidence.png"
fig6.savefig(out6, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out6}")

# 4c. source_comparison_classes.png — % class distribution by source
mix = df_filt.groupby(["source", "class_name"]).size().rename("n").reset_index()
mix["pct"] = mix.groupby("source")["n"].transform(lambda s: 100 * s / s.sum())
mix["source_label"] = mix["source"].map({"mic": "Prototipo (mic)", "mobile": "Móvil"})

fig7, ax7 = plt.subplots(figsize=(11, 5))
x = np.arange(len(CLASS_ORDER))
for i, (src, label) in enumerate([("mic", "Prototipo (mic)"), ("mobile", "Móvil")]):
    sub = mix[mix["source"] == src].set_index("class_name")["pct"]
    sub = sub.reindex(CLASS_ORDER, fill_value=0)
    ax7.bar(x + (i - 0.5) * width, sub.values, width=width,
            color=SRC_COLORS[src], edgecolor="white", linewidth=0.4, label=label, alpha=0.85)
ax7.set_xticks(x)
ax7.set_xticklabels(CLASS_ORDER, rotation=40, ha="right", fontsize=9)
ax7.set_ylabel("% detecciones (por fuente)", fontsize=11)
ax7.set_title("Distribución porcentual de clases por fuente (tras filtrado)", fontsize=12)
ax7.grid(axis="y", alpha=0.3, linewidth=0.6)
ax7.spines[["top", "right"]].set_visible(False)
ax7.legend(fontsize=10)
plt.tight_layout()
out7 = FIGS / "source_comparison_classes.png"
fig7.savefig(out7, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out7}")

print("\nAll figures generated.")
