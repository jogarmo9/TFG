"""
Preview all maps at low resolution before final export.
Shows a comparison grid so you can judge zoom/center before committing.

Usage:
    python scripts/preview_maps.py

Then adjust zoom/center in export_maps.py and re-run this to iterate.
Final export (full resolution): python scripts/export_maps.py
"""

import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from html2image import Html2Image

REPO    = Path(__file__).parent.parent
OUTPUTS = REPO / "outputs"
PREVIEW = REPO / "outputs" / "_preview"
PREVIEW.mkdir(exist_ok=True)

# ── Import map config from export_maps.py ────────────────────────────────────
# (html_name, png_name, w, h, center, zoom)
import importlib.util, sys
spec = importlib.util.spec_from_file_location("export_maps", Path(__file__).parent / "export_maps.py")
em   = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)
MAPS = em.MAPS

# ── Export at preview resolution ─────────────────────────────────────────────
PREVIEW_W, PREVIEW_H = 900, 540   # ~half of final 2000×1200
hti = Html2Image(
    output_path=str(PREVIEW),
    custom_flags=[
        "--virtual-time-budget=8000",
        "--hide-scrollbars",
        "--default-background-color=FFFFFFFF",
    ],
)

tmp_files = []
preview_paths = []

for html_name, png_name, _w, _h, center, zoom in MAPS:
    html_path = OUTPUTS / html_name
    if not html_path.exists():
        print(f"SKIP (not found): {html_name}")
        continue

    src = html_path
    tmp = None
    if center is not None or zoom is not None:
        tmp = em.patch_map(html_path, center=center, zoom=zoom)
        src = tmp
        tmp_files.append(tmp)

    out_name = f"preview_{png_name}"
    label = f"zoom={zoom or 'orig'}  center={center or 'orig'}"
    print(f"  {html_name} ({label}) -> {out_name}")
    hti.screenshot(html_file=str(src), save_as=out_name, size=(PREVIEW_W, PREVIEW_H))
    preview_paths.append((PREVIEW / out_name, html_name, zoom, center))

for tmp in tmp_files:
    tmp.unlink(missing_ok=True)

# ── Show comparison grid ──────────────────────────────────────────────────────
n = len(preview_paths)
cols = 2
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4.5))
axes = axes.flatten() if n > 1 else [axes]

for ax, (img_path, html_name, zoom, center) in zip(axes, preview_paths):
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
    ax.set_title(
        f"{html_name}\nzoom={zoom or 'orig'}  center={center or 'orig'}",
        fontsize=8, pad=4
    )
    ax.axis("off")

for ax in axes[n:]:
    ax.axis("off")

plt.suptitle(
    "MAP PREVIEW — adjust zoom/center in scripts/export_maps.py then re-run",
    fontsize=10, fontweight="bold", y=1.01
)
plt.tight_layout()
out_grid = PREVIEW / "preview_grid.png"
fig.savefig(out_grid, dpi=120, bbox_inches="tight")
plt.show()
print(f"\nGrid saved: {out_grid}")
print("Edit MAPS in scripts/export_maps.py, then run scripts/export_maps.py for final PNGs."
      .encode("ascii", "replace").decode())
