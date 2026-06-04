"""
Export key Folium HTML maps to PNG for LaTeX inclusion.
Requires: html2image (pip install html2image)
Requires: Chrome or Edge installed on the system.

Usage:
    python scripts/export_maps.py
Output PNGs go to TFG_LateX/figs/
"""

import re
import tempfile
from pathlib import Path

from html2image import Html2Image

REPO   = Path(__file__).parent.parent
OUTPUTS = REPO / "outputs"
FIGS   = REPO / "TFG_LateX" / "figs"
FIGS.mkdir(exist_ok=True)

# (html_name, png_name, width, height, center_override, zoom_override)
# center_override: None = keep original, [lat, lon] = set center
# zoom_override:   None = keep original, int = force this Leaflet zoom level
MAPS = [
    ("map_heatmap_conf070.html", "map_heatmap_conf070.png", 2000, 1200, [39.452, -0.379], 11.3),
    ("map_density_conf070.html", "map_density_conf070.png", 2000, 1200, [39.452, -0.379], 11.0),
    ("map_consistency.html",     "map_consistency.png",     2000, 1200, [39.452, -0.379], 13.0),
    ("map_etse_comparison.html", "map_etse_comparison.png", 2000, 1200, [39.469, -0.407], 12.0),
    ("map_speed_all.html",       "map_speed_all.png",       2000, 1200, [39.450, -0.385], 11.0),
]


def patch_map(html_path: Path, center=None, zoom: int = None) -> Path:
    """Return path to a temp HTML file with Leaflet center/zoom patched."""
    html = html_path.read_text(encoding="utf-8")

    if zoom is not None:
        # Folium puts zoom in the L.map options object: "zoom": <number>
        html = re.sub(r'("zoom"\s*:\s*)\d+(?:\.\d+)?', rf'\g<1>{zoom}', html)

    if center is not None:
        lat, lon = center
        # Folium: center: [lat, lon]
        html = re.sub(
            r'(center\s*:\s*\[)\s*[\d.\-]+\s*,\s*[\d.\-]+\s*(\])',
            rf'\g<1>{lat}, {lon}\2',
            html,
        )

    tmp = tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(patched := html)
    tmp.close()
    return Path(tmp.name)


# --virtual-time-budget lets headless Chrome run JS/load tiles before snapping.
# Without it the screenshot often fires on a blank or half-rendered map.
hti = Html2Image(
    output_path=str(FIGS),
    custom_flags=[
        "--virtual-time-budget=8000",
        "--hide-scrollbars",
        "--default-background-color=FFFFFFFF",
    ],
)

for html_name, png_name, w, h, center, zoom in MAPS:
    html_path = OUTPUTS / html_name
    if not html_path.exists():
        print(f"SKIP (not found): {html_name}")
        continue

    src = html_path
    tmp = None
    if center is not None or zoom is not None:
        tmp = patch_map(html_path, center=center, zoom=zoom)
        src = tmp

    label = f"zoom={zoom or 'orig'}, center={center or 'orig'}"
    print(f"Exporting {html_name} ({label}) -> {png_name} ...")
    hti.screenshot(html_file=str(src), save_as=png_name, size=(w, h))
    print(f"  -> TFG_LateX/figs/{png_name}")

    if tmp is not None:
        tmp.unlink(missing_ok=True)

print("Done.")
