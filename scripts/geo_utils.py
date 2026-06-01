"""
geo_utils.py — utilidades de análisis espacio-acústico para el TFG.

Reúne helpers reutilizados por los notebooks de análisis (02, 03, 04):
- Constantes de clases acústicas y su relevancia vial.
- Cálculo de velocidad/aceleración por trackpoint (haversine).
- Join temporal velocidad ↔ detección (mismo patrón que prepare_mic.run_join).
- Intervalo de confianza de Wilson (sin dependencia de statsmodels).
- Agregación espacial en grilla (danger score sin osmnx).

Clave de datos:
- Dataset principal: data/processed/predictions_geo.parquet
  cols: microfono_id, t_start, t_end, class, confidence, date, duration_s,
        source ('mic'|'mobile'), session_id, source_file, lat, lon, trayecto
- Tracks GPS: data/processed/tracks.parquet
  cols: date, trayecto, source, lat, lon, ele, time
- La clave de agrupación es `trayecto` (+ `source`), NO `session_id`
  (session_id es NaN para la fuente mic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Clases acústicas
# --------------------------------------------------------------------------- #
CLASS_NAMES = {
    0: "Horn", 1: "Siren", 2: "Pets", 3: "Physiological", 4: "Speech",
    5: "Ring Tone", 6: "Vibrating", 7: "Notifications", 8: "Cry",
}

# Relevancia para el dominio de conducción urbana.
ROAD_CLASSES = {0, 1}          # Horn, Siren — peligro vial directo
CONTEXT_CLASSES = {4}          # Speech — contexto (radio/charla), NO peligro
OUT_OF_DOMAIN = {2, 3, 5, 6, 7, 8}  # clases del dataset YOLO original, ruido en coche

# Pesos de severidad para el danger score (solo clases viales).
SEVERITY = {0: 1.0, 1: 2.0}    # Siren pesa más que Horn


def class_name(cid) -> str:
    """Nombre legible de un class_id (acepta int o float)."""
    return CLASS_NAMES.get(int(cid), f"class_{cid}")


# --------------------------------------------------------------------------- #
# Geometría / velocidad
# --------------------------------------------------------------------------- #
def haversine_m(lat1, lon1, lat2, lon2):
    """Distancia en metros entre pares de coordenadas (vectorizado)."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_speed(tracks: pd.DataFrame, group_keys=("trayecto", "source")) -> pd.DataFrame:
    """
    Añade speed_ms y accel_ms2 a un DataFrame de tracks GPS.

    Deriva velocidad de puntos GPS consecutivos (distancia haversine / Δt) y
    aceleración como derivada de la velocidad. Agrupa por trayecto+source para
    no cruzar trayectos distintos.
    """
    group_keys = list(group_keys)
    t = tracks.sort_values(group_keys + ["time"]).copy()
    g = t.groupby(group_keys, group_keys=False)
    lat_prev = g["lat"].shift()
    lon_prev = g["lon"].shift()
    dist = haversine_m(lat_prev, lon_prev, t["lat"], t["lon"])
    dt = g["time"].diff().dt.total_seconds()
    t["speed_ms"] = (dist / dt).where(dt > 0)
    t["speed_kmh"] = t["speed_ms"] * 3.6
    t["accel_ms2"] = g["speed_ms"].diff() / dt
    return t


def join_speed_to_detections(det: pd.DataFrame, tracks_speed: pd.DataFrame,
                             group_keys=("trayecto", "source"),
                             tol_s: int = 30) -> pd.DataFrame:
    """
    Une velocidad/aceleración a cada detección por cercanía temporal.

    Mismo patrón que prepare_mic.run_join: merge_asof 'nearest' con tolerancia
    por grupo (trayecto+source). Detección sin trackpoint dentro de la
    tolerancia → speed NaN.
    """
    group_keys = list(group_keys)
    speed_cols = ["speed_ms", "speed_kmh", "accel_ms2"]
    out = []
    det_sorted = det.sort_values("t_start")
    tr_sorted = tracks_speed.sort_values("time")
    for key, dsub in det_sorted.groupby(group_keys):
        mask = np.ones(len(tr_sorted), dtype=bool)
        for k, v in zip(group_keys, key if isinstance(key, tuple) else (key,)):
            mask &= (tr_sorted[k] == v).to_numpy()
        tsub = tr_sorted.loc[mask, ["time"] + speed_cols]
        if tsub.empty:
            out.append(dsub.assign(**{c: np.nan for c in speed_cols}))
            continue
        merged = pd.merge_asof(
            dsub, tsub, left_on="t_start", right_on="time",
            direction="nearest", tolerance=pd.Timedelta(seconds=tol_s),
        ).drop(columns="time")
        out.append(merged)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# Estadística — Wilson CI (sin statsmodels)
# --------------------------------------------------------------------------- #
def wilson_ci(success: int, n: int, z: float = 1.96):
    """
    Intervalo de confianza de Wilson para una proporción.

    Devuelve (p_hat, lo, hi). Robusto con n pequeño (validación parcial).
    n == 0 → (nan, 0, 1).
    """
    if n == 0:
        return (np.nan, 0.0, 1.0)
    p = success / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (p, max(0.0, center - half), min(1.0, center + half))


def class_precision(labels: pd.DataFrame, class_col="class", tp_col="is_tp"):
    """
    Precisión por clase con IC de Wilson a partir de etiquetas manuales.
    labels: una fila por detección escuchada, con tp_col ∈ {0,1}.
    """
    rows = []
    for cid, sub in labels.groupby(class_col):
        succ, n = int(sub[tp_col].sum()), len(sub)
        p, lo, hi = wilson_ci(succ, n)
        rows.append({"class": cid, "name": class_name(cid), "n": n,
                     "tp": succ, "precision": p, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Agregación espacial en grilla (danger score, sin osmnx)
# --------------------------------------------------------------------------- #
def grid_aggregate(df: pd.DataFrame, value_col=None, cell_deg: float = 0.0005,
                   lat_col="lat", lon_col="lon"):
    """
    Agrega puntos en celdas de grilla regular.

    cell_deg=0.0005 ≈ 55 m lat × 43 m lon (lat ~39.4°). Devuelve DataFrame con
    centro de celda (lat, lon), recuento y suma de value_col (si se da).
    """
    d = df.copy()
    d["_gi"] = np.floor(d[lat_col] / cell_deg).astype(int)
    d["_gj"] = np.floor(d[lon_col] / cell_deg).astype(int)
    agg = {lat_col: "count"}
    if value_col:
        agg = {value_col: "sum"}
    g = d.groupby(["_gi", "_gj"]).agg(count=(lat_col, "size"),
                                      value=(value_col, "sum") if value_col else (lat_col, "size"))
    g = g.reset_index()
    g["cell_lat"] = (g["_gi"] + 0.5) * cell_deg
    g["cell_lon"] = (g["_gj"] + 0.5) * cell_deg
    return g


# --------------------------------------------------------------------------- #
# Carga de datos
# --------------------------------------------------------------------------- #
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parent.parent  # raíz del repo (independiente del cwd)


def _resolve(path):
    """Resuelve rutas relativas contra la raíz del repo (funciona desde notebooks/ o raíz)."""
    p = _Path(path)
    return p if p.is_absolute() or p.exists() else (_REPO_ROOT / p)


def load_geo(path="data/processed/predictions_geo.parquet") -> pd.DataFrame:
    return pd.read_parquet(_resolve(path))


def load_tracks(path="data/processed/tracks.parquet") -> pd.DataFrame:
    return pd.read_parquet(_resolve(path))


# Trayectos a excluir (GPS defectuoso / sin audio) — ver TFG_CONTEXT §12.
BAD_TRAYECTOS = {
    "MASANASA-SILLA_2_Revisar_GPS",
    "11-03-2026_skip1", "11-03-2026_skip2",
}
