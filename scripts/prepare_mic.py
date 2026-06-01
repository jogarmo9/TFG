"""
prepare_mic.py
==============
ETL para datos de micrófonos fijos (ruta diaria).

Bloque A: predicciones_clean.csv -> predictions_mic.parquet
          Incluye NMS cruzado entre M1 y M2 (elimina duplicados de mismo evento).
Bloque B: data/raw/**/*.gpx      -> tracks_mic.parquet
          Cada GPX = un trayecto con nombre ORIGEN-DESTINO_N.
Bloque C: join espacio-temporal  -> predictions_geo.parquet + tracks.parquet

Uso:
  python scripts/prepare_mic.py
  python scripts/prepare_mic.py --reprocess-all      # fuerza regenerar todo
  python scripts/prepare_mic.py --skip-join          # solo A+B, sin join
  python scripts/prepare_mic.py --no-cross-mic-nms   # desactiva NMS M1/M2
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import gpxpy
import numpy as np
import pandas as pd

ROOT          = Path(__file__).parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR       = ROOT / "data" / "raw"
CLEAN_CSV     = PROCESSED_DIR / "predicciones_clean.csv"
CLEAN_CSV_RAW = PROCESSED_DIR / "predicciones_clean_raw.csv"

TIME_CORRECTIONS = {"23-03-2026": -1}   # día -> delta en horas

THRESHOLD_MIC = 4    # segundos: margen join GPS ↔ predicción mic
THRESHOLD_MOB = 10   # segundos: margen join GPS ↔ predicción móvil

# ── Nombres de trayecto para datos Mic ───────────────────────────────────────
# ROUTE_DEFAULT: (nombre_gpx1_mañana, nombre_gpx2_tarde) por defecto.
# ROUTE_NAMES:   sobreescritura por fecha "DD-MM-YYYY" si algún día es distinto.
ROUTE_DEFAULT: tuple[str, str] = ("PAIPORTA-ETSE", "ETSE-PAIPORTA")
ROUTE_NAMES:   dict[str, tuple[str, str]] = {
    # "DD-MM-YYYY": ("ORIGEN-DESTINO", "DESTINO-ORIGEN"),
}
# Fechas que tienen GPX pero se excluyen del contador de rutas (fallo GPS u otra razón).
# Sus trackpoints se guardan en tracks_mic con nombre "<fecha>_skip" para no perder el GPS.
SKIP_DATES: set[str] = {
    "11-03-2026",  # fallo de sincronía GPS — predictions fuera del rango del track
}

# ── Parámetros NMS cruzado M1/M2 ─────────────────────────────────────────────
CROSS_MIC_TOL_S   = 1.0   # ventana temporal máxima (s) para buscar duplicados
CROSS_MIC_IOU_THR = 0.3   # IoU temporal mínimo para considerar duplicado


# ──────────────────────────────────────────────────────────────
# NMS CRUZADO M1 / M2
# ──────────────────────────────────────────────────────────────

def cross_mic_nms(df: pd.DataFrame,
                  tol_s: float = CROSS_MIC_TOL_S,
                  iou_thresh: float = CROSS_MIC_IOU_THR) -> pd.DataFrame:
    """
    Elimina predicciones duplicadas entre M1 y M2.

    M1 y M2 graban el mismo audio simultáneamente, por lo que pueden detectar
    el mismo evento físico. Dos detecciones (una de M1 y una de M2) se
    consideran duplicadas si:
      - Misma clase (class) y misma fecha (date)
      - |t_start_M1 - t_start_M2| ≤ tol_s
      - IoU temporal ≥ iou_thresh

    Se conserva la detección de mayor confianza (en empate, se prioriza M1).
    """
    if df["microfono_id"].nunique() < 2:
        return df

    drop_idx: set = set()

    for (date, cls), grp in df.groupby(["date", "class"]):
        m1 = grp[grp["microfono_id"] == 1]
        m2 = grp[grp["microfono_id"] == 2]
        if m1.empty or m2.empty:
            continue

        tol = pd.Timedelta(seconds=tol_s)

        for i1, r1 in m1.iterrows():
            if i1 in drop_idx:
                continue
            nearby_m2 = m2[abs(m2["t_start"] - r1["t_start"]) <= tol]
            for i2, r2 in nearby_m2.iterrows():
                if i2 in drop_idx:
                    continue
                # IoU temporal
                inter = max(
                    pd.Timedelta(0),
                    min(r1["t_end"], r2["t_end"]) - max(r1["t_start"], r2["t_start"])
                ).total_seconds()
                if inter <= 0:
                    continue
                dur1 = (r1["t_end"] - r1["t_start"]).total_seconds()
                dur2 = (r2["t_end"] - r2["t_start"]).total_seconds()
                iou = inter / (dur1 + dur2 - inter + 1e-9)
                if iou >= iou_thresh:
                    if r1["confidence"] >= r2["confidence"]:
                        drop_idx.add(i2)
                    else:
                        drop_idx.add(i1)
                        break  # r1 eliminado, pasar a siguiente r1

    n_before = len(df)
    result = df.drop(index=drop_idx)
    n_drop = n_before - len(result)
    if n_drop:
        print(f"  [cross-mic NMS] {n_before} -> {len(result)} "
              f"({n_drop} duplicados M1/M2 eliminados)")
    else:
        print(f"  [cross-mic NMS] Sin duplicados M1/M2 detectados.")
    return result


# ──────────────────────────────────────────────────────────────
# BLOQUE A — PREDICCIONES MIC
# ──────────────────────────────────────────────────────────────

def process_predictions(raw: bool = False,
                        apply_cross_nms: bool = True) -> pd.DataFrame:
    label    = "_raw" if raw else ""
    csv_path = CLEAN_CSV_RAW if raw else CLEAN_CSV
    out_path = PROCESSED_DIR / f"predictions_mic{label}.parquet"
    tag      = "[A-raw]" if raw else "[A]"
    print(f"{tag} Cargando predicciones {'raw ' if raw else ''}limpias...")

    if not csv_path.exists():
        if raw:
            print(f"  [WARN] No encontrado: {csv_path.name} — omitiendo raw")
            return pd.DataFrame()
        sys.exit(f"[ERROR] No encontrado: {csv_path}\n       Ejecutar: python scripts/infer_clean.py")

    df = pd.read_csv(csv_path)

    df = df.rename(columns={
        "mic_id":           "microfono_id",
        "timestamp_onset":  "t_start",
        "timestamp_offset": "t_end",
        "class_id":         "class",
    })

    df["t_start"] = pd.to_datetime(df["t_start"], format="mixed")
    df["t_end"]   = pd.to_datetime(df["t_end"],   format="mixed")

    # date antes de convertir zona horaria
    df["date"] = df["t_start"].dt.strftime("%d-%m-%Y")

    # correcciones horarias puntuales
    for day, hours in TIME_CORRECTIONS.items():
        mask = df["date"] == day
        if mask.any():
            print(f"  Corrección {hours:+d}h -> {day}")
            df.loc[mask, "t_start"] += pd.Timedelta(hours=hours)
            df.loc[mask, "t_end"]   += pd.Timedelta(hours=hours)

    df["t_start"] = (
        df["t_start"]
        .dt.tz_localize("Europe/Madrid", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    df["t_end"] = (
        df["t_end"]
        .dt.tz_localize("Europe/Madrid", ambiguous="infer", nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    df["duration_s"] = (df["t_end"] - df["t_start"]).dt.total_seconds()
    df["class"]      = pd.to_numeric(df["class"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df = df.dropna(subset=["class", "confidence", "t_start", "t_end"])
    df["class"] = df["class"].astype(int)
    df["source"] = "mic"

    # NMS cruzado M1/M2 (solo para predicciones de producción, no raw)
    if apply_cross_nms and not raw:
        df = cross_mic_nms(df)

    df.to_parquet(out_path, index=False)
    print(f"  [OK] {len(df)} predicciones, {df['date'].nunique()} días -> {out_path.name}")
    return df


# ──────────────────────────────────────────────────────────────
# BLOQUE B — TRACKS GPS MIC
# ──────────────────────────────────────────────────────────────

def process_tracks() -> pd.DataFrame:
    """
    Parsea todos los GPX de data/raw/. Cada archivo GPX recibe un trayecto
    con nombre ORIGEN-DESTINO_N (N = número de ocurrencia de esa ruta).

    Por defecto se usa ROUTE_DEFAULT = ("PAIPORTA-ETSE", "ETSE-PAIPORTA"):
      - Primer GPX de cada día (hora más temprana) -> PAIPORTA-ETSE_N
      - Segundo GPX del día                        -> ETSE-PAIPORTA_N

    Solo se incrementa el contador para fechas que:
      1. No están en SKIP_DATES, y
      2. Tienen al menos una predicción en predicciones_clean.csv.
    Las fechas excluidas se guardan con nombre "<fecha>_skip" o "<fecha>_nogps".
    """
    print("[B] Cargando trazas GPS y asignando trayectos...")

    gpx_files = sorted(RAW_DIR.rglob("*.gpx"))
    if not gpx_files:
        print("  [WARN] No se encontraron .gpx en data/raw/")
        return pd.DataFrame()

    # Fechas con audio en predicciones_clean.csv
    if CLEAN_CSV.exists():
        _csv = pd.read_csv(CLEAN_CSV, usecols=["timestamp_onset"])
        audio_dates: set[str] = set(
            pd.to_datetime(_csv["timestamp_onset"], format="mixed")
            .dt.strftime("%d-%m-%Y")
        )
    else:
        audio_dates = None  # sin CSV: contar todas

    # Agrupar por fecha (carpeta padre del GPX)
    date_gpx: dict[str, list[Path]] = defaultdict(list)
    for f in gpx_files:
        date_gpx[f.parent.name].append(f)

    # Ordenar fechas cronológicamente
    sorted_dates = sorted(
        date_gpx.keys(),
        key=lambda d: pd.to_datetime(d, format="%d-%m-%Y")
    )

    route_counters: dict[str, int] = {}
    rows = []

    for day in sorted_dates:
        day_files = sorted(date_gpx[day])   # orden alfabético ≡ orden temporal (nombre lleva hora)

        # ¿Esta fecha cuenta para la numeración de rutas?
        skip = day in SKIP_DATES
        no_audio = (audio_dates is not None) and (day not in audio_dates)

        route_tuple = ROUTE_NAMES.get(day, ROUTE_DEFAULT)

        for idx, f in enumerate(day_files):
            if skip or no_audio:
                # Trackpoints guardados con nombre neutro (no se pierde el GPS)
                suffix = "skip" if skip else "noaudio"
                trayecto_id = f"{day}_{suffix}{idx + 1}"
            else:
                if idx < len(route_tuple):
                    route_name = route_tuple[idx]
                else:
                    route_name = f"TRAYECTO-{idx + 1}"
                route_counters[route_name] = route_counters.get(route_name, 0) + 1
                trayecto_id = f"{route_name}_{route_counters[route_name]}"

            with open(f, encoding="utf-8") as fh:
                gpx = gpxpy.parse(fh)

            n_pts = 0
            for track in gpx.tracks:
                for seg in track.segments:
                    for pt in seg.points:
                        rows.append({
                            "date":     day,
                            "trayecto": trayecto_id,
                            "source":   "mic",
                            "lat":      pt.latitude,
                            "lon":      pt.longitude,
                            "ele":      pt.elevation,
                            "time":     pt.time,
                        })
                        n_pts += 1
            tag = "  [SKIP]" if (skip or no_audio) else " "
            print(f"{tag} {trayecto_id}  ← {f.name}  ({n_pts} pts)")

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    out = PROCESSED_DIR / "tracks_mic.parquet"
    df.to_parquet(out, index=False)
    named = df[~df["trayecto"].str.contains("_skip|_noaudio", regex=True)]["trayecto"].nunique()
    print(f"  [OK] {len(df)} trackpoints, {named} trayectos nombrados -> {out.name}")
    return df


# ──────────────────────────────────────────────────────────────
# BLOQUE C — JOIN UNIFICADO
# ──────────────────────────────────────────────────────────────

def run_join():
    print("[C] Join espacio-temporal...")

    pred_mic  = pd.read_parquet(PROCESSED_DIR / "predictions_mic.parquet")
    pred_mob_path = PROCESSED_DIR / "predictions_mobile.parquet"
    if pred_mob_path.exists():
        pred_mob = pd.read_parquet(pred_mob_path)
        pred_mob = pred_mob.rename(columns={
            "timestamp_onset":  "t_start",
            "timestamp_offset": "t_end",
            "class_id":         "class",
            "mic_id":           "microfono_id",
        })
        pred_mob["t_start"]    = pd.to_datetime(pred_mob["t_start"], utc=True)
        pred_mob["t_end"]      = pd.to_datetime(pred_mob["t_end"],   utc=True)
        pred_mob["duration_s"] = (pred_mob["t_end"] - pred_mob["t_start"]).dt.total_seconds()
        pred_mob["class"]      = pred_mob["class"].astype(int)
        pred_mob["date"]       = pred_mob["t_start"].dt.tz_convert("Europe/Madrid").dt.strftime("%d-%m-%Y")
        pred_combined = pd.concat([pred_mic, pred_mob], ignore_index=True)
        print(f"  Predicciones: {len(pred_mic)} mic + {len(pred_mob)} móvil = {len(pred_combined)} total")
    else:
        pred_combined = pred_mic
        print(f"  Predicciones: {len(pred_mic)} mic (sin datos móvil)")

    # ── Tracks: mic (con trayecto) + mobile (trayecto = session_id) ──
    trk_mic  = pd.read_parquet(PROCESSED_DIR / "tracks_mic.parquet")
    trk_mob_path = PROCESSED_DIR / "tracks_mobile.parquet"
    if trk_mob_path.exists():
        trk_mob = pd.read_parquet(trk_mob_path)
        trk_mob["date"]     = trk_mob["time"].dt.tz_convert("Europe/Madrid").dt.strftime("%d-%m-%Y")
        trk_mob["trayecto"] = trk_mob["session_id"]
        trk_cols = ["date", "trayecto", "source", "lat", "lon", "ele", "time"]
        tracks_all = pd.concat([trk_mic[trk_cols], trk_mob[trk_cols]], ignore_index=True)
        print(f"  Tracks: {len(trk_mic)} mic + {len(trk_mob)} móvil = {len(tracks_all)} total")
    else:
        trk_cols = ["date", "trayecto", "source", "lat", "lon", "ele", "time"]
        tracks_all = trk_mic[trk_cols].copy()
        print(f"  Tracks: {len(trk_mic)} mic (sin datos móvil)")

    tracks_all.to_parquet(PROCESSED_DIR / "tracks.parquet", index=False)

    # Join espacio-temporal via merge_asof (nearest GPS within threshold per source).
    # La columna "trayecto" se hereda del GPS más cercano -> asignación automática.
    pred_combined["t_mid"] = (
        pred_combined["t_start"] + (pred_combined["t_end"] - pred_combined["t_start"]) / 2
    )
    n_total = len(pred_combined)

    gps_ref = (
        tracks_all[["date", "time", "lat", "lon", "trayecto"]]
        .rename(columns={"time": "gps_time"})
        .sort_values("gps_time")
    )

    parts = []
    for src, thr in [("mic", THRESHOLD_MIC), ("mobile", THRESHOLD_MOB)]:
        sub = pred_combined[pred_combined["source"] == src].sort_values("t_mid").copy()
        if sub.empty:
            continue
        merged = pd.merge_asof(
            sub, gps_ref,
            left_on="t_mid", right_on="gps_time",
            by="date",
            tolerance=pd.Timedelta(seconds=thr),
            direction="nearest",
        )
        merged = merged.drop(columns=["gps_time"])
        parts.append(merged)

    if not parts:
        print("  [WARN] Sin predicciones para join")
        return

    pred_joined = pd.concat(parts, ignore_index=True).sort_values("t_start")
    pred_joined = pred_joined.drop(columns=["t_mid"])

    # Diagnóstico: eventos descartados por falta de GPS
    discarded = pred_joined[pred_joined["lat"].isna()]
    if len(discarded):
        print(f"  Descartados sin GPS: {len(discarded)}")
        for (date, src), g in discarded.groupby(["date", "source"]):
            thr_used = THRESHOLD_MIC if src == "mic" else THRESHOLD_MOB
            print(f"    {date} [{src}]: {len(g)} eventos — sin trackpoint ±{thr_used}s")
    else:
        print("  Todos los eventos tienen GPS asignado.")

    pred_geo = pred_joined.dropna(subset=["lat", "lon"])
    pred_geo.to_parquet(PROCESSED_DIR / "predictions_geo.parquet", index=False)

    n_mic = (pred_geo["source"] == "mic").sum()
    n_mob = (pred_geo["source"] == "mobile").sum() if "mobile" in pred_geo["source"].values else 0
    pct   = len(pred_geo) / n_total * 100
    print(f"  [OK] {len(pred_geo)} eventos geolocalizados ({pct:.1f}%) — {n_total - len(pred_geo)} descartados")
    print(f"       Mic: {n_mic} ({pred_geo[pred_geo['source']=='mic']['trayecto'].nunique()} trayectos) "
          f"| Móvil: {n_mob} ({pred_geo[pred_geo['source']=='mobile']['trayecto'].nunique() if n_mob else 0} sesiones)")
    print(f"  -> predictions_geo.parquet")
    print(f"  -> tracks.parquet")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ETL micrófonos fijos -> predictions_geo.parquet")
    parser.add_argument("--reprocess-all", action="store_true",
                        help="Regenera todos los parquets aunque ya existan")
    parser.add_argument("--skip-join", action="store_true",
                        help="Solo genera predictions_mic.parquet y tracks_mic.parquet, sin join")
    parser.add_argument("--no-cross-mic-nms", action="store_true",
                        help="Desactiva el NMS cruzado entre M1 y M2 (por defecto activado)")
    args = parser.parse_args()

    apply_nms = not args.no_cross_mic_nms

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    mic_out = PROCESSED_DIR / "predictions_mic.parquet"
    trk_out = PROCESSED_DIR / "tracks_mic.parquet"

    # Bloque A
    if args.reprocess_all or not mic_out.exists():
        process_predictions(raw=False, apply_cross_nms=apply_nms)
        process_predictions(raw=True,  apply_cross_nms=False)   # raw: sin NMS cruzado
    else:
        print(f"[A] SKIP predictions_mic.parquet (ya existe — usa --reprocess-all para forzar)")

    # Bloque B
    if args.reprocess_all or not trk_out.exists():
        process_tracks()
    else:
        print(f"[B] SKIP tracks_mic.parquet (ya existe — usa --reprocess-all para forzar)")

    # Bloque C
    if not args.skip_join:
        run_join()
    else:
        print("[C] SKIP join (--skip-join activado)")

    print("\n[DONE] ETL mic completado.")


if __name__ == "__main__":
    main()
