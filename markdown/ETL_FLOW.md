# Flujo de Ejecución ETL

## Rama A — Micrófonos

**1. Preparación de audio**
```
audios brutos → scripts/clean_audio_prueba.py → data/clean/*.wav
```
Declip + filtro Wiener + lowpass 7500 Hz. Salida: WAVs nombrados `YYYYMMDD_HH_MM_SS_MSMS_MX.wav`.
Skip automático si ya existe; `--reprocess-all` para forzar.

**2. Inferencia mic**
```
data/clean/*.wav → scripts/infer_clean.py → data/processed/predicciones_clean.csv
```
Chunks 10s → mel [128×626] → pad [1,3,128,640] → YOLO ONNX → conf ≥ 0.1 → NMS 1D IoU ≥ 0.7.
Skip por `source_file`; `--reprocess-all` para forzar.

**3. ETL mic**
```
data/processed/predicciones_clean.csv → predictions_mic.parquet
data/raw/{DD-MM-YYYY}/*.gpx           → tracks_mic.parquet
```
`scripts/prepare_mic.py` Bloques A + B.

---

## Rama B — Móvil

**1. Pipeline móvil**
```
data/mobile/{sesion}/meta.json   → lee audio_start_utc
data/mobile/{sesion}/audio.*     → convierte a WAV 16kHz mono (miniaudio para MP3)
                                 → filtro Wiener inline
                                 → inferencia YOLO (misma lógica que infer_clean.py)
                                 → conf ≥ 0.1, NMS 1D IoU ≥ 0.7
data/mobile/{sesion}/track.gpx   → trackpoints UTC
```
```
→ data/processed/predictions_mobile.parquet
→ data/processed/tracks_mobile.parquet
```
`scripts/prepare_mobile.py`. Append con deduplicación por `session_id`.

---

## Merge + Join (`scripts/prepare_mic.py` Bloque C)

```
predictions_mic + predictions_mobile  →  pred_combined
tracks_mic      + tracks_mobile       →  data/processed/tracks.parquet

Para cada detección:
  t_mid = t_start + (t_end - t_start) / 2
  buscar trackpoint con |t_mid - t_gps| < threshold
    mic:    threshold =  4s  (timestamps de filename precisos)
    móvil:  threshold = 30s  (desfase entre apps absorbido por meta.json)
  → asignar lat, lon

→ data/processed/predictions_geo.parquet
```

---

## Orden de Ejecución

```
1. scripts/clean_audio_prueba.py        (solo si hay WAVs nuevos en audios/)
2. scripts/infer_clean.py               (→ predicciones_clean.csv)
3. scripts/prepare_mobile.py            (opcional, si hay sesiones en data/mobile/)
4. scripts/prepare_mic.py               (merge + join GPS → predictions_geo.parquet)

5. notebooks/01_etl.ipynb               (validaciones visuales del ETL)
6. notebooks/02_analysis.ipynb          (análisis estadístico)
7. notebooks/02b_analysis_gps_danger.ipynb (GPS + zonas peligrosas)
8. notebooks/03_maps.ipynb              (visualización geográfica)
9. notebooks/visualize_detections.ipynb (timeline + reproductor audio)
```

### Flags útiles

```bash
python scripts/clean_audio_prueba.py --reprocess-all
python scripts/infer_clean.py --reprocess-all
python scripts/prepare_mobile.py --session data/mobile/NOMBRE_SESION
python scripts/prepare_mic.py --reprocess-all
python scripts/prepare_mic.py --skip-join        # solo A+B, sin join GPS
```

---

## Outputs del ETL

| Archivo | Generado por | Descripción |
|---------|-------------|-------------|
| `predicciones_clean.csv` | `infer_clean.py` | Detecciones mic raw |
| `predictions_mic.parquet` | `prepare_mic.py` | Predicciones mic transformadas |
| `tracks_mic.parquet` | `prepare_mic.py` | GPS mic |
| `predictions_mobile.parquet` | `prepare_mobile.py` | Predicciones móvil |
| `tracks_mobile.parquet` | `prepare_mobile.py` | GPS móvil |
| `predictions_geo.parquet` | `prepare_mic.py` | Dataset final con lat/lon |
| `tracks.parquet` | `prepare_mic.py` | GPS combinado (mic + móvil) |
