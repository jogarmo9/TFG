# Diseño ETL Rediseñado — TFG SED + GPS

## Contexto

El sistema de micrófonos generó audios que fueron limpiados y re-procesados → `predicciones_clean.csv`.
Se añade una segunda fuente: **grabaciones de móvil** (carpeta con `.gpx` + audio), donde GPX y audio los graba **apps distintas** → posible desfase temporal.
El objetivo es unificar ambas fuentes en un único dataset geo-localizado.

---

## Estado Actual

| Fuente | Datos | Estado |
|--------|-------|--------|
| Micrófonos audio | `clean/*.wav` → `predicciones_clean.csv` | ✅ Procesado |
| Micrófonos GPS | `data/raw/{DD-MM-YYYY}/*.gpx` | ✅ Disponible |
| Móvil audio | `data/mobile/{sesion}/audio.*` | ✅ Inferencia implementada |
| Móvil GPS | `data/mobile/{sesion}/track.gpx` | ✅ Procesado |

ETL implementado en `scripts/prepare_mic.py` (mic) y `scripts/prepare_mobile.py` (móvil).
El join espacio-temporal se ejecuta en `prepare_mic.py` (Bloque C).
`notebooks/01_etl.ipynb` es exclusivamente de validaciones visuales.

---

## Problema Clave: Sincronización Móvil

Apps separadas → el inicio del audio y del GPS pueden diferir segundos/minutos.

**Estrategia elegida**: cada sesión móvil incluye un `meta.json` que el usuario rellena con el inicio real del audio.

```json
{
  "audio_start_utc": "2026-05-10T14:23:05Z",
  "mic_id": 0,
  "session_id": "YYYYMMDD_ida",
  "notes": "Inicio grabación ~2s antes que GPS"
}
```

> **Fallback** si no hay `meta.json`: usar `mtime` del archivo de audio (advertencia en log).

---

## Estructura de Carpetas

```
TFG/
├── data/
│   ├── raw/                          # Datos históricos mic system
│   │   └── {DD-MM-YYYY}/
│   │       ├── *.gpx  (x2)
│   │       └── predicciones*.txt
│   │
│   ├── clean/                        # Audios limpios del mic system
│   │   └── YYYYMMDD_HH_MM_SS_MSMS_MX.wav
│   │
│   ├── mobile/                       # Grabaciones del móvil
│   │   └── NOMBRE_SESION/
│   │       ├── track.gpx             # Track GPS de la sesión
│   │       ├── audio.{wav|mp3|m4a}   # Grabación del móvil
│   │       └── meta.json             # audio_start_utc, mic_id, session_id
│   │
│   └── processed/                    # Outputs del ETL
│       ├── predicciones_clean.csv    # Detecciones mic raw (fuente de verdad)
│       ├── predictions_mic.parquet   # predicciones_clean.csv transformado
│       ├── tracks_mic.parquet        # GPS tracks de micrófonos fijos
│       ├── predictions_mobile.parquet# Detecciones de sesiones móvil
│       ├── tracks_mobile.parquet     # GPS tracks de sesiones móvil
│       ├── predictions_geo.parquet   # Dataset final con lat/lon ← principal
│       └── tracks.parquet            # GPS combinado (mic + móvil)
│
├── scripts/
│   ├── clean_audio_prueba.py         # Wiener filter + declip + lowpass
│   ├── infer_clean.py                # Inferencia YOLO → predicciones_clean.csv
│   ├── prepare_mobile.py             # Pipeline móvil: audio + GPX → parquet
│   └── prepare_mic.py                # ETL mic: CSV + GPX → predictions_geo.parquet
│
└── notebooks/
    ├── 01_etl.ipynb                  # Solo validaciones visuales (no hace ETL)
    ├── 02_analysis.ipynb
    └── 03_maps.ipynb
```

---

## Flujo ETL

```
┌──────────────────────────────────────────────────────────────┐
│  RAMA A — Sistema de micrófonos (prepare_mic.py Bloque A+B)  │
│                                                              │
│  predicciones_clean.csv                                      │
│    → parse timestamps (format='mixed')                       │
│    → tz_localize Europe/Madrid → UTC                         │
│    → TIME_CORRECTIONS (23-03-2026: -1h)                      │
│    → source='mic', duration_s                                │
│    → data/processed/predictions_mic.parquet                  │
│                                                              │
│  data/raw/{fecha}/*.gpx  (2 por día)                         │
│    → gpxpy → trackpoints con timestamp UTC                   │
│    → data/processed/tracks_mic.parquet                       │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│  RAMA B — Grabaciones móvil (prepare_mobile.py)              │
│                                                              │
│  Para cada data/mobile/{sesion}/:                            │
│  1. Leer meta.json → audio_start_utc                         │
│     (fallback: mtime del archivo de audio + warning)         │
│  2. Convertir audio a WAV 16kHz mono:                        │
│     MP3 → miniaudio.mp3_read_file_f32() (sin ffmpeg)         │
│     Otros → librosa.load(sr=16000, mono=True)                │
│  3. Filtro Wiener inline (declip + spectral + lowpass)       │
│  4. Inferencia YOLO (misma lógica que infer_clean.py)        │
│     conf ≥ 0.1, NMS 1D IoU ≥ 0.7                            │
│  5. Timestamps offset desde audio_start_utc                  │
│  6. source='mobile', append con deduplicación por session_id │
│    → data/processed/predictions_mobile.parquet               │
│    → data/processed/tracks_mobile.parquet                    │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│  MERGE + JOIN TEMPORAL (prepare_mic.py Bloque C)             │
│                                                              │
│  pred_combined = concat(predictions_mic, predictions_mobile) │
│  tracks_all    = concat(tracks_mic, tracks_mobile)           │
│  → data/processed/tracks.parquet                             │
│                                                              │
│  Para cada detección:                                        │
│    t_mid = t_start + (t_end − t_start) / 2                   │
│    → trackpoint más cercano en tiempo dentro de threshold    │
│    → asignar lat, lon                                        │
│                                                              │
│  Threshold por source:                                       │
│    'mic'    → 4 s   (timestamps de filename precisos)        │
│    'mobile' → 30 s  (desfase entre apps absorbido por meta)  │
│                                                              │
│    → data/processed/predictions_geo.parquet                  │
└──────────────────────────────────────────────────────────────┘
```

---

## Schema del Dataset Final (`predictions_geo.parquet`)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `microfono_id` | int | 1, 2 (mic system) · 0 (mobile) |
| `source` | str | `'mic'` · `'mobile'` |
| `session_id` | str | YYYYMMDD (mic) · nombre sesión (móvil) |
| `t_start` | datetime UTC | inicio detección |
| `t_end` | datetime UTC | fin detección |
| `class` | int | 0–8 |
| `confidence` | float | 0.1–1.0 |
| `duration_s` | float | duración en segundos |
| `date` | str | DD-MM-YYYY |
| `lat` | float | latitud (join GPS) |
| `lon` | float | longitud (join GPS) |

---

## Equipo de Preprocesado Acústico (idéntico en ambas ramas)

Ambos pipelines aplican el mismo preprocesado antes de la inferencia YOLO:

1. **Declipping**: interpola muestras saturadas (`|x| ≥ 0.85`) con CubicSpline
2. **Filtro Wiener**: estima ruido como percentil 15 del espectro → gain = `max(0, (mag - 0.85·noise) / mag)`
3. **Lowpass Butterworth** orden 4 a 7500 Hz
4. 2 pasadas por defecto (`--passes 2`)

---

## Plantilla meta.json

```json
{
  "audio_start_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "mic_id": 0,
  "session_id": "YYYYMMDD_ida",
  "notes": "descripción opcional del desfase o contexto"
}
```

Guardar en `data/mobile/{sesion}/meta.json` antes de ejecutar el ETL.

---

## Nota sobre Desfase Móvil

Si el desfase entre apps es constante por sesión (e.g., siempre ~30 s), el campo `audio_start_utc`
en `meta.json` lo absorbe. Si varía dentro de la sesión, las detecciones sin GPS quedarán
`lat=NaN` y no aparecerán en `predictions_geo.parquet`.

El dashboard de % join por día en `01_etl.ipynb` permite detectar sesiones con mal sync
y ajustar `audio_start_utc` en el `meta.json` correspondiente.

---

## Verificación

1. `python scripts/prepare_mic.py` → imprime conteos, genera 4 parquets (`predictions_mic`, `tracks_mic`, `predictions_geo`, `tracks`)
2. Segunda ejecución sin `--reprocess-all` → `[SKIP]` sin reprocesar
3. Ejecutar `01_etl.ipynb` completo → todas las celdas corren sin error
4. `pred_geo['source'].value_counts()` muestra `mic` y `mobile` (si se ejecutó prepare_mobile)
5. `predictions_geo.parquet` contiene columnas compatibles con NB02/NB03
