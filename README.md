# TFG — Análisis Acústico Espacial en Conducción Urbana

Detección automática de eventos acústicos (bocinas, sirenas, voz, etc.) durante trayectos en coche. Modelo YOLOv5n ONNX sobre espectrogramas mel. Dos fuentes: micrófonos fijos + smartphone. Resultados geolocalizados vía GPS para identificar zonas de riesgo acústico.

Más detalle técnico: [`docs/TFG_CONTEXT.md`](docs/TFG_CONTEXT.md)

---

## Estructura del Proyecto

```
TFG/
├── data/
│   ├── audios/          WAVs brutos micrófonos (entrada a clean_audio.py)
│   ├── raw/             GPX + predicciones raw originales (por fecha)
│   ├── clean/           WAVs limpios Wiener+ImpWiener
│   ├── clean_demucs/    WAVs stem voz Demucs (Speech — método actual)
│   ├── clean_dfn/       WAVs limpios DFN3-75 (Speech — legacy)
│   ├── mobile/          Sesiones smartphone (audio + meta.json + track.gpx)
│   └── processed/       Outputs del ETL
│       ├── predicciones_clean.csv      inferencia mic (CSV raw)
│       ├── predictions_mic.parquet
│       ├── tracks_mic.parquet
│       ├── predictions_mobile.parquet
│       ├── tracks_mobile.parquet
│       ├── predictions_geo.parquet     ← DATASET PRINCIPAL (lat/lon)
│       ├── predictions_filtered.parquet  conf ≥ 0.80
│       ├── danger_scores.parquet
│       └── tracks.parquet
│
├── models/
│   └── YOLOv5n_original.onnx
│
├── scripts/
│   ├── clean_audio.py        preprocesado Wiener/DFN3
│   ├── infer_clean.py        inferencia YOLO → predicciones_clean.csv
│   ├── prepare_mobile.py     pipeline móvil completo
│   ├── prepare_mic.py        ETL mic + join GPS
│   ├── gen_meta.py           genera meta.json desde GPX
│   └── geo_utils.py          helpers compartidos (velocidad, Wilson CI, grilla)
│
├── notebooks/
│   ├── 00_raw_vs_clean_comparison.ipynb  impacto filtro Wiener
│   ├── 01_etl.ipynb                      validación ETL
│   ├── 02_reliability_and_classes.ipynb  fiabilidad + triaje de dominio
│   ├── 02b_class_statistics.ipynb        estadísticas descriptivas por clase
│   ├── 03_acoustic_mobility.ipynb        acústica × movilidad GPS
│   ├── 04_danger_maps.ipynb              mapas de peligro
│   ├── 04_mobile_preprocess_eval.ipynb   validación preprocesado mobile
│   └── 05_dfn3_speech_tuning.ipynb       tuning DFN3 + comparativa Wiener
│
├── docs/
│   ├── TFG_CONTEXT.md        documento maestro del TFG
│   ├── ETL_ORDEN_EJECUCION.md  orden de ejecución completo del pipeline
│   └── pipeline/
│       ├── 01_captura_audio.md
│       ├── 02_preprocesado_mic.md
│       ├── 03_inferencia_mic.md
│       ├── 04_etl_mic.md
│       ├── 05_pipeline_mobile.md
│       └── 06_analisis_resultados.md
│
├── outputs/                  figuras y mapas HTML generados por notebooks
├── validation/               hoja de etiquetado manual (is_tp) para NB-02
└── requirements.txt
```

---

## Instalación

```bash
pip install -r requirements.txt
```

Demucs y DFN3 (Speech) requieren `.venv311` (Python 3.11). GPU AMD en Windows: `pip install torch-directml`. Ver [`docs/pipeline/02_preprocesado_mic.md`](docs/pipeline/02_preprocesado_mic.md).

---

## Orden de Ejecución del Pipeline

> Detalle completo en [`docs/ETL_ORDEN_EJECUCION.md`](docs/ETL_ORDEN_EJECUCION.md).

```
# Paso 1 — Preprocesado Wiener+ImpWiener (clases ≠ Speech)
python scripts/clean_audio.py --method wiener --impulse-removal

# Paso 2 — Preprocesado Demucs (stem voz, solo Speech) — requiere .venv311
.venv311\Scripts\python.exe scripts/clean_audio.py --method demucs

# Paso 3 — Inferencia YOLO dual-clean (pass 2 Speech usa Demucs por defecto)
python scripts/infer_clean.py --dual-clean

# Paso 4 — Pipeline móvil (si hay sesiones nuevas)
python scripts/prepare_mobile.py

# Paso 5 — ETL mic + join GPS → predictions_geo.parquet
python scripts/prepare_mic.py --reprocess-all

# → Abrir notebooks en orden: 00 → 01 → 02 → 02b → 03 → 04
```

---

## Clases Acústicas

| ID | Clase | Dominio |
|----|-------|---------|
| 0 | Horn (bocina) | Vial — peligro |
| 1 | Siren (sirena) | Vial — peligro |
| 4 | Speech | Contexto (radio/charla) |
| 2,3,5,6,7,8 | Pets/Physiological/Ring Tone/Vibrating/Notifications/Cry | Fuera de dominio (YOLO OOD) |

---

## Parámetros Clave

| Parámetro | Valor |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Chunk inferencia | 10 s |
| Confianza mínima (detección) | 0.10 |
| Confianza producción | 0.80 |
| NMS IoU | 0.70 |
| Join GPS mic | 4 s |
| Join GPS mobile | 30 s |
| Danger score grilla | 0.0005° (~50 m) |
| Pesos Horn / Siren | 1.0 / 2.0 |

---

*TFG — Grado en Ingeniería Informática, Universitat de València.*
