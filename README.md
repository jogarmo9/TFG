# TFG — Análisis Acústico Espacial en Conducción Urbana

Detección automática de eventos acústicos (bocinas, sirenas, voz, etc.) durante trayectos en coche. Modelo YOLOv5n ONNX sobre espectrogramas mel. Dos fuentes de datos: micrófonos fijos en vehículo + grabaciones de móvil. Resultados geolocalizados via GPS para identificar zonas de riesgo acústico.

---

## Estructura del Proyecto

```
TFG/
├── data/
│   ├── audios/                     # WAVs brutos de micrófonos (entrada a clean_audio_prueba.py)
│   ├── raw/                        # GPX + predicciones raw originales (por fecha)
│   │   └── DD-MM-YYYY/
│   │       ├── *.gpx
│   │       └── predicciones*.txt   # predicciones crudas (solo para comparación NB00)
│   ├── clean/                      # WAVs limpios (Wiener filter, 16kHz mono)
│   ├── mobile/                     # Sesiones grabadas con móvil
│   │   └── NOMBRE_SESION/
│   │       ├── meta.json           # audio_start_utc, session_id, mic_id
│   │       ├── audio.mp3           # grabación móvil
│   │       └── track.gpx           # ruta GPS
│   └── processed/                  # Outputs del pipeline
│       ├── predicciones_clean.csv  # detecciones mic (Wiener), fuente verdad
│       ├── predictions_mic.parquet # predicciones mic transformadas
│       ├── tracks_mic.parquet      # GPS tracks mic
│       ├── predictions_mobile.parquet
│       ├── tracks_mobile.parquet
│       ├── predictions_geo.parquet # dataset final con lat/lon ← principal
│       └── tracks.parquet          # GPS combinado (mic + móvil)
│
├── models/
│   └── YOLOv5n_original.onnx      # modelo YOLO (se descarga automáticamente)
│
├── scripts/
│   ├── clean_audio_prueba.py       # limpieza audio: declip + Wiener + lowpass
│   ├── infer_clean.py              # inferencia YOLO → predicciones_clean.csv
│   ├── prepare_mobile.py           # pipeline móvil: audio + GPX → parquet
│   └── prepare_mic.py              # ETL mic: CSV + GPX → predictions_geo.parquet
│
├── notebooks/
│   ├── 00_raw_vs_clean_comparison.ipynb  # impacto del filtro Wiener
│   ├── 01_etl.ipynb                      # validaciones visuales del ETL
│   ├── 02_analysis.ipynb                 # análisis estadístico
│   ├── 02b_analysis_gps_danger.ipynb     # GPS + zonas peligrosas
│   ├── 03_maps.ipynb                     # mapas Folium interactivos
│   └── visualize_detections.ipynb        # timeline + reproductor audio
│
├── markdown/                       # documentación del proyecto
│   ├── PIPELINES.md
│   ├── SECCIONES.md
│   ├── ETL_FLOW.md
│   └── ETL_REDESIGN.md
│
├── outputs/
│   ├── figures/                    # gráficos (ignorado en git)
│   └── maps/                       # mapas HTML (ignorado en git)
│
└── requirements.txt
```

---

## Instalación

```bash
pip install -r requirements.txt
```

El modelo YOLO se descarga automáticamente al ejecutar `infer_clean.py` o `prepare_mobile.py` si no existe en `models/`.

---

## Orden de Ejecución

```bash
# 1. Limpiar audio (Wiener filter) — solo si hay WAVs nuevos en data/audios/
python scripts/clean_audio_prueba.py

# 2. Inferencia YOLO → predicciones_clean.csv
python scripts/infer_clean.py

# 3. Pipeline móvil (opcional, si hay sesiones en data/mobile/)
python scripts/prepare_mobile.py

# 4. ETL mic: transforma CSV + GPX, join espacio-temporal → predictions_geo.parquet
python scripts/prepare_mic.py

# → Abrir notebooks en orden:
#   01_etl.ipynb          (validaciones ETL)
#   02_analysis.ipynb     (estadísticas)
#   02b_analysis_gps_danger.ipynb  (GPS + zonas peligrosas)
#   03_maps.ipynb         (mapas interactivos)
```

### Flags útiles

```bash
python scripts/clean_audio_prueba.py --reprocess-all     # fuerza relimpiar todo
python scripts/infer_clean.py --reprocess-all            # fuerza re-inferir todo
python scripts/prepare_mobile.py --session data/mobile/NOMBRE_SESION
python scripts/prepare_mic.py --reprocess-all            # fuerza regenerar parquets
python scripts/prepare_mic.py --skip-join                # solo genera predictions_mic + tracks_mic
```

---

## Clases Acústicas

| ID | Clase |
|----|-------|
| 0 | Horn (bocina) |
| 1 | Siren (sirena) |
| 2 | Pets |
| 3 | Physiological |
| 4 | Speech |
| 5 | Ring Tone |
| 6 | Vibrating |
| 7 | Notifications |
| 8 | Cry |

---

## Parámetros de Inferencia

| Parámetro | Valor |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Chunk | 10 s |
| Confianza mínima | 0.10 |
| NMS IoU | 0.70 |
| Join GPS threshold mic | 4 s |
| Join GPS threshold móvil | 30 s |

---

*TFG — Grado en Ingeniería Informática, Universitat de València.*
