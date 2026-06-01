# Paso 4 — ETL Micrófonos Fijos (prepare_mic.py)

## Descripción

Transforma `predicciones_clean.csv` (detecciones brutas) en un dataset analítico geolocalizados. El script integra también los datos del pipeline móvil y realiza el join espacio-temporal GPS.

**Script:** `scripts/prepare_mic.py`  
**Salidas principales:** `predictions_mic.parquet`, `tracks_mic.parquet`, `predictions_geo.parquet`, `tracks.parquet`

---

## 4.1 Bloque A — Predicciones Mic

**Entrada:** `data/processed/predicciones_clean.csv`  
**Salida:** `data/processed/predictions_mic.parquet`

### Transformaciones
```python
# Renombrado de columnas
mic_id           → microfono_id
timestamp_onset  → t_start
timestamp_offset → t_end
class_id         → class

# Parseo temporal
t_start, t_end = pd.to_datetime(format='mixed')

# Correcciones horarias puntuales (hardcoded por incidencia)
TIME_CORRECTIONS = {"23-03-2026": -1}   # -1 hora ese día

# Localización zona horaria
tz_localize("Europe/Madrid", ambiguous='infer', nonexistent='shift_forward')
  → .tz_convert("UTC")

# Campos derivados
duration_s = (t_end - t_start).total_seconds()
date       = t_start.dt.strftime("%d-%m-%Y")
source     = "mic"
```

### NMS Cruzado M1/M2
Elimina detecciones duplicadas entre micrófonos M1 y M2 del mismo evento físico.

**Criterio de duplicado** (para misma clase y mismo día):
- `|t_start_M1 - t_start_M2| ≤ 1.0 s`
- IoU temporal ≥ 0.30

**Resolución:** se conserva la detección con mayor confianza. En empate, prioridad a M1.

```python
iou = interseccion / (dur1 + dur2 - interseccion + 1e-9)
```

Activo por defecto; desactivable con `--no-cross-mic-nms`.

---

## 4.2 Bloque B — Tracks GPS Mic

**Entrada:** `data/raw/DD-MM-YYYY/*.gpx`  
**Salida:** `data/processed/tracks_mic.parquet`

### Estructura de trayectos
- Cada día tiene 2 GPX: `PAIPORTA-ETSE_N` (mañana) y `ETSE-PAIPORTA_N` (tarde)
- Contador N se incrementa solo si la fecha tiene predicciones y no está en `SKIP_DATES`
- `SKIP_DATES = {"11-03-2026"}` — GPS falla de sincronía (predicciones fuera de ventana)

### Asignación de nombres de trayecto
```python
ROUTE_DEFAULT = ("PAIPORTA-ETSE", "ETSE-PAIPORTA")
# Primer GPX del día → PAIPORTA-ETSE_N
# Segundo GPX       → ETSE-PAIPORTA_N
# ROUTE_NAMES sobreescribe por fecha si algún día es distinto
```

### Columnas de salida
```
date, trayecto, source, lat, lon, ele, time(UTC)
```

---

## 4.3 Bloque C — Join Espacio-temporal

**Entradas:** `predictions_mic.parquet` + `predictions_mobile.parquet` + `tracks_mic.parquet` + `tracks_mobile.parquet`  
**Salidas:** `predictions_geo.parquet`, `tracks.parquet`

### Algoritmo
```python
# Para cada detección, calcular punto medio temporal
t_mid = t_start + (t_end - t_start) / 2

# merge_asof: busca trackpoint GPS más cercano en tiempo
# dentro de un umbral por fuente
pd.merge_asof(pred, gps, left_on='t_mid', right_on='gps_time',
              by='date', tolerance=timedelta(seconds=threshold),
              direction='nearest')
```

| Fuente | Threshold | Justificación |
|--------|-----------|---------------|
| mic | 4 s | Timestamp derivado del nombre de fichero (ms de precisión) |
| mobile | 30 s | Desfase apps GPS/audio absorbido en meta.json |

### Detecciones sin GPS
- Quedan con `lat=NaN, lon=NaN` → se descartan de `predictions_geo.parquet`
- Caso conocido: `11-03-2026` (mic) — predicciones fuera de ventana GPS

### Schema final (`predictions_geo.parquet`)
| Columna | Tipo | Descripción |
|---------|------|-------------|
| `microfono_id` | int | 1, 2 (mic) · 0 (mobile) |
| `source` | str | `'mic'` · `'mobile'` |
| `session_id` / `trayecto` | str | ID de trayecto |
| `t_start` | datetime UTC | Inicio detección |
| `t_end` | datetime UTC | Fin detección |
| `class` | int | 0–8 |
| `confidence` | float | 0.1–1.0 |
| `duration_s` | float | segundos |
| `date` | str | DD-MM-YYYY |
| `lat` / `lon` | float | Coordenadas join GPS |

---

## 4.4 Volumen de datos (Mayo 2026)

| Dataset | Eventos | Fuente |
|---------|---------|--------|
| predictions_mic.parquet | ~5.801 | Mic M1+M2 (6 fechas) |
| predictions_mobile.parquet | ~3.867 | 15 sesiones mobile |
| **predictions_geo.parquet** | **~9.668** | **Total geolocalizados** |
| tracks.parquet | 28.006 pts GPS | 21 trayectos |

---

## 4.5 Ejecución

```bash
# Completo (A + B + C)
python scripts/prepare_mic.py --reprocess-all

# Solo ETL sin join (rápido — omite Bloque C)
python scripts/prepare_mic.py --skip-join

# Sin NMS cruzado M1/M2
python scripts/prepare_mic.py --no-cross-mic-nms
```

Bloque C siempre se ejecuta (join actualiza al añadir sesiones mobile sin re-inferir).
