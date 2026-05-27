# Pipelines de Inferencia Acústica — TFG

Existen dos pipelines para generar detecciones acústicas con el modelo YOLOv5n ONNX.
Ambos producen el mismo esquema de salida y aplican el mismo preprocesado de audio.

---

## Pipeline 1 — Micrófonos Fijos (ruta diaria)

**Script:** `scripts/infer_clean.py`  
**Entrada:** `data/clean/*.wav`  
**Salida:** `data/processed/predicciones_clean.csv`

### Paso previo — Limpieza de audio

Antes de ejecutar `infer_clean.py`, los WAV crudos deben pasar por `scripts/clean_audio_prueba.py`:

```
data/audios/*.wav  →  clean_audio_prueba.py  →  data/clean/*.wav
```

`clean_audio_prueba.py` aplica por canal:
1. **Declipping**: detecta muestras saturadas (`|x| ≥ 0.85`), interpola con CubicSpline
2. **Filtro Wiener**: estima ruido como percentil 15 de las magnitudes espectrales → gain = `max(0, (mag - α·noise) / mag)` con `α=0.85`
3. **Lowpass Butterworth** orden 4 a 7500 Hz
- Por defecto 2 pasadas (`--passes 2`)
- Skip automático si el archivo ya existe en `clean/` (usa `--reprocess-all` para forzar)

### Ejecución con skip/resume

```bash
python scripts/clean_audio_prueba.py              # salta ya limpiados
python scripts/clean_audio_prueba.py --reprocess-all   # fuerza todo
python scripts/infer_clean.py                     # salta ya inferidos
python scripts/infer_clean.py --reprocess-all     # fuerza todo
```

### Proceso de inferencia (`infer_clean.py`)

Para cada `*.wav` en `data/clean/` no ya procesado:

**1. Parseo del nombre de fichero**
```
20260305_15_21_47_0562_M1.wav
→ mic_id=1, file_start=2026-03-05 15:21:47.056
```

**2. Carga de audio**
```python
librosa.load(wav_path, sr=16000, mono=True)
```

**3. División en chunks de 10 segundos**
```python
n_chunks = ceil(n_samples / 160_000)
# último chunk: zero-pad hasta 160_000 muestras
```

**4. Por chunk → Mel spectrogram**
```python
STFT: n_fft=2048, hop=256, win=2048, center=True, window='hann'
→ melspectrogram: 128 mels, fmin=0, fmax=8000, htk=True
→ amplitude_to_dB
→ clip(-50, 80)
→ normalizar: (x - (-38.5)) / (41.37 - (-38.5))
# resultado: array float32 [128, ~626]
```

**5. Padding + tensor YOLO**
```python
# pad izquierda: 7 cols, pad derecha: hasta 640 cols total
# valor de padding: 0.447058824
# tensor: [1, 3, 128, 640]  (3 canales = misma imagen repetida)
```

**6. Inferencia YOLO**
```python
preds = session.run(None, {"images": tensor})
# shape [1, 13, 6400] → transpose → [6400, 13]
# columnas: [xc, yc, w, h, conf_cls0..conf_cls8]
```

**7. Decodificación de coordenadas**
```python
inner = 640 - 2*7 = 626   # columnas útiles
x1_sec = clip((xc - w/2 - 7) * 10 / 626, 0, 10)
x2_sec = clip((xc + w/2 - 7) * 10 / 626, 0, 10)
cls_id = argmax(preds[:, 4:])
conf   = max(preds[:, 4:])
```

**8. Filtro de confianza**
```python
conf >= 0.1  AND  x2 > x1
```

**9. NMS 1D (Non-Maximum Suppression)**
```python
# ordenar por confianza descendente
# para cada box superviviente, eliminar los siguientes con IoU >= 0.7
IoU_1D(b1, b2) = intersección / unión   (en segundos)
```

**10. Offset temporal absoluto**
```python
onset  = file_start + chunk_i*10s + x1_sec
offset = file_start + chunk_i*10s + x2_sec
```

**11. Escritura en CSV** (append si ya existe, write si es nuevo)
```
mic_id, timestamp_onset, timestamp_offset, class_id, confidence, source_file, session_id, source
```
- `session_id` = YYYYMMDD derivado del nombre de fichero
- `source` = `"mic"`

---

## Pipeline 2 — Sesiones Móvil (grabación con teléfono)

**Script:** `scripts/prepare_mobile.py`  
**Entrada:** `data/mobile/<NOMBRE_SESION>/` (audio + GPX + meta.json)  
**Salida:** `data/processed/predictions_mobile.parquet` + `data/processed/tracks_mobile.parquet`

### Estructura de entrada

```
data/mobile/PAIPORTA-ALDAIA/
├── meta.json          ← timestamps + session_id
├── audio.mp3          ← grabación del teléfono
└── track.gpx          ← ruta GPS
```

**`meta.json`:**
```json
{
  "audio_start_utc": "2026-05-23T12:18:47Z",
  "mic_id": 0,
  "session_id": "20260523_ida",
  "notes": "..."
}
```

### Ejecución

```bash
python scripts/prepare_mobile.py                              # todas las sesiones
python scripts/prepare_mobile.py --session data/mobile/PAIPORTA-ALDAIA
```

### Proceso por sesión

**1. Lectura de meta.json**
- `audio_start_utc`: timestamp UTC de inicio de grabación
- Fallback si no existe `meta.json`: usa `mtime` del archivo de audio (impreciso)

**2. Conversión a WAV 16kHz mono**
- MP3 → `miniaudio.mp3_read_file_f32()` → resample con librosa si sr ≠ 16kHz → `soundfile.write(PCM_16)`
- Otros formatos → `librosa.load(sr=16000, mono=True)`
- WAV ya en 16kHz → copia directa
- Nombre de salida: `YYYYMMDD_HH_MM_SS_0000_M{mic_id}.wav`

**3. Filtro Wiener (mismo que pipeline fijo)**
```python
from clean_audio_prueba import clean_audio
clean_audio(wav_raw, wav_clean, passes=2)
```
Aplica declipping + Wiener + lowpass idéntico al pipeline de micrófonos fijos.

**4. Inferencia YOLO** (`infer_wav_direct`)

Mismo proceso exacto que `infer_clean.py`:
- Carga WAV limpio → chunks 10s → mel [128×626] → pad [1,3,128,640]
- YOLO → decodificar → filtrar conf ≥ 0.1 → NMS 1D IoU ≥ 0.7
- Offset temporal usando `audio_start_utc` del meta.json

**5. Construcción del DataFrame**
```python
df_preds["session_id"] = session_id   # de meta.json
df_preds["source"]     = "mobile"
df_preds["timestamp_onset"]  = pd.to_datetime(..., format="ISO8601", utc=True)
df_preds["timestamp_offset"] = pd.to_datetime(..., format="ISO8601", utc=True)
```

**6. Parseo del track GPX**
- Extrae `lat, lon, ele, time` de todos los trackpoints
- `time` forzado a UTC
- Guardado en `tracks_mobile.parquet`

**7. Guardado append con deduplicación**
```python
# Si ya existe el parquet, elimina la sesión anterior y concatena
existing = existing[existing["session_id"] != session_id]
pd.concat([existing, df_preds]).to_parquet(...)
```

---

## Esquema de salida unificado

Ambos pipelines producen el mismo esquema de columnas:

| Columna | Tipo | Descripción |
|---------|------|-------------|
| `mic_id` | int | ID del micrófono (0 = móvil, 1/2 = fijos) |
| `timestamp_onset` | datetime UTC | Inicio de la detección |
| `timestamp_offset` | datetime UTC | Fin de la detección |
| `class_id` | float | Clase acústica (0–8) |
| `confidence` | float | Confianza del modelo (0.1–1.0) |
| `source_file` | str | Nombre del WAV procesado |
| `session_id` | str | YYYYMMDD (fijo) o nombre sesión (móvil) |
| `source` | str | `"mic"` o `"mobile"` |

**Clases (class_id → nombre):**
```
0=Horn  1=Siren  2=Pets  3=Physiological  4=Speech
5=Ring Tone  6=Vibrating  7=Notifications  8=Cry
```

---

## Parámetros del modelo (idénticos en ambos)

| Parámetro | Valor |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Chunk duration | 10 s |
| n_fft / hop / win | 2048 / 256 / 2048 |
| Mel bins | 128 |
| fmin / fmax | 0 / 8.000 Hz |
| dB clip | [−50, 80] |
| Normalización | min=−38.5, max=41.37 |
| YOLO input | [1, 3, 128, 640] |
| Padding cols | 7 cada lado |
| Padding value | 0.447058824 |
| Confianza mínima | 0.10 |
| NMS IoU threshold | 0.70 |

---

## Flujo completo hasta `predictions_geo.parquet`

```
data/audios/*.wav
    └─→ clean_audio_prueba.py (Wiener)
        └─→ data/clean/*.wav
            └─→ infer_clean.py
                └─→ data/processed/predicciones_clean.csv

data/mobile/<sesion>/{audio, gpx, meta.json}
    └─→ prepare_mobile.py (Wiener inline)
        └─→ data/processed/predictions_mobile.parquet
        └─→ data/processed/tracks_mobile.parquet

prepare_mic.py  ← ETL script (reemplaza lógica del antiguo 01_etl.ipynb)
    ├─ [A] lee predicciones_clean.csv → predictions_mic.parquet
    ├─ [B] lee data/raw/**/*.gpx      → tracks_mic.parquet
    ├─ [C] concat mic + mobile (si existe)
    ├─     join espacio-temporal (umbral 4s mic / 30s móvil)
    └─→ data/processed/predictions_geo.parquet   ← dataset principal
    └─→ data/processed/tracks.parquet

notebooks/01_etl.ipynb  ← solo validaciones visuales
    ├─ lee predictions_geo.parquet
    ├─ cobertura por día y fuente
    ├─ validación overlap temporal GPS ↔ predicciones
    ├─ calidad del join (% geolocalizados)
    └─ distribución de clases por fuente
```
