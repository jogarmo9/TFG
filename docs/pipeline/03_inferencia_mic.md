# Paso 3 — Inferencia YOLO (Micrófonos Fijos)

## Descripción

Los WAVs limpios se pasan por el modelo YOLOv5n ONNX para detectar eventos acústicos. El pipeline produce timestamps absolutos de cada detección (onset/offset UTC) y los almacena en CSV.

**Script:** `scripts/infer_clean.py`  
**Entradas:** `data/clean/*.wav` + `data/clean_dfn/*.wav` (modo dual)  
**Salida:** `data/processed/predicciones_clean.csv`

---

## 3.1 Modelo

| Atributo | Valor |
|----------|-------|
| Arquitectura | YOLOv5n adaptado para detección temporal 1D |
| Formato | ONNX (compatible CPU/GPU) |
| Input | Tensor `[1, 3, 128, 640]` (espectrograma mel triplicado) |
| Output | `[1, 13, 6400]` → 6400 anchors × 13 valores (xc, yc, w, h, conf×9 clases) |
| Fichero | `models/YOLOv5n_original.onnx` |
| Descarga | Automática si no existe (URL GitHub LFS del repositorio del modelo) |

---

## 3.2 Proceso de inferencia por fichero WAV

### A) Parseo del nombre de fichero
```
20260305_15_21_47_0562_M1.wav
→ mic_id=1
→ file_start = 2026-03-05 15:21:47.056 (UTC)
```
Los últimos 4 dígitos (MSMS) se dividen por 10 para obtener milisegundos.

### B) Segmentación en chunks de 10 segundos
```python
n_chunks = ceil(n_samples / 160_000)
# Último chunk: zero-padding hasta 160_000 muestras
```

### C) Mel spectrogram por chunk
```python
# STFT
X = |STFT(audio, n_fft=2048, hop=256, win=2048, window='hann', center=True)|
# Mel filterbank
mel = melspectrogram(S=X, sr=16000, n_mels=128, fmin=0, fmax=8000, htk=True, norm=None, power=1.0)
# dB + clip + normalización
mel = amplitude_to_dB(mel)
mel = clip(mel, -50, 80)
mel = (mel - (-38.5)) / (41.37 - (-38.5))    # resultado ∈ [0, 1] aprox.
# Dimensiones: [128, ~626]
```

### D) Padding y tensor YOLO
```python
# Padding izquierda: 7 cols (PAD_COLS), valor 0.447058824
# Padding derecha: hasta 640 cols totales
tensor = [mel_pad, mel_pad, mel_pad]   # [3, 128, 640] → [1, 3, 128, 640]
```
Los 7 columnas de padding representan el contexto temporal en el borde del chunk. El valor 0.447058824 es el valor neutro aprendido durante el entrenamiento.

### E) Inferencia ONNX
```python
preds = session.run(None, {"images": tensor})
# shape: [1, 13, 6400] → transpose → [6400, 13]
# columnas: [xc, yc, w, h, conf_cls0, ..., conf_cls8]
cls_id = argmax(preds[:, 4:])      # clase predicha
conf   = max(preds[:, 4:])         # confianza máxima
```

### F) Decodificación de coordenadas temporales
```python
inner = 640 - 2×7 = 626            # columnas útiles (sin padding)
x1_sec = clip((xc - w/2 - 7) × 10 / 626, 0, 10)
x2_sec = clip((xc + w/2 - 7) × 10 / 626, 0, 10)
```
La coordenada `xc` en píxeles se convierte a segundos dentro del chunk de 10s.

### G) Filtro de confianza
```python
conf >= 0.10   AND   x2 > x1
```

### H) NMS 1D (Non-Maximum Suppression)
```python
# Greedy, ordenado por confianza descendente, cross-class
def iou_1d(b1, b2):
    inter = max(0, min(b1[1], b2[1]) - max(b1[0], b2[0]))
    union = (b1[1]-b1[0]) + (b2[1]-b2[0]) - inter
    return inter/union if union > 0 else 0.0

# Eliminar boxes con IoU >= 0.70 respecto al superviviente de mayor conf
```

### I) Offset temporal absoluto
```python
onset  = file_start + chunk_i × 10s + x1_sec
offset = file_start + chunk_i × 10s + x2_sec
```

---

## 3.3 Modo Dual-Clean (procesamiento diferenciado por clase)

Activa con `--dual-clean`. Combina Wiener e ImpWiener (para clases ≠ Speech) con DFN3 (para Speech).

```
PASS 1: data/clean/    → inferir todas las clases EXCEPTO Speech (class_id=4)
PASS 2: data/clean_dfn/ → inferir SOLO Speech (class_id=4)
→ combinar en predicciones_clean.csv
```

**Justificación:** Wiener genera armónicos residuales en la banda de voz que provocan falsos positivos en Speech. DFN3 suprime el ruido sin introducir estos artefactos. Para las demás clases, Wiener es suficiente y más rápido.

```bash
python scripts/infer_clean.py --dual-clean
```

### Aceleración GPU (ONNX)
```python
# Detección automática de CUDAExecutionProvider
available = ort.get_available_providers()
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
session = ort.InferenceSession(model_path, providers=providers)
```

---

## 3.4 Mecanismo de skip/resume

Sin `--reprocess-all`: el script lee los `source_file` ya presentes en `predicciones_clean.csv` y salta los WAVs ya procesados. Modo append.

Con `--reprocess-all`: sobreescribe el CSV completo.

---

## 3.5 Formato de salida (CSV)

```
mic_id, timestamp_onset, timestamp_offset, class_id, confidence, source_file, session_id, source
```

| Columna | Ejemplo | Descripción |
|---------|---------|-------------|
| `mic_id` | 1 | ID micrófono (1=M1, 2=M2) |
| `timestamp_onset` | 2026-03-05T15:21:52.056 | Inicio detección (UTC, ISO) |
| `timestamp_offset` | 2026-03-05T15:21:57.332 | Fin detección |
| `class_id` | 0.0 | Clase acústica (float, 0–8) |
| `confidence` | 0.847 | Confianza del modelo |
| `source_file` | 20260305_15_21_47_0562_M1.wav | WAV origen |
| `session_id` | 20260305 | YYYYMMDD |
| `source` | mic | Fuente |

---

## 3.6 Parámetros del modelo (fijos, deben coincidir con entrenamiento)

| Parámetro | Valor |
|-----------|-------|
| Sample rate | 16.000 Hz |
| Chunk | 10 s = 160.000 muestras |
| n_fft / hop / win | 2048 / 256 / 2048 |
| Mel bins | 128 |
| fmin / fmax | 0 / 8.000 Hz |
| dB clip | [−50, 80] |
| Normalización | min=−38.5, max=41.37 |
| YOLO input | [1, 3, 128, 640] |
| Padding cols | 7 por lado |
| Padding value | 0.447058824 |
| Conf. mínima | 0.10 |
| NMS IoU | 0.70 |
