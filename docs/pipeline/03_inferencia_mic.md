# Paso 3 — Inferencia YOLO (Micrófonos Fijos)

## Descripción

Los WAVs limpios se pasan por el modelo YOLOv5n ONNX para detectar eventos acústicos. El pipeline produce timestamps absolutos de cada detección (onset/offset UTC) y los almacena en CSV.

**Script:** `scripts/infer_clean.py`  
**Entradas:** `data/clean/*.wav` (pass 1) + `data/clean_demucs/*.wav` o `data/clean_dfn/*.wav` (pass 2 Speech)  
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

### H) NMS 1D intra-chunk (Non-Maximum Suppression)
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

### J) NMS cross-file (dedup entre archivos solapados)

Los archivos WAV grabados tienen duración fija de ~10s pero timestamps de inicio separados solo 2–5s entre sí (solapamiento real medido: ~5s). Un evento real queda dentro de la ventana de 2–3 archivos consecutivos y se detecta múltiples veces.

```
14_21_36 (cubre 14:21:36–14:21:46)  →  detecta evento en 14:21:42 ✓
14_21_39 (cubre 14:21:39–14:21:49)  →  detecta mismo evento en 14:21:42 ✓ (duplicado)
14_21_41 (cubre 14:21:41–14:21:51)  →  detecta mismo evento en 14:21:42 ✓ (duplicado)
```

Tras procesar todos los archivos del directorio, se aplica un segundo NMS sobre timestamps absolutos, agrupado por `(mic_id, class_id)`:

```python
# Greedy sobre timestamps absolutos (en segundos epoch)
# Agrupa por (mic_id, class_id) — M1 y M2 son micrófonos distintos, no se suprime entre ellos
CROSS_IOU_THRESH = 0.3

# Ventana temporal: solo compiten detecciones con |onset_A - onset_B| <= tol_s
# tol_s=15.0 (default) — cubre el solapamiento real (~5s) con margen x3
# Evita que dos eventos distintos de la misma clase en momentos alejados
# del día compitan erróneamente entre sí
```

Umbral IoU 0.3 (más bajo que el intra-chunk de 0.7) porque el mismo evento visto desde dos archivos distintos puede tener bounds ligeramente diferentes (contexto mel diferente en cada archivo).

La ventana temporal `tol_s=15s` es crítica: sin ella, el NMS greedy compara cada detección contra todas las del día en la misma clase, pudiendo suprimir eventos reales alejados en el tiempo que casualmente tienen IoU alto por duración similar.

El log de ejecución muestra el impacto:
```
[NMS cross-file] raw: 312 → 187 | det: 198 → 124
```

---

## 3.3 Modo Dual-Clean (procesamiento diferenciado por clase)

Activa con `--dual-clean`. Combina Wiener (clases != Speech) con separación de voz Demucs (Speech).

```
PASS 1: data/clean/         → inferir todas las clases EXCEPTO Speech (class_id=4)
PASS 2: data/clean_demucs/  → inferir SOLO Speech (class_id=4)   ← método actual
       (o data/clean_dfn/   → Speech con DFN3 legacy, via --speech-source dfn3)
→ combinar en predicciones_clean.csv
```

**Justificación pass 1:** Wiener genera armónicos residuales en la banda de voz que provocan FP en Speech. Para las demás clases (Horn, Siren, Ring Tone…) Wiener es suficiente y más rápido.

**Justificación pass 2:** DFN3 (denoiser) deja residuo tonal con ruido muy intenso → YOLO predice Speech FP. Demucs (separador de fuentes) extrae el stem de voz: sin voz real → stem silencioso → sin FP. Ver [02_preprocesado_mic.md §2.2](02_preprocesado_mic.md).

```bash
# Demucs (default, método actual)
python scripts/infer_clean.py --dual-clean

# DFN3 legacy (comparación)
python scripts/infer_clean.py --dual-clean --speech-source dfn3
```

### Aceleración GPU (ONNX)
```python
# Detección automática de CUDAExecutionProvider
available = ort.get_available_providers()
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
session = ort.InferenceSession(model_path, providers=providers)
```

---

## 3.4 Workflow completo con prefiltro VAD (método recomendado)

Demucs sobre todo el dataset es lento. El prefiltro descarta WAVs sin voz real antes de lanzar Demucs, reduciendo el lote a ~30–50% del total.

### ¿Qué es `data/clean_cand/`?

Wiener aplicado **sin HPSS** sobre todos los WAVs brutos. Es el audio de entrada al prefiltro: lo suficientemente limpio para que silero-vad detecte voz, lo suficientemente rápido de generar (sin el paso HPSS). Solo se genera una vez.

```bash
python scripts/clean_audio.py --method wiener --impulse-removal ^
  --clean-dir data/clean_cand --reprocess-all
```

---

### Opción A — Prefiltro silero-vad (recomendado)

silero-vad está entrenado para detectar voz humana real; no dispara con ruido de motor/tráfico. Al contrario que YOLO-Speech sobre Wiener, no confunde armónicos residuales con voz → selección más limpia.

**Script:** `scripts/vad_candidates.py`  
**Salidas:** `data/processed/speech_vad_scores.csv`, `data/processed/speech_candidates.txt`

```bash
# 1. Puntuar todos los WAVs (pesado, una sola vez — crea speech_vad_scores.csv)
.venv311\Scripts\python.exe scripts/vad_candidates.py score --in-dir data/clean_cand

# 2. Ver barrido de umbrales y elegir (instantáneo desde cache)
.venv311\Scripts\python.exe scripts/vad_candidates.py select --threshold 0.5
#    Imprime tabla: umbral | candidatos | % dataset
#    threshold=0.5 → típicamente 30–50% del dataset

# 3. Demucs solo sobre candidatos (requiere .venv311)
.venv311\Scripts\python.exe scripts/clean_audio.py --method demucs ^
  --file-list data/processed/speech_candidates.txt --reprocess-all

# 4. Inferencia dual-clean, pass 2 solo sobre candidatos
python scripts/infer_clean.py --dual-clean --use-candidates
```

Parámetros de `vad_candidates.py`:

| Subcomando | Flag | Default | Descripción |
|------------|------|---------|-------------|
| `score` | `--in-dir` | `data/clean_cand` | Carpeta de WAVs a puntuar |
| `score` | `--scores-csv` | `data/processed/speech_vad_scores.csv` | CSV de salida |
| `select` | `--threshold` | `0.5` | Umbral `max_prob` silero para marcar candidato |
| `select` | `--out` | `data/processed/speech_candidates.txt` | Lista de candidatos |

---

### Opción B — Prefiltro YOLO-Speech sobre clean_cand

Usa el propio YOLO para detectar ficheros que disparan la clase Speech sobre el audio Wiener-sin-HPSS. Más rápido de preparar que silero (no necesita `vad_candidates.py score`), pero puede incluir FP por ruido musical.

```bash
# 1. YOLO prefiltro → speech_candidates.txt
python scripts/infer_clean.py --build-candidates ^
  --cand-dir data/clean_cand --cand-conf 0.06

# 2. Demucs solo sobre candidatos
.venv311\Scripts\python.exe scripts/clean_audio.py --method demucs ^
  --file-list data/processed/speech_candidates.txt --reprocess-all

# 3. Inferencia dual-clean, pass 2 solo sobre candidatos
python scripts/infer_clean.py --dual-clean --use-candidates
```

`--cand-conf 0.06`: umbral bajo a propósito — recall alto, no importa que incluya algo de ruido porque Demucs filtrará.

---

### Comparativa de prefiltros

| | silero-vad (Opción A) | YOLO-speech (Opción B) |
|---|---|---|
| Precision | Alta (no dispara con tráfico) | Media (FP por ruido musical) |
| Candidatos seleccionados | ~30–50% | ~62% |
| Paso previo pesado | `vad_candidates.py score` (una vez) | ninguno |
| Requiere `.venv311` en prefiltro | Sí (silero) | No (YOLO ONNX) |

---

### Post-filtro VAD Speech — crispeos residuales Demucs

El stem vocal de Demucs a veces contiene crispeos residuales (artefactos del modelo de separación) con suficiente duración y energía para que YOLO prediga Speech cuando no hay voz real. El filtro de duración mínima no es suficiente porque estos artefactos pueden durar varios segundos.

**Solución:** post-filtro silero-VAD sobre las detecciones Speech del pass 2, aplicado **después del NMS cross-file**, leyendo los stems ya existentes en `data/clean_demucs/`. No requiere re-ejecutar Demucs.

```powershell
# Inferencia dual-clean con post-filtro VAD (requiere .venv311 para silero-vad)
.venv311\Scripts\python.exe scripts/infer_clean.py --dual-clean --vad-speech-filter

# Ajustar umbral (default 0.4 — más alto = más estricto, mayor riesgo de perder voz real)
.venv311\Scripts\python.exe scripts/infer_clean.py --dual-clean --vad-speech-filter --vad-speech-threshold 0.5
```

El log muestra el impacto:
```
[VAD-filter] 312 → 287 Speech  (25 FP residuales descartados, threshold=0.4)
```

**Cómo funciona:**
1. Para cada detección Speech superviviente al NMS cross-file, carga el stem Demucs correspondiente (ya en `data/clean_demucs/`).
2. Extrae la ventana temporal de la detección del stem.
3. Corre silero-VAD frame a frame (ventanas de 512 muestras @ 16kHz).
4. Conserva si `max_prob >= threshold`; descarta si no (crispeo sin voz real).

**Umbral 0.4:** mismo valor calibrado en el prefiltro VAD de candidatos. Coherente con la etapa A. Ajustar si hay falsos descartes (voz muy suave) o FP residuales que persisten.

**Nota:** solo activo con `--speech-source demucs` (default). Con `--speech-source dfn3` se ignora.

---

### Flags relevantes de `infer_clean.py`

| Flag | Descripción |
|------|-------------|
| `--build-candidates` | Etapa B: YOLO sobre `--cand-dir` → `speech_candidates.txt` (no escribe CSV final) |
| `--cand-dir` | Carpeta de entrada para `--build-candidates` (default: `data/clean_cand`) |
| `--cand-conf` | Umbral de confianza del prefiltro YOLO (default: 0.06) |
| `--use-candidates` | En `--dual-clean`: limita el pass 2 a los ficheros en `speech_candidates.txt` |
| `--vad-speech-filter` | Post-filtro silero-VAD sobre Speech del pass 2 (crispeos residuales Demucs) |
| `--vad-speech-threshold` | Umbral silero-VAD para conservar detección Speech (default: 0.4) |

---

## 3.6 Mecanismo de skip/resume

Sin `--reprocess-all`: el script lee los `source_file` ya presentes en `predicciones_clean.csv` y salta los WAVs ya procesados. Modo append.

Con `--reprocess-all`: sobreescribe el CSV completo.

---

## 3.7 Formato de salida (CSV)

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

## 3.8 Parámetros del modelo (fijos, deben coincidir con entrenamiento)

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
| NMS IoU intra-chunk | 0.70 |
| NMS IoU cross-file | 0.30 |
| NMS ventana temporal cross-file (`tol_s`) | 15 s |
