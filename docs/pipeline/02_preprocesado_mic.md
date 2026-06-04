# Paso 2 — Preprocesado de Audio (Micrófonos Fijos)

## Descripción

Los WAVs brutos de `data/audios/` se limpian antes de la inferencia. El pipeline aplica una cadena de filtros para eliminar ruido estacionario, impulsos (clicks/crispeos) y artefactos de saturación. La salida se almacena en `data/clean/` (Wiener), `data/clean_dfn/` (DFN3 legacy) y `data/clean_demucs/` (Demucs — separación de voz, método actual para Speech).

**Script:** `scripts/clean_audio.py`

---

## 2.1 Pipeline Wiener + ImpWiener → `data/clean/`

### Cadena de procesado (por canal)

```
WAV bruto → Declip → remove_impulses (×2) → Wiener (×2) → HP 100Hz → LP 7999Hz → WAV limpio
```

### Fase 1: Declipping
- Detecta muestras saturadas: `|x| ≥ 0.85` (normalizadas a [-1, 1])
- Interpola con CubicSpline sobre el conjunto de muestras no saturadas
- 2 pasadas con umbral decreciente (0.85, 0.82)
- Objetivo: recuperar transientes recortados por limitación hardware

### Fase 2: Eliminación de Impulsos (`remove_impulses`)
Activada con `--impulse-removal`. Elimina clicks y crispeos de banda ancha.

**Pasada 1 (global RMS):**
```python
med  = medfilt(audio, kernel_size=11)           # mediana local
rms  = sqrt(mean(audio²))                        # energía global
mask = |audio - med| > 2.5 × rms               # picos broadband
audio[mask] = med[mask]                          # sustituir por mediana
```

**Pasada 2 (global RMS, parámetros más sensibles):**
```python
rms2  = sqrt(mean(audio²))                       # RMS ya sin picos grandes
med2  = medfilt(audio, kernel_size=15)           # kernel más ancho
mask2 = |audio - med2| > 1.5 × rms2
audio[mask2] = med2[mask2]
```

**Justificación:** el filtro Wiener trabaja en el dominio frecuencial con estimación estadística del suelo de ruido (percentil 15 de frames). Un impulso broadband de un solo frame no modifica el percentil 15 → Wiener no lo elimina. La detección por `|x - mediana| > k × RMS` captura picos de amplitud elevada con ancho temporal muy corto (característica definitoria del impulso acústico).

### Fase 3: Filtro Wiener (substracción espectral)
```python
# Por frame (win=2048, hop=256, ventana Hanning):
spec  = rfft(frame × hanning)
mag   = |spec|
noise = percentile(mags_all_frames, 15, axis=0)   # estimación ruido estacionario
gain  = max(0, (mag - nr × noise) / (mag + 1e-8)) # nr = 0.85
cleaned_frame = irfft(gain × mag × exp(j·angle(spec)))
```
- 2 pasadas completas (declip→impulse→Wiener repetido)
- `nr_strength=0.85`: elimina ~85% del suelo de ruido estimado
- `max(0, ...)`: hard thresholding → puede generar ruido musical (armonicos residuales) en transiciones

### Fase 4: Filtros paso-banda
```python
HP Butterworth orden 4, fc=100 Hz   # elimina rumble de motor < 100Hz
LP Butterworth orden 4, fc=7999 Hz  # anti-aliasing (Nyquist=8000Hz)
```

### Ejecución
```bash
python scripts/clean_audio.py --method wiener --impulse-removal --reprocess-all
```

---

## 2.2 Pipeline Demucs → `data/clean_demucs/` *(método actual para Speech)*

Reemplaza DFN3 como preprocesado del pass 2 (Speech). Separación de fuentes en vez de denoising.

### ¿Por qué Demucs en lugar de DFN3?

DFN3 es un *denoiser*: atenúa el ruido estacionario pero deja residuo tonal cuando el ruido es muy intenso (motor, tráfico). Ese residuo tiene estructura armónica que YOLO confunde con voz → falsos positivos de Speech elevados. Ajustar `atten_lim_db` no resuelve el problema con ruido intenso.

**Demucs** (`htdemucs`) es un *separador de fuentes*: extrae el stem de **voz** de la mezcla sin intentar atenuar el resto. En chunks sin voz real, el stem queda casi en silencio → YOLO no detecta Speech. La voz real sobrevive limpia. Cambia el mecanismo, no el parámetro.

### Cadena Demucs
```
WAV bruto (16k) → resample 16k→44.1k → estéreo → htdemucs → stem vocals → mono → resample 44.1k→16k → WAV
```

### Parámetros
- Modelo: `htdemucs` (default) o `htdemucs_ft` (mejor calidad, ~4x más lento)
- SR interno: 44.100 Hz; salida: 16.000 Hz PCM_16
- Requiere `.venv311` (Python 3.11 + torch)

### Selección de dispositivo (automático)
Prioridad: **CUDA** (NVIDIA) > **DirectML** (GPU AMD/Intel en Windows, `pip install torch-directml`) > **CPU**

- CPU: ~10 s por fichero de 5 s
- GPU: sustancialmente más rápido

### Ejecución — lote completo
```bash
# Requiere .venv311
.venv311\Scripts\python.exe scripts/clean_audio.py --method demucs --reprocess-all

# Modelo de mayor calidad (más lento)
.venv311\Scripts\python.exe scripts/clean_audio.py --method demucs --demucs-model htdemucs_ft --reprocess-all

# Solo un rango de fechas (recomendado para validar antes del lote completo)
.venv311\Scripts\python.exe scripts/clean_audio.py --method demucs --date-from 20260414 --date-to 20260414
```

### Ejecución — solo candidatos Speech (modo acelerado con prefiltro VAD)

Demucs es lento (~10 s/fichero en CPU). Con el prefiltro se reduce el lote a los WAVs con voz probable (~30–50% del dataset). Ver workflow completo en [03_inferencia_mic.md §3.4](03_inferencia_mic.md).

```bash
# Paso 0: generar data/clean_cand/ (Wiener sin HPSS, rápido, solo una vez)
python scripts/clean_audio.py --method wiener --impulse-removal ^
  --clean-dir data/clean_cand --reprocess-all

# Paso 1A: puntuar con silero-vad (pesado, una vez)
.venv311\Scripts\python.exe scripts/vad_candidates.py score --in-dir data/clean_cand

# Paso 1B: elegir umbral y volcar lista de candidatos
.venv311\Scripts\python.exe scripts/vad_candidates.py select --threshold 0.5

# Paso 2: Demucs SOLO sobre candidatos
.venv311\Scripts\python.exe scripts/clean_audio.py --method demucs ^
  --file-list data/processed/speech_candidates.txt --reprocess-all
```

---

## 2.3 Pipeline DFN3 → `data/clean_dfn/` *(legacy)*

Método anterior para Speech. Conservado por compatibilidad y para comparación en NB-07.

### DeepFilterNet3
- Red neuronal de denoising en el dominio ERBN
- `atten_lim_db=75`: supresión agresiva conservando estructura tonal de voz
- SR interno: 48.000 Hz; requiere Python 3.11 (`.venv311`)
- Limitación: residuo tonal con ruido muy intenso → FP de Speech

### Ejecución
```bash
.venv311\Scripts\python.exe scripts/clean_audio.py --method dfn3 --atten-lim-db 75.0 --reprocess-all
```

---

## 2.4 Parámetros configurables

| Parámetro | Default | CLI flag | Descripción |
|-----------|---------|----------|-------------|
| `nr_strength` | 0.85 | — | Fuerza supresión Wiener |
| `passes` | 2 | `--passes` | Pasadas Wiener+declip |
| `impulse_removal` | False | `--impulse-removal` | Activa filtro mediana |
| `impulse_kernel` | 11 | `--impulse-kernel` | Kernel mediana P1 |
| `impulse_threshold` | 2.5 | `--impulse-threshold` | Umbral RMS P1 (σ) |
| `impulse_passes` | 2 | `--impulse-passes` | Número de pasadas mediana |
| `atten_lim_db` | 75.0 | `--atten-lim-db` | Supresión máxima DFN3 |
| `lp_cutoff` | 7999 | — | Frecuencia corte LP |
| `hp_cutoff` | 100 | — | Frecuencia corte HP |

---

## 2.5 Salidas

| Directorio | Contenido | Usado para |
|------------|-----------|-----------|
| `data/clean/` | WAV Wiener+ImpWiener | Inferencia clases != Speech (pass 1) |
| `data/clean_demucs/` | WAV stem vocals Demucs | Inferencia Speech — **método actual** |
| `data/clean_dfn/` | WAV DFN3-75 | Inferencia Speech — legacy (comparación) |

Los WAVs mantienen el mismo nombre que el original en `data/audios/`.
