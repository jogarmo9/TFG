# Paso 2 — Preprocesado de Audio (Micrófonos Fijos)

## Descripción

Los WAVs brutos de `data/audios/` se limpian antes de la inferencia. El pipeline aplica una cadena de filtros para eliminar ruido estacionario, impulsos (clicks/crispeos) y artefactos de saturación. La salida se almacena en `data/clean/` (Wiener) y `data/clean_dfn/` (DFN3 para Speech).

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

## 2.2 Pipeline DFN3 → `data/clean_dfn/`

Usado exclusivamente para inferencia de la clase **Speech** (ver Paso 3).

### ¿Por qué DFN3 para Speech?
El filtro Wiener genera armonicos residuales en la banda de voz (300–3000 Hz) como artefacto del hard-thresholding. El modelo YOLO es sensible a estas resonancias y las clasifica erróneamente como Speech → falsos positivos elevados. DeepFilterNet3 (DFN3) utiliza una red neuronal profunda que preserva la estructura tonal de la voz sin introducir residuos armónicos.

### DeepFilterNet3
- Red neuronal de denoising en el dominio ERBN (bandas de ruido de banda uniforme)
- Parámetro clave: `atten_lim_db` = máximo dB de supresión
  - 75 dB: supresión agresiva, mantiene estructura harmónica de voz → elegido en nb05
  - 100 dB: sin límite de supresión
- SR interno: 48.000 Hz (el pipeline resamplea 16kHz→48kHz→16kHz)
- Requiere Python 3.11 (entorno `.venv311`)

### Cadena DFN3
```
WAV bruto → librosa resample 16k→48k → DFN3 (atten_lim_db=75) → librosa resample 48k→16k → WAV
```

### Optimizaciones implementadas
- **Modelo cargado una sola vez** (`_load_dfn3_model()`) antes del loop → evita recarga ×16k
- **CUDA automático**: `model.to("cuda")` + `audio.to("cuda")` si `torch.cuda.is_available()`
- En CPU: ~2-5s por fichero; en GPU: ~0.3-1s por fichero

### Ejecución
```bash
# Requiere .venv311
.venv311\Scripts\python.exe scripts/clean_audio.py --method dfn3 --atten-lim-db 75.0 --reprocess-all
```

---

## 2.3 Parámetros configurables

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

## 2.4 Salidas

| Directorio | Contenido | Usado para |
|------------|-----------|-----------|
| `data/clean/` | WAV Wiener+ImpWiener | Inferencia clases ≠ Speech |
| `data/clean_dfn/` | WAV DFN3-75 | Inferencia clase Speech |

Los WAVs mantienen el mismo nombre que el original en `data/audios/`.
