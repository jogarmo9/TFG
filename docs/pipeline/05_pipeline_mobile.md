# Paso 5 — Pipeline Móvil (prepare_mobile.py)

## Descripción

Pipeline completo e integrado para sesiones de grabación con smartphone. Realiza en un solo script la conversión de audio, preprocesado con SNR-gate Wiener, inferencia YOLO y parseo del track GPS.

**Script:** `scripts/prepare_mobile.py`  
**Entradas:** `data/mobile/<SESION>/` (audio + GPX + meta.json)  
**Salidas:** `data/processed/predictions_mobile.parquet`, `data/processed/tracks_mobile.parquet`

---

## 5.1 Proceso por sesión

### A) Lectura de meta.json
```json
{
  "audio_start_utc": "2026-05-23T14:18:42Z",
  "mic_id": 0,
  "session_id": "PAIPORTA-ALDAIA"
}
```
- `audio_start_utc` es el campo crítico para la sincronización temporal
- **Fallback 1**: si no existe `meta.json`, `gen_meta.py` lo genera desde el primer trackpoint GPX (puede haber desfase)
- **Fallback 2**: si no hay GPX, usa `mtime` del fichero de audio (impreciso — advertencia en log)

### B) Conversión a WAV 16kHz mono
| Formato entrada | Método |
|-----------------|--------|
| WAV | Copia directa si ya es 16kHz mono |
| MP3 | `miniaudio.mp3_read_file_f32()` → resample librosa si sr≠16kHz → `soundfile.write(PCM_16)` |
| MP4 / M4A | `ffmpeg` (ruta hardcoded + `shutil.which` fallback) |
| OGG, FLAC, otros | `librosa.load(sr=16000, mono=True)` |

Nombre de salida: `YYYYMMDD_HH_MM_SS_0000_M{mic_id}.wav`

### C) SNR-Gate Wiener (chunk a chunk)

Estrategia adaptativa: el filtro Wiener solo se aplica a chunks con ruido relevante.

```python
SNR_GATE_DB = 8.0  # umbral

for chunk_10s in audio:
    snr = estimate_snr(chunk)
    if snr > SNR_GATE_DB:
        output = chunk_raw          # chunk limpio → no filtrar
    else:
        output = wiener_mobile(chunk)  # chunk ruidoso → aplicar Wiener
```

**Estimación de SNR:**
```python
energias_frames = [mean(frame²) for frame in frames]
noise_energy  = mean(energias[energias <= percentil(20)])
signal_energy = mean(energias[energias >  percentil(20)])
SNR = 10 × log10(signal_energy / noise_energy)
```

**Parámetros Wiener Mobile** (optimizados en notebook 04):
```python
_WIENER_MOBILE_PARAMS = dict(hp_cutoff=100, nr_strength=0.50, lp_cutoff=8000, passes=2)
```
`nr_strength=0.50` (más conservador que mic: 0.85) para preservar más componentes de señal en grabaciones con señal de interés variable.

### D) Inferencia YOLO (inline)

Mismo proceso que `infer_clean.py` (ver Paso 3):
- Chunks 10s → mel [128×626] → pad [1,3,128,640] → YOLO → conf ≥ 0.10 → NMS IoU ≥ 0.70
- Timestamps absolutos usando `audio_start_utc` del meta.json

```python
onset = audio_start_utc + chunk_i × 10s + x1_sec
```

**Nota**: el ONNX en este script usa `CPUExecutionProvider` (pendiente de migración GPU).

### E) Parseo del track GPX
```python
# Para cada trackpoint:
rows.append({lat, lon, ele, time})
# time forzado a UTC si no tiene tzinfo
```

### F) Guardado con deduplicación por sesión
```python
# Append incremental: elimina sesión anterior antes de concatenar
existing = existing[existing["session_id"] != session_id]
pd.concat([existing, df_new]).to_parquet(path)
```
Permite re-ejecutar una sesión individual sin duplicar filas.

---

## 5.2 Tres versiones de predicciones por sesión

| Fichero | Contenido | Uso |
|---------|-----------|-----|
| `predictions_mobile.parquet` | Wiener + NMS (producción) | Dataset principal |
| `predictions_mobile_raw.parquet` | Wiener + pre-NMS | Debug/análisis |
| `predictions_mobile_noWiener.parquet` | Audio crudo + NMS | Comparación notebook 04 |

---

## 5.3 Diferencias respecto al pipeline Mic

| Aspecto | Mic | Mobile |
|---------|-----|--------|
| Audio entrada | WAV brutos (mic embebido) | MP3/MP4/WAV (smartphone) |
| Preprocesado | ImpWiener global (siempre) | SNR-gate Wiener (adaptativo) |
| nr_strength | 0.85 | 0.50 |
| Clase Speech | DFN3-75 (dual-clean) | Wiener estándar (pendiente migrar) |
| GPS sync | Timestamp en nombre fichero (ms) | meta.json (estimado, ±s) |
| mic_id | 1 (M1) o 2 (M2) | 0 (único micrófono) |
| NMS cruzado | Sí (M1 vs M2, prepare_mic.py) | No aplica |

---

## 5.4 Ejecución

```bash
# Todas las sesiones
python scripts/prepare_mobile.py

# Una sesión específica
python scripts/prepare_mobile.py --session data/mobile/PAIPORTA-ALDAIA

# Ajustar umbral SNR-gate
python scripts/prepare_mobile.py --snr-gate 6.0

# Forzar reprocesado completo
python scripts/prepare_mobile.py --reprocess-all
```
