# Orden de Ejecución — ETL Completo

Fecha actualización: Junio 2026  
Pipeline actual: Wiener+ImpWiener (clases≠Speech) + DFN3-75 (Speech) + dual-clean

---

## Paso 1 — Preprocesado Wiener + ImpWiener

**Entrada:** `data/audios/*.wav`  
**Salida:** `data/clean/*.wav`  
**Entorno:** cualquier Python (no requiere .venv311)

```powershell
python scripts/clean_audio.py --method wiener --impulse-removal --reprocess-all
```

Cadena: `Declip → Mediana×2 (impulse) → Wiener×2 → HP 100Hz → LP 7999Hz`

Omitir `--reprocess-all` para saltarse los WAVs ya existentes en `data/clean/`.

---

## Paso 2 — Preprocesado DFN3-75 (solo Speech)

**Entrada:** `data/audios/*.wav`  
**Salida:** `data/clean_dfn/*.wav`  
**Entorno:** `.venv311` (Python 3.11) + GPU recomendada

```powershell
.venv311\Scripts\python.exe scripts/clean_audio.py --method dfn3 --atten-lim-db 75.0 --reprocess-all
```

El modelo DFN3 se carga una sola vez antes del loop. CUDA se activa automáticamente si disponible.

---

## Paso 3 — Inferencia YOLO (dual-clean)

**Entrada:** `data/clean/` (clases≠Speech) + `data/clean_dfn/` (Speech)  
**Salida:** `data/processed/predicciones_clean.csv`  
**Entorno:** cualquier Python

```powershell
python scripts/infer_clean.py --dual-clean
```

Pass 1: `data/clean/` → todas las clases excepto Speech (class_id=4)  
Pass 2: `data/clean_dfn/` → solo Speech

Para re-inferir desde cero: añadir `--reprocess-all` (sobreescribe CSV completo).

---

## Paso 4 — Pipeline Móvil (si hay sesiones nuevas)

**Entrada:** `data/mobile/SESION/{audio, meta.json, track.gpx}`  
**Salida:** `data/processed/predictions_mobile.parquet` + `data/processed/tracks_mobile.parquet`  
**Entorno:** cualquier Python

```powershell
# Todas las sesiones
python scripts/prepare_mobile.py

# Solo una sesión
python scripts/prepare_mobile.py --session data/mobile/NOMBRE_SESION

# Forzar reprocesado completo
python scripts/prepare_mobile.py --reprocess-all
```

Si hay sesiones sin `meta.json`, generarlo primero:
```powershell
python scripts/gen_meta.py
```

---

## Paso 5 — ETL Mic + Join GPS

**Entrada:** `predicciones_clean.csv` + `data/raw/**/*.gpx` + parquets mobile  
**Salida:** `predictions_mic.parquet`, `tracks_mic.parquet`, `predictions_geo.parquet`, `tracks.parquet`  
**Entorno:** cualquier Python

```powershell
python scripts/prepare_mic.py --reprocess-all
```

Bloques internos:
- **[A]** `predicciones_clean.csv` → `predictions_mic.parquet` (timezone, NMS cruzado M1/M2, correcciones)
- **[B]** `data/raw/**/*.gpx` → `tracks_mic.parquet`
- **[C]** Join espacio-temporal → `predictions_geo.parquet` + `tracks.parquet`

Solo re-ejecutar join (sin re-inferir):
```powershell
python scripts/prepare_mic.py   # Bloque C siempre se ejecuta
```

---

## Resumen visual

```
data/audios/*.wav
    │
    ├─[Paso 1]─ clean_audio.py --method wiener --impulse-removal
    │               └─→ data/clean/*.wav
    │
    └─[Paso 2]─ clean_audio.py --method dfn3 --atten-lim-db 75.0   (.venv311)
                    └─→ data/clean_dfn/*.wav

data/clean/ + data/clean_dfn/
    │
    └─[Paso 3]─ infer_clean.py --dual-clean
                    └─→ data/processed/predicciones_clean.csv

data/mobile/SESION/
    │
    └─[Paso 4]─ prepare_mobile.py
                    ├─→ data/processed/predictions_mobile.parquet
                    └─→ data/processed/tracks_mobile.parquet

predicciones_clean.csv + predictions_mobile.parquet + data/raw/**/*.gpx
    │
    └─[Paso 5]─ prepare_mic.py --reprocess-all
                    ├─→ data/processed/predictions_mic.parquet
                    ├─→ data/processed/tracks_mic.parquet
                    ├─→ data/processed/predictions_geo.parquet   ← DATASET PRINCIPAL
                    └─→ data/processed/tracks.parquet
```

---

## Flags útiles

| Flag | Script | Efecto |
|------|--------|--------|
| `--reprocess-all` | clean_audio, infer_clean, prepare_mic, prepare_mobile | Sobreescribe existentes |
| `--impulse-removal` | clean_audio | Activa filtro mediana anti-impulsos |
| `--atten-lim-db 75.0` | clean_audio | Supresión máxima DFN3 (default: 75) |
| `--dual-clean` | infer_clean | Combina Wiener + DFN3 por clase |
| `--session RUTA` | prepare_mobile | Solo procesa esa sesión |
| `--skip-join` | prepare_mic | Solo Bloques A+B, sin join GPS |
| `--no-cross-mic-nms` | prepare_mic | Desactiva NMS cruzado M1/M2 |
| `--snr-gate 8.0` | prepare_mobile | Umbral SNR para Wiener adaptativo |
