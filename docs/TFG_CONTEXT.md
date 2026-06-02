# Contexto TFG — Detección de Eventos Acústicos en Conducción Urbana

## Información Académica

- **Título tentativo:** Sistema de Detección de Eventos Acústicos en Entorno Urbano de Conducción
- **Institución:** Universitat de València (UV)
- **Grado:** Ingeniería Informática
- **Email autor:** jp.gadea@yurest.com
- **Modelo de IA:** YOLOv5n ONNX (adaptado para SED temporal 1D)
- **Período de datos:** Marzo–Mayo 2026

---

## 1. Motivación y Contexto

El entorno acústico urbano durante la conducción es rico en información: bocinas, sirenas, conversaciones y ruidos de alerta constituyen señales que, correctamente analizadas, permiten caracterizar la peligrosidad de distintas zonas de una ciudad. A diferencia de los sistemas de detección visual (cámaras), el audio es omnidireccional, funciona en condiciones de baja visibilidad y no requiere hardware caro.

El área de estudio son rutas urbanas del área metropolitana de Valencia: trayectos diarios entre Paiporta, Aldaia, Silla, Malvarrosa, Burjasot y Masanasa, zona que incluye barrios afectados por las inundaciones de la DANA (2024). La detección de patrones acústicos en estas rutas tiene relevancia directa para la evaluación de la seguridad vial post-emergencia.

---

## 2. Objetivo General

Desarrollar un sistema automático de detección y geolocalización de eventos acústicos durante trayectos en vehículo urbano, empleando técnicas de Deep Learning sobre espectrogramas mel, con dos fuentes de captura: micrófonos embebidos (sistema fijo) y smartphone (sistema portable). El dataset resultante permite identificar zonas con alta densidad de eventos de alerta (bocinas, sirenas) y construir un mapa de peligrosidad acústica.

---

## 3. Objetivos Específicos

1. Diseñar e implementar un pipeline de captura + preprocesado de audio para dos modalidades de sensor
2. Adaptar el modelo YOLOv5n para detección de eventos acústicos (SED) sobre espectrogramas mel 1D
3. Desarrollar un sistema de preprocesado diferenciado por clase acústica (Wiener para ruido estacionario, DFN3 para clase Speech)
4. Implementar el join espacio-temporal GPS ↔ detecciones para geolocalización automática
5. Analizar la distribución espacial y temporal de eventos acústicos en rutas reales
6. Construir un mapa de danger score basado en densidad de Horn y Siren por zona geográfica

---

## 4. Modelo de Detección — YOLOv5n (SED)

### Arquitectura
- **Base:** YOLOv5n (nano) reentrenado para Sound Event Detection
- **Input:** tensor `[1, 3, 128, 640]` — espectrograma mel (128 bins × 640 columnas)
- **Output:** `[1, 13, 6400]` — 6400 anchors × (4 coords + 9 scores de clase)
- **Detección temporal 1D:** el eje X del espectrograma representa el tiempo (10 s/chunk), el eje Y la frecuencia — el modelo detecta bounding boxes temporales [x1_sec, x2_sec]
- **Clases:** 9 (Horn, Siren, Pets, Physiological, Speech, Ring Tone, Vibrating, Notifications, Cry)
- **Formato:** ONNX (portable, sin dependencia de PyTorch en inferencia)

### Preproceso de entrada
```
Audio 16kHz → STFT (n_fft=2048, hop=256) → mel filterbank (128 mels, fmin=0, fmax=8kHz)
→ amplitude_to_dB → clip(-50, 80) → normalizar (min=-38.5, max=41.37)
→ pad a 640 cols (7 cols padding, valor 0.447) → triplicar canal → [1,3,128,640]
```

### Post-proceso
- Umbral confianza: 0.10 (detección) → 0.80 (análisis/producción)
- NMS 1D: IoU temporal greedy, umbral 0.70, cross-class (mismo evento puede activar múltiples clases)

---

## 5. Fuentes de Datos

### 5.1 Micrófonos Fijos (Sistema Embebido)

Sistema instalado en vehículo con dos micrófonos:
- **M1**: posición frontal
- **M2**: posición trasera
- Grabación WAV 16kHz, mono, PCM_16
- Ficheros de ~5s, nombrado `YYYYMMDD_HH_MM_SS_MSMS_MX.wav` (precision ms en nombre)
- **~16.383 ficheros WAV** en `data/audios/`
- Período: Marzo–Abril 2026
- Rutas: `PAIPORTA-ETSE` (mañana) + `ETSE-PAIPORTA` (tarde), 6 días de grabación
- GPS: 2 ficheros GPX por día en `data/raw/DD-MM-YYYY/`

### 5.2 Grabaciones Móvil (Smartphone)

Sesiones con smartphone para cubrir rutas adicionales:
- **15 sesiones** entre barrios del área valenciana (Mayo 2026)
- Audio: MP3/MP4/WAV (según app de grabación)
- GPS: fichero GPX de app separada → sincronización vía `meta.json`
- `mic_id=0` para todas las sesiones móvil
- Rutas: Paiporta↔Aldaia, Paiporta↔Malvarrosa, Silla↔Masanasa, Silla↔Burjasot, Paiporta↔Gym

### 5.3 Dataset Combinado (Junio 2026)

| Métrica | Valor |
|---------|-------|
| Eventos geolocalizados | ~9.668 |
| Eventos mic (M1+M2) | ~5.801 |
| Eventos mobile | ~3.867 |
| Puntos GPS | 28.006 |
| Trayectos únicos | 21 (6 mic + 15 mobile) |

---

## 6. Pipeline Técnico

### Diagrama de flujo

```
CAPTURA
├── Mic: data/audios/*.wav  ──────────────────────────────────────────────┐
│   └── GPS: data/raw/DD-MM-YYYY/*.gpx                                   │
│                                                                          │
└── Mobile: data/mobile/SESION/{audio, meta.json, track.gpx}             │
                                                                          │
PREPROCESADO                                                              │
├── Mic: clean_audio.py                                                   │
│   ├── data/clean/    ← Wiener+ImpWiener+HPSS-61                        │
│   │     Declip→Mediana×2→Wiener×2→HP→LP→HPSS armónico k=61             │
│   │     (HPSS elimina crispeos percusivos → reduce FP Ring Tone/Vibrat.)│
│   └── data/clean_dfn/ ← DFN3-75 (para Speech solamente)               │
│                                                                          │
└── Mobile: prepare_mobile.py (inline)                                    │
    └── SNR-gate Wiener (solo chunks con SNR ≤ 8dB)                      │
                                                                          │
INFERENCIA                                                                │
├── Mic: infer_clean.py --dual-clean                                      │
│   ├── data/clean/    → YOLO clases ≠ Speech  (HPSS ya aplicado)        │
│   └── data/clean_dfn/ → YOLO clase Speech                              │
│   └── → data/processed/predicciones_clean.csv                          │
│                                                                          │
└── Mobile: prepare_mobile.py (inline)                                    │
    └── → data/processed/predictions_mobile.parquet                      │
                                                                          │
ETL (prepare_mic.py)                                                      │
├── [A] predicciones_clean.csv → predictions_mic.parquet                  │
│   (timezone, NMS cruzado M1/M2, correcciones horarias)                 │
├── [B] GPX → tracks_mic.parquet                                          │
└── [C] Join espacio-temporal                                             │
    ├── mic: tolerance=4s                                                 │
    ├── mobile: tolerance=30s                                             │
    └── → predictions_geo.parquet ← DATASET PRINCIPAL                    │
        → tracks.parquet                                                  │
                                                                          │
ANÁLISIS (notebooks)                                                      │
├── 00: Validación preprocesado mobile                                    │
├── 01: Validación preprocesado mic + DFN3                                │
├── 02: Raw vs Clean (impacto Wiener)                                     │
├── 03: Validación ETL (cobertura, join quality)                          │
├── 04: Fiabilidad 6 pasadas (CV por clase, consistencia espacial)        │
│        + artefacto Speech↔Ring Tone                                     │
├── 04b: Estadística de clases (densidad det/km, det/min, duración)       │
├── 05: Acústica vs movilidad (velocidad, índice distracción)             │
└── 06: Mapas Folium (exposición, danger, KDE, consistencia)             │
```

---

## 7. Decisiones de Diseño Clave

### 7.1 Procesamiento Dual por Clase Acústica

**Problema:** el filtro Wiener introduce armónicos residuales en la banda de voz (300–3000 Hz) como artefacto del hard-thresholding espectral (`max(0, gain)`). El modelo YOLO es sensible a estas resonancias y las clasifica como Speech → tasa de falsos positivos elevada para esa clase.

**Solución:** pipeline dual-clean
- Clases 0–3, 5–8: Wiener (rápido, eficiente, suficiente)
- Clase 4 (Speech): DFN3-75 (red neuronal, preserva estructura armónica de voz)

**Parámetro DFN3:** `atten_lim_db=75` elegido tras comparativa en notebook 05: minimiza Speech false positives sin eliminar completamente detecciones reales.

### 7.2 Eliminación de Impulsos (ImpWiener)

**Problema:** el entorno de conducción genera impulsos de banda ancha (clicks de motor, golpes, crispeos de contacto). El filtro Wiener no los elimina porque trabaja con el percentil 15 del espectro como estimación de ruido — un impulso en un frame aislado no modifica esta estadística.

**Solución:** filtro de mediana en dos pasadas antes del Wiener:
- P1 (kernel=11, σ=2.5): picos grandes (~0.5% de muestras)
- P2 (kernel=15, σ=1.5): residuos más finos tras P1

### 7.3 SNR-Gate para Mobile

**Problema:** las grabaciones de smartphone mezclan segmentos con señal de interés clara (sirenas en intersecciones) con segmentos muy ruidosos (aceleración intensa). Aplicar Wiener uniformemente puede atenuar la señal real en chunks con buen SNR.

**Solución:** SNR-gate — solo se filtra si SNR estimado ≤ 8 dB. El SNR se estima por energía de frames: el 20% inferior = ruido; el resto = señal.

### 7.4 NMS Cruzado M1/M2

**Problema:** M1 y M2 registran el mismo evento físico simultáneamente → duplicados en el dataset.

**Solución:** tras el ETL, se elimina la detección de menor confianza de cada par (misma clase, mismo día, |t_start_M1 - t_start_M2| ≤ 1s, IoU temporal ≥ 0.30).

### 7.5 Join GPS por Fuente

Los timestamps de micrófonos fijos tienen precisión de milisegundos (embebidos en el nombre de fichero). Los timestamps de mobile tienen una incertidumbre de varios segundos (estimada en `meta.json`). Threshold diferenciado: 4s (mic) vs 30s (mobile).

---

## 8. Preprocesado Acústico — Justificación Técnica

### Filtro Wiener (substracción espectral)
```
Estimación ruido: noise[f] = percentil_15(|STFT(frame)|) para cada bin f
Ganancia:        gain[f] = max(0, (|X[f]| - nr × noise[f]) / |X[f]|)
Señal limpia:    X_clean[f] = gain[f] × |X[f]| × exp(j·φ[f])
```
- Ventaja: simple, rápido, eficaz contra ruido estacionario (motor, AC)
- Limitación: hard thresholding → ruido musical (armonicos residuales)
- Parámetros: nr=0.85 (mic), nr=0.50 (mobile), 2 pasadas

### DeepFilterNet3 (DFN3)
- Red neuronal entrenada para denoising en el dominio de bandas de ruido de banda equivalente
- `atten_lim_db`: limita la supresión máxima en dB para preservar señal débil
- Ventaja sobre Wiener: no introduce ruido musical; preserva estructura armónica de voz
- Desventaja: requiere Python 3.11, GPU recomendada, ~10× más lento que Wiener en CPU
- SR interno: 48kHz → requiere resample desde/hacia 16kHz

### Eliminación de Impulsos (mediana)
- Tipo de ruido objetivo: impulsos broadband de duración < 1 ms (clicks de motor, contacto metálico)
- Detección: `|x - mediana_local| > k × RMS` — la mediana es robusta a outliers → el umbral captura el impulso sin afectar la señal
- No aborda: ruido musical de Wiener (diferente naturaleza — resonancias tonales periódicas)

---

## 9. Métricas y Resultados Preliminares

### Dataset final
- **predictions_geo.parquet:** ~9.668 eventos con lat/lon
- **predictions_filtered.parquet:** eventos con conf ≥ 0.80

### Distribución de clases (estimada, pre-filtrado, conf ≥ 0.10)
Las clases con mayor presencia en entorno de conducción urbana son Speech (diálogos de radio, conversaciones), Horn (bocinas en tráfico), Siren (emergencias) y Physiological.

### Danger Score
- Grilla 0.001° (~111m × 86m)
- Score = Horn × 1.0 + Siren × 2.0, normalizado 0–100
- Las zonas con mayor danger score corresponden a intersecciones con alta densidad de tráfico y pasos de emergencias en las rutas monitorizadas

### Comparativa Mic vs Mobile
| Métrica | Mic | Mobile |
|---------|-----|--------|
| Fuentes | M1 + M2 (2 mic/día) | 1 mic/sesión |
| nr_strength | 0.85 | 0.50 |
| Precisión GPS join | 4s | 30s |
| Normalizar por | duración × 2 mics | duración × 1 mic |

---

## 10. Estructura de Archivos del Proyecto

```
TFG/
├── data/
│   ├── audios/          WAVs brutos mic (16kHz, ~16k ficheros)
│   ├── raw/             GPX + predicciones originales por fecha
│   ├── clean/           WAVs limpios Wiener+ImpWiener
│   ├── clean_dfn/       WAVs limpios DFN3-75 (Speech)
│   ├── mobile/          Sesiones smartphone (15 rutas, Mayo 2026)
│   └── processed/       Parquets del ETL
│       ├── predicciones_clean.csv      CSV inferencia mic
│       ├── predictions_mic.parquet
│       ├── tracks_mic.parquet
│       ├── predictions_mobile.parquet
│       ├── tracks_mobile.parquet
│       ├── predictions_geo.parquet     ← Dataset principal
│       ├── predictions_filtered.parquet ← conf≥0.80
│       ├── danger_scores.parquet       ← Grilla peligrosidad
│       └── tracks.parquet
│
├── models/
│   └── YOLOv5n_original.onnx          Modelo SED
│
├── scripts/
│   ├── clean_audio.py                 Preprocesado Wiener + DFN3
│   ├── infer_clean.py                 Inferencia YOLO (dual-clean)
│   ├── prepare_mobile.py              Pipeline móvil completo
│   ├── prepare_mic.py                 ETL mic + join GPS
│   └── gen_meta.py                    Genera meta.json desde GPX
│
├── notebooks/
│   ├── 00_mobile_preprocess_eval.ipynb   Validación preprocesado mobile
│   ├── 01_dfn3_speech_tuning.ipynb       Validación preprocesado mic + DFN3
│   ├── 02_raw_vs_clean_comparison.ipynb  Impacto Wiener
│   ├── 03_etl.ipynb                      Validación ETL
│   ├── 04_reliability_and_classes.ipynb  Fiabilidad 6 pasadas + Speech↔Ring
│   ├── 04b_class_statistics.ipynb        Estadística de clases (densidades)
│   ├── 05_acoustic_mobility.ipynb        Acústica vs movilidad (velocidad)
│   └── 06_danger_maps.ipynb              Mapas Folium (danger, exposición)
│
└── docs/
    ├── TFG_CONTEXT.md          ← este fichero
    ├── ANALISIS_PROPUESTA.md   Propuesta de análisis y mapas (brainstorming)
    └── pipeline/
        ├── 01_captura_audio.md
        ├── 02_preprocesado_mic.md
        ├── 03_inferencia_mic.md
        ├── 04_etl_mic.md
        ├── 05_pipeline_mobile.md
        └── 06_analisis_resultados.md
```

---

## 11. Tecnologías Empleadas

| Tecnología | Rol |
|-----------|-----|
| Python 3.11 | Entorno principal (DFN3 requiere 3.11) |
| PyTorch | Runtime DeepFilterNet3 |
| ONNX Runtime | Inferencia YOLOv5n (CPU/CUDA) |
| librosa | STFT, mel filterbank, resample |
| scipy | Filtros Butterworth, filtro mediana, STFT |
| DeepFilterNet (`df`) | Denoising neuronal Speech |
| soundfile / miniaudio | I/O audio (WAV, MP3 sin ffmpeg) |
| gpxpy | Parseo ficheros GPX |
| pandas / pyarrow | ETL y almacenamiento parquet |
| Folium | Mapas interactivos HTML |
| Jupyter / ipywidgets | Notebooks de análisis |

---

## 12. Limitaciones y Trabajo Futuro

### Limitaciones actuales
- **Mobile sin DFN3**: el pipeline mobile aún no usa el procesado dual-clean (pendiente de implementar)
- **MASANASA-SILLA_2_Revisar_GPS**: sesión con GPS potencialmente defectuoso (flaggeada)
- **11-03-2026 (mic)**: fecha sin solapamiento GPS — predicciones no geolocalizadas
- **Validación manual limitada**: umbral 0.80 validado por muestreo auditivo, no por anotaciones exhaustivas
- **Modelo preentrenado**: YOLOv5n no fue entrenado con datos propios — se usa sin fine-tuning en el dominio específico de conducción
- **Confound sensor ↔ compañía**: la comparación "solo (mic) vs acompañado (mobile)" está confundida con la modalidad de sensor (2 mics vs 1, nr 0.85 vs 0.50, rutas y fechas distintas). Las diferencias entre fuentes se reportan como descriptivas, no causales. Las 6 pasadas ALDAIA-PAIPORTA (mismo sensor, condiciones iguales) son la base limpia de fiabilidad. Ver `docs/ANALISIS_PROPUESTA.md`.

### Líneas de trabajo futuro
- Extender pipeline dual-clean (DFN3 Speech) al pipeline mobile
- Fine-tuning del modelo YOLOv5n con anotaciones propias del entorno de conducción
- Sistema de alertas en tiempo real integrado en el vehículo
- Análisis longitudinal: repetir mediciones en las mismas rutas tras mejoras de infraestructura

---

## 13. Glosario

| Término | Definición |
|---------|------------|
| SED | Sound Event Detection — detección y temporalización de eventos acústicos en un audio continuo |
| Wiener filter | Filtro de substracción espectral que atenúa el ruido estimando el suelo de ruido estacionario |
| DFN3 | DeepFilterNet3 — red neuronal de denoising entrenada con datos acústicos reales |
| atten_lim_db | Parámetro DFN3: máximo de atenuación en dB aplicable a cada banda frecuencial |
| mel spectrogram | Representación tiempo-frecuencia con escala de frecuencias perceptual (filtros triangulares mel) |
| NMS (1D) | Non-Maximum Suppression temporal: elimina detecciones solapadas dejando solo la de mayor confianza |
| IoU temporal | Intersección sobre unión de dos intervalos de tiempo [x1, x2] |
| SNR-gate | Estrategia adaptativa: filtrar solo chunks con SNR bajo el umbral |
| meta.json | Fichero de metadatos de cada sesión mobile con timestamp de inicio y session_id |
| join GPS | Asignación de coordenadas lat/lon a cada detección por cercanía temporal con trackpoints GPS |
| dual-clean | Modo de inferencia que combina dos fuentes de audio preprocesadas diferentemente (Wiener + DFN3) |
| danger score | Métrica de peligrosidad acústica por zona geográfica basada en densidad de Horn+Siren |
| ImpWiener | Pipeline combinado: eliminación de impulsos (mediana) + filtro Wiener |
