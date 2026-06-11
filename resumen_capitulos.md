# Resumen de capítulos — TFG Detección Acústica

> Documento de trabajo para revisar y proponer cambios en el texto.  
> Indica qué añadir, cambiar o eliminar y aplicamos los cambios.

---

## Capítulo 1 — Introducción (`introduccion.tex`)

### §1.1 Contexto y motivación
El ruido como factor de seguridad vial en entornos urbanos. Propuesta: integrar audio y GPS para identificar zonas acústicamente peligrosas con patrones reproducibles en lugar de detecciones aisladas.

**Estado:** completo. Coherente con el trabajo real.

### §1.2 Objetivo general
Cuantificar la exposición acústica durante conducción real combinando dos fuentes de audio (prototipo Raspberry Pi 4 + smartphone) con trazas GPS para identificar zonas de mayor concentración de eventos sonoros.

**Estado:** completo.

### §1.3 Objetivos específicos
Ocho objetivos: diseño de captura, preprocesado diferenciado, inferencia del modelo, ETL+GPS, análisis de clases, exposición a voz, análisis espacial y visualización.

**Estado:** completo. Verificar que los 8 objetivos aún se cubren en los resultados.

### §1.4 Estructura del documento
Guía de los 5 capítulos. Menciona: cap. 2 teoría, cap. 3 metodología, cap. 4 resultados, cap. 5 conclusiones.

**Estado:** completo.

---

## Capítulo 2 — Marco Teórico (`marco_teorico.tex`)

### §2.1 Detección de Eventos Sonoros (SED)
SED como clasificación con localización temporal de múltiples clases simultáneas aplicada a entornos de conducción.

### §2.2 Representación espectral
- STFT: 2048 FFT, hop 256, 16 kHz → ~16 ms resolución temporal.
- Mel spectrogram: 128 bandas (0–8 kHz), normalizado a [0,1].

### §2.3 YOLO aplicado a audio
Detector de paso único adaptado a espectrogramas. Predice cajas temporales + probabilidades de clase. NMS con IoU=0.7.

### §2.4 VAD y separación de fuentes
- VAD: prefiltro de ventanas con actividad vocal antes de la inferencia.
- Separación (Demucs): extrae componente vocal para mejorar la clase Speech.

### §2.5 Archivos GPX y trazas GPS
Estándar GPX: lat, lon, altitud, timestamp por trackpoint. Cruce temporal ±4 s con predicciones.

### §2.6 Proceso ETL
Extracción de predicciones y GPX, sincronización temporal, columna de origen, consolidación en parquet unificado.

### §2.7 Evaluación estadística
- CV (σ/μ): reproducibilidad. CV < 0.25 = clase estable.
- Umbral dinámico (μ\_c + σ\_c): reduce FP preservando detecciones frecuentes.

**Estado general:** completo. Coherente con el pipeline implementado.

---

## Capítulo 3 — Metodología (`metodologia.tex`)

### §3.1 Diseño general de la metodología
Arquitectura dual: prototipo + móvil con pipelines diferenciados. Base de datos unificada con columna `source`.

### §3.2 Dispositivos de captura
- **Prototipo Raspberry Pi 4:** estéreo, 16 kHz mono; ruido de distorsión significativo.
- **Teléfono móvil:** M4A/MP3, 44.1 kHz → 16 kHz mono; mayor calidad, preprocesado conservador.
- **GPS:** trazas GPX simultáneas.

### §3.3 Modelo de detección acústica (YOLOv5n)
1.9M parámetros, 9.63 MB ONNX, ~226 ms/segmento en Cortex-A72. Backbone (Conv+C3), Neck (SPPF), Head (3 escalas, 14 valores/anchor: 1 conf + 4 bbox + 9 clases). **Nuevo: diagrama TikZ añadido** (fig:yolo\_arch).

### §3.4 Pipeline de procesamiento, inferencia y ETL
Cuatro etapas: captura → preprocesado diferenciado → inferencia 10 s → join GPS.

- **§3.4.1 Preprocesado por fuente:** prototipo (Declip+Wiener×2+impulse+HP/LP), móvil (Wiener SNR-gate, SNR>8dB directo, ≤8dB Wiener α=0.50, HP 100 Hz).
- **§3.4.2 Tratamiento específico de la voz:** VAD silero prefiltro → Demucs stem vocal. Ya no marcado como "en validación".
- **§3.4.3 Inferencia y postprocesado:** ventanas 10 s, mel 128 bandas, NMS IoU=0.7.
- **§3.4.4 Integración espacial GPS:** ±4 s prototipo, ±30 s móvil; parquet unificado con `source`.

**Diagrama TikZ del pipeline (fig:pipeline\_flow):** actualizado con NMS cross-file (IoU≥0.30, ventana 15 s), VAD prefiltro candidatos, Demucs stem vocal, ~8.876 eventos → eliminado.

### §3.5 Recogida de datos
Rutas recurrentes PAIPORTA-ETSE / ETSE-PAIPORTA (~13 km, 17–21 min). Rutas variadas adicionales con móvil.

### §3.6 Análisis estadístico y visualización
4 etapas: distribución → umbral dinámico → CV reproducibilidad → mapas espaciales (Folium + matplotlib estático).

**Estado general:** completo. Coherente con el código.

---

## Capítulo 4 — Resultados (`resultados.tex`)

> ⚠️ **Nota al inicio del capítulo:** "resultados preliminares sujetos a revisión" — mantener hasta versión definitiva.

### §4.1 Efecto del preprocesado sobre representaciones espectrales
Compara espectrogramas de mel antes/después del filtrado para prototipo y móvil. **4 figuras** (spec\_mic\_before/after, spec\_mobile\_before/after).

- **Prototipo:** filtrado agresivo elimina energía difusa baja frecuencia.
- **Móvil:** cambio mínimo, solo atenúa ruido de fondo.

**Cifras pendientes de actualización:** ninguna (cualitativo).  
**⚠️ PNGs pendientes de exportar:** `spectrogram_mic_before/after.png`, `spectrogram_mobile_before/after.png` (desde `notebooks/08_spectrograms.ipynb`).

### §4.2 Conjunto de datos analizado
**8.876 eventos geolocalizados** (⚠️ cifra por actualizar tras reprocesado). Presenta distribución por trayecto (fig 4.5 absoluta, fig 4.6 tasa normalizada ev/min).

- Menciona "7 rutas de ida y vuelta (12 sesiones)" — **⚠️ posiblemente obsoleto** (datos tienen 11 trayectos por dirección = 22 trayectos mic).

### §4.3 Distribución de clases y fiabilidad del modelo
- Distribución antes/después del umbral dinámico (n=15.168 → n=2.904, 19.2%) — **⚠️ cifras por actualizar**.
- Distribución de confianza por clase (violín).
- Sensibilidad al umbral (curva ev. supervivientes vs umbral).
- CV por clase en 6 pasadas PAIPORTA-ETSE.

**Nota texto:** quedan los nombres "Horn" y "Siren" en la lista de definición de clases (línea 74, descriptiva). ¿Eliminar?

### §4.4 Exposición a voz en cabina
Analiza Speech como indicador de distracción. Rutas ETSE en solitario → Speech ≠ conversación (radio/GPS/llamadas). Medianas prototipo 13.5 % vs móvil 17.4 % — **⚠️ cifras por actualizar**.

Co-ocurrencia temporal entre clases (stat\_cooccurrence).

### §4.5 Análisis espacial: zonas de mayor actividad acústica
- **§4.5.1 Densidad y zonas conflictivas:** heatmap conf≥0.70 + mapa densidad ev/km (segmentos ~100 m).
- **§4.5.2 Comparación por dirección:** mapa comparativo ida (mañana) vs vuelta (tarde) con polylines det/km, misma escala verde-rojo con colorbar.

### §4.6 Movilidad acústica
Histogramas distribución velocidades GPS (todos trackpoints vs instantes de detección). Ligera sobrerrepresentación velocidades bajas. Mapa velocidad media GPS por segmento.

### §4.7 Comparación entre fuentes
- Tasa detección: móvil 2.3 ev/min vs prototipo 1.1 ev/min — **⚠️ cifras por actualizar**.
- Distribución porcentual de clases por fuente: Speech > Ring Tone > Vibrating en ambas, con mayor Vibrating/Physiological en prototipo.

### §4.8 Discusión
Resumen de aportaciones de la metodología dual. Limitación principal: FP del prototipo y supresión por preprocesado agresivo. Próximo paso: validación manual.

**Estado general:** coherente con el trabajo. Cifras numéricas necesitarán actualización completa tras reprocesar datos.

---

## Capítulo 5 — Conclusiones (`conclusiones.tex`)

### §5.1 Conclusiones
- Metodología dual funciona; distorsión del prototipo mitigada con preprocesado diferenciado.
- Voz: VAD + separación de fuentes implementado (Demucs operativo).
- Base de datos unificada con columna source → análisis diferenciado.
- Join GPS ±4 s → precisión espacial suficiente para mapas de exposición.
- CV como métrica de priorización de zonas.

**Estado:** coherente. Revisar si las conclusiones sobre Speech siguen marcando algo como WIP (ya no está en validación).

### §5.2 Trabajo futuro
8 líneas: ampliar sesiones, ensemble/voting entre fuentes, early-fusion, análisis WNR, **validación manual** (prioritario), variables externas (meteo/horario/tráfico), correlación velocidad-evento, completar subpipeline Speech.

---

## Apéndice (`apendice.tex`)

### Elementos pendientes de completar
- Convocatoria en portada
- Agradecimientos
- Validación Speech (VAD+Demucs) — ⚠️ ya implementado; revisar si este pendiente sigue vigente
- PNG mapas desde HTML interactivos — ⚠️ ya generados
- Rutas no recurrentes analizadas
- Ética/privacidad y reproducibilidad

### Secciones a valorar (opcionales)
- Estado del arte (ruido ambiental, deep learning acústico, análisis geoespacial)
- Ética y privacidad (captura de voz en escenario real)
- Reproducibilidad (estructura carpetas, notebooks, esquema BD, parámetros pipeline)

---

## Resumen de pendientes críticos

| Item | Tipo | Urgencia |
|---|---|---|
| Actualizar cifras n=15.168, n=2.904, 8.876 eventos, porcentajes | Tras reprocesar | Alta |
| Exportar 4 espectrogramas desde notebook 08 | Figura | Alta |
| Corregir "7 rutas de ida y vuelta (12 sesiones)" | Texto §4.2 | Media |
| Decidir si eliminar Horn/Siren de lista de clases §4.3 (L74) | Editorial | Baja |
| Actualizar pendientes del apéndice (Speech ya operativo, mapas PNG ya existen) | Editorial | Media |
| Completar convocatoria y agradecimientos | Portada | Media |
| Revisar conclusiones sobre Speech "en validación" | Texto §5.1 | Baja |
