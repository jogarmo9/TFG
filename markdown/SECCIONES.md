# Secciones de los Notebooks — TFG Análisis Acústico

---

## 00_raw_vs_clean_comparison.ipynb

| ID | Sección |
|----|---------|
| hdr001 | # 00 — Comparación Raw vs Clean |
| hdr002 | ## 1. Cobertura de Fechas |
| hdr003 | ## 2. Impacto Total — Solo Días Compartidos |
| hdr004 | ## 3. Cambio por Clase |
| hdr005 | ## 4. Distribución de Confianza |
| hdr006 | ## 5. Análisis por Día — Impacto de la Limpieza |
| hdr007 | ## 6. Zoom: Días con Mayor Impacto |
| hdr008 | ## 7. Días Solo en RAW — Pendientes de Limpiar |
| hdr009 | ## 8. Resumen Ejecutivo |

---

## 01_etl.ipynb

ETL movido a `scripts/prepare_mic.py`. Notebook solo validaciones.

| ID | Sección |
|----|---------|
| nb01_title | # 01 — Validación del ETL |
| nb01_setup | Setup: carga parquets procesados |
| nb01_s1_hdr | ## 1. Cobertura de Datos por Fuente |
| nb01_s2_hdr | ## 2. Validación Temporal — Overlap GPS ↔ Predicciones |
| nb01_s3_hdr | ## 3. Calidad del Join — % Geolocalizados por Día |
| nb01_s4_hdr | ## 4. Distribución de Clases por Fuente |
| nb01_s5_hdr | ## 5. Anomalías Conocidas |

---

## 02_analysis.ipynb

| ID | Sección |
|----|---------|
| 70fbfdc4 | # 02 — Análisis Estadístico de Eventos Acústicos |
| f5a2b3c4 | ## 1. Análisis de Obtención de Umbrales |
| 11d4d6bc | ## 2. Validación Visual de Confianza (Boxplots) |
| h7i8j9k0 | ## 3. Aplicación del Filtrado |
| a4f55c5b | ## 4. Análisis Descriptivo (Frecuencias) |
| c3ee224f | ## 5. Comparativa por Sensor (Micrófono) |
| 850177bd | ## 6. Evolución Temporal Diaria |
| v1a2l3i4 | ## 7. Validación Auditiva de Umbrales (Muestreo en la Frontera) |
| speech-continuity-header | ## 8. Validación de Continuidad de Speech |

---

## 02b_analysis_gps_danger.ipynb

| ID | Sección |
|----|---------|
| hdr001 | # 02b — GPS, Patrones Temporales y Zonas Peligrosas |
| hdr002 | ## 9. Estadísticas GPS del Coche |
| hdr003 | ## 10. Patrones Temporales y Duración de Eventos |
| hdr004 | ## 11. Score de Peligrosidad por Segmento km |
| hdr005 | ## 12. Comparativa Ida vs Vuelta (Mann-Whitney U) |
| hdr006 | ## 13. Validación de Zonas Peligrosas Conocidas |

---

## 03_maps.ipynb

| ID | Sección |
|----|---------|
| header | # 03 — Visualización Geoespacial e Impacto por Zona |
| f_h | ## 1. DATOS FILTRADOS (CONFIANZA ALTA) |
| f_heat_md | ### 1.1 Mapa de Calor |
| f_layers_md | ### 1.2 Capas por Clase |
| f_seg_md | ### 1.3 Densidad por KM (Semáforo) |
| r_h | ## 2. DATOS RAW (TODAS LAS DETECCIONES) |
| r_heat_md | ### 2.1 Mapa de Calor (Raw) |
| r_layers_md | ### 2.2 Capas por Clase (Raw) |
| r_seg_md | ### 2.3 Densidad por KM (Semáforo Raw) |
| raw_filtered_05_md | ### 2.4 Análisis de Sensibilidad por Confianza |
| speed_h | ## 2.5 Análisis de Velocidad de Desplazamiento |
| s3_hdr | ## 3. Mapa de Calor — Sólo Horn + Siren (Eventos Peligrosos) |
| s4_hdr | ## 4. Danger Score — Ruta Coloreada por Peligrosidad |
| s5_hdr | ## 5. Heatmaps por Clase Acústica |
| s6_hdr | ## 6. Cluster de Marcadores Interactivo |
| s7_hdr | ## 7. Contornos KDE — Núcleos de Riesgo |
| s8_hdr | ## 8. Trayectos Móvil vs Sistema Micrófonos |
| s9_hdr | ## 9. Zoom en Zonas Peligrosas Conocidas |

---

## visualize_detections.ipynb

| ID | Sección |
|----|---------|
| cell-0 | # Visualización de Detecciones SED |
| cell-2 | ## 1. Carga y limpieza de datos |
| cell-4 | ## 2. Distribución global de clases |
| cell-6 | ## 3. Distribución por hora del día (heatmap) |
| cell-8 | ## 4. Timeline por día y micrófono |
| cell-10 | ## 5. Confianza media por clase |
| cell-12 | ## 6. Reproducción de audios por día |
| cell-15 | ## 7. Resumen estadístico por día |

---

## SED.ipynb

Notebook exploratorio / prototipo. Sin secciones formales.
Contiene: inferencia YOLO manual, visualización de espectrogramas, exportación de clips de audio por clase.
