# Paso 6 — Análisis y Visualización de Resultados

## Descripción

A partir de `predictions_geo.parquet`, seis notebooks cubren validación, estadística, acústica×movilidad y mapas.

Helper compartido: `scripts/geo_utils.py` — clases/relevancia, haversine/velocidad, join temporal, Wilson CI, grilla espacial.

---

## Notebooks (orden de ejecución)

### 00 — Comparación Raw vs Clean
**`00_raw_vs_clean_comparison.ipynb`**  
Valida el impacto del filtro Wiener: cobertura, cambio en detecciones por clase, distribución de confianza pre/post.

---

### 01 — Validación ETL
**`01_etl.ipynb`**  
Verifica integridad del ETL: cobertura por fuente, solapamiento GPS↔predicciones, % join por día/sesión.  
Si `% join < 80%` en alguna sesión → revisar `audio_start_utc` en `meta.json` o tolerancia de join.

---

### 02 — Fiabilidad y clases
**`02_reliability_and_classes.ipynb`** — *¿En qué podemos confiar?*

| Sección | Contenido |
|---------|-----------|
| A2 — Triaje de dominio | Frecuencia de las 6 clases OOD; argumento para restringir peligro a Horn+Siren |
| Sensibilidad al umbral | Eventos supervivientes por clase y fuente a cada umbral 0.10–0.95 |
| A1 — Validación parcial | Muestreo estratificado clase×conf → etiquetado manual TP/FP → precisión por clase con IC Wilson |
| Calibración | Curva confianza→precisión empírica; umbral operativo por clase |

**Salidas:** `validation/sampling_sheet.csv` (rellenar `is_tp`), `outputs/nbA_*.png`

---

### 02b — Estadísticas descriptivas por clase
**`02b_class_statistics.ipynb`**

| Sección | Contenido |
|---------|-----------|
| 1 — Confianza | Violines + tabla P25/med/P75 por clase y fuente |
| 2 — Duración | Distribución `t_end−t_start`; correlación Spearman conf↔duración |
| 3 — Tasa de detección | Eventos/min y eventos/km por clase, trayecto y fuente; top-10 trayectos por Horn+Siren/km |
| 4 — Co-ocurrencia | Matriz 9×9 absoluta + P(B\|A) condicional (ventana ±5 s) |
| Tabla maestra | `outputs/stat_master_by_class.csv` |

---

### 03 — Acústica × movilidad GPS
**`03_acoustic_mobility.ipynb`** — *¿Qué dicen los eventos sobre el lugar y la conducción?*

| Sección | Contenido |
|---------|-----------|
| B1 — Velocidad | Derivada GPS (haversine/Δt); distribución por trackpoint y por detección |
| B2 — Stop-and-honk | ¿Horn se concentra a baja velocidad / en frenada? Violin + tabla `pct_low` + `brake_frac` |
| B3 — Repetibilidad | Densidad eventos/min por pasada en corredores repetidos (PAIPORTA↔ETSE ×6, etc.) |
| B4 — Mic vs Mobile | Mezcla de clases y tasa de detección comparada entre sensores |

---

### 04 — Mapas de peligro
**`04_danger_maps.ipynb`** — *El mapa de peligro, honesto.*

| Sección | Contenido |
|---------|-----------|
| C1 — Danger score v2 | KDE en grilla 0.0005° (~50m), pesos severidad × precisión (si hay validación), IC bootstrap sobre trayectos |
| C2 — Coroplético | Figura estrella: celdas coloreadas por score; borde en celdas estables (≥80% bootstrap) |
| C3 — Interactivo | Folium con capas (trayectos, Horn, Siren, heatmap), popups clase/conf/hora |
| C4 — Trayectoria velocidad | Ruta coloreada por velocidad + marcadores Horn/Siren |

**Salidas:** `outputs/map_danger_choropleth.html`, `outputs/map_interactive.html`, `outputs/map_speed_trajectory.html`

> Solo Horn(0) + Siren(1) suman al danger score. Speech = contexto, nunca peligro.

---

### 04/05 — Notebooks de preprocesado (validación técnica)

| Notebook | Propósito |
|----------|-----------|
| `04_mobile_preprocess_eval.ipynb` | Validación preprocesado mobile: grid search `nr_strength` vs `hp_cutoff`, comparativa Raw/Wiener |
| `05_dfn3_speech_tuning.ipynb` | Comparativa Raw/Wiener/ImpWiener/DFN3-75; exploración impulse removal; justificación `atten_lim_db=75` |

---

## Parámetros de análisis

| Parámetro | Valor |
|-----------|-------|
| Umbral producción | 0.80 |
| Grilla danger score | 0.0005° (~50 m) |
| Peso Horn / Siren | 1.0 / 2.0 |
| Ventana co-ocurrencia | ±5 s |
| Bootstrap remuestreos | 200 |
| Estabilidad celda | ≥ 80% remuestreos |
| Trayectos excluidos | `MASANASA-SILLA_2_Revisar_GPS`, `11-03-2026_skip*` |
