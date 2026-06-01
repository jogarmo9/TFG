# Propuesta de Análisis y Mapas — TFG

Documento de brainstorming. Recoge ideas de análisis estadístico y cartográfico
sobre el dataset de detecciones acústicas geolocalizadas. No vincula a la
implementación actual; es punto de partida para los notebooks de análisis.

---

## Contexto del dataset

- **Clases (9):** Horn, Siren, Pets, Physiological, Speech, Ring Tone, Vibrating, Notifications, Cry
- **Detección:** box temporal `[x1_sec, x2_sec]` → **cada detección tiene duración**; tiene confidence.
- **GPS:** lat/lon interpolado por timestamp sobre el track.
- **Fuentes:**
  - **mic** = conductor *solo*, 2 micrófonos fijos (M1+M2), nr=0.85, ruta fija PAIPORTA-ETSE, marzo–abril.
  - **mobile** = conductor *acompañado*, 1 mic smartphone, nr=0.50, rutas variadas, mayo.
- **6 pasadas ALDAIA-PAIPORTA**: mismas condiciones aproximadas (solo varió ligeramente la hora) → **réplica pura**.

### Categorización funcional de clases
- **Distracción / cabina:** Speech, Ring Tone, Notifications, Vibrating, Cry, Pets, Physiological
- **Alerta / externa:** Horn, Siren

---

## ⚠️ Aviso de validez (crítico para la defensa)

La comparación **"solo vs acompañado" está confundida con la modalidad de sensor**:
mic y mobile difieren en nº de micrófonos, fuerza de filtrado (nr), rutas, fechas y precisión GPS.
→ Una diferencia en Speech entre mic y mobile **no es atribuible a la compañía** sin aislar el sensor.

**Implicación:**
- mic vs mobile se reporta como **descriptivo**, con el confound declarado explícitamente. No se vende como causal.
- Las **6 pasadas** (mismo sensor, mismas condiciones) son el activo más limpio → base de fiabilidad.

---

## Reframe central: las 6 pasadas = estudio de fiabilidad

6 pasadas en condiciones iguales → mide el **suelo de ruido del detector** (repetibilidad).
Cuantificar esa variabilidad ANTES de afirmar nada permite interpretar el resto del
análisis contra ese suelo. Es el ancla metodológica del TFG.

---

## Bloque A — Fiabilidad (6 pasadas ALDAIA-PAIPORTA)

Valida todo lo demás.

- Conteo por clase × pasada → tabla **media ± std + coeficiente de variación (CV)**.
- Clases con CV alto = poco fiables → caveat para las hipótesis.
- **Consistencia espacial:** grilla ~100 m, ¿se activan las mismas celdas en las 6 pasadas?
  - Métrica: % celdas presentes en ≥4/6 pasadas; índice Jaccard entre pares de pasadas.
- **Salida:** "el detector repite la clase X con CV Y%" → confianza cuantificada.
- **Figuras:** barras de CV por clase; mapa de consistencia (verde=estable, rojo=esporádico).

---

## Bloque B — Estadística general (todos los trayectos)

- Densidad por clase: **det/km** y **det/min** (usa GPS + duración).
- **Duración media de detección por clase** → exposición vs evento puntual.
- **Índice de distracción** = % de tiempo con alguna clase-cabina activa (unión de intervalos).
  - Reportar también la versión por conteo de eventos (duración ≠ frecuencia, mensaje distinto).
- **Perfil por velocidad** (velocidad derivada del GPS): bin de velocidad × clase
  → hipótesis "parado/lento en cruce → más Horn/Speech".
- **mic vs mobile:** descriptivo, **con confound declarado**.
- **Tablas:** clase × (n, %, det/km, dur_media, conf_media, CV).

### Hipótesis candidatas a defender
- Acompañado → más Speech → mayor potencial de distracción *(declarar confound de sensor)*.
- Zonas lentas/paradas → más eventos de alerta (Horn).
- La exposición (duración) cuenta una historia distinta al conteo de eventos.

---

## Bloque C — Mapas

Normalización explícita: **mic ÷2 micrófonos, mobile ÷1**.

1. **Exposición sonora** = Σ duración de detección por celda.
2. **Densidad normalizada por tramo (1 km)** = det/seg de tramo → corrige velocidad/paradas.
3. **Densidad normalizada por trayecto** → comparar rutas de distinta longitud.
4. **Danger externo** = Horn×1 + Siren×2 (métrica ya existente).
5. **Distracción cabina** = Σ clases-cabina, normalizada.
6. **KDE / heatmap** de hotspots sobre todas las rutas.
7. **Mapa de consistencia** (del Bloque A).

Cada métrica de ponderación responde una pregunta distinta:
- cruda → "¿dónde dispara el modelo?"
- por duración → "¿dónde hay más exposición sonora?"
- por tramo → "¿dónde hay más densidad real corrigiendo la velocidad?"
- por trayecto → "¿qué ruta es más ruidosa?"

---

## Bloque D — Speech ↔ Ring Tone (artefacto del detector)

Hipótesis: YOLO predice Speech y Ring Tone juntas cuando suena música.
(El doc técnico ya documenta que Wiener induce falsos Speech — banda de voz.)

- **Co-ocurrencia:** IoU temporal Speech ∩ Ring Tone vs esperado por azar.
- **Asimetría:** ¿Ring Tone cae casi siempre dentro de la ventana de Speech?
- **Confidence:** confianza de Ring Tone cuando co-ocurre vs cuando aparece sola
  → confianza baja = señal de falso positivo.
- **Extensión** a la familia "móvil": Notifications ↔ Vibrating, Ring Tone ↔ Notifications.
- **Salida:** regla de post-proceso propuesta (merge de clases o supresión condicional)
  + impacto cuantificado en los conteos.

---

## Tratamiento del umbral de confianza

No se busca exactitud → usar **dos cortes (0.10 y 0.80)** y mostrar que el *patrón*
(ranking de clases, hotspots) se mantiene entre ambos. Robustez sin precisión:
se presenta como fortaleza metodológica, no como limitación.

---

## Preguntas abiertas (pendientes de decidir)

1. **Foco de la memoria:** ¿descriptivo (caracterizar el entorno sonoro) o accionable
   (mapa de peligro como producto)? Define qué bloque es resultado estrella vs apoyo.
2. **Nº de trayectos mobile** aparte de las 6 pasadas (dimensiona el Bloque B).
3. **Definición final del índice de distracción**: ¿duración (% tiempo activo) o conteo?

---

## Mapeo a notebooks (propuesto)

| Bloque | Contenido | Notebook sugerido |
|--------|-----------|-------------------|
| A | Fiabilidad 6 pasadas (CV, consistencia espacial) | `04_reliability_and_classes.ipynb` |
| B | Estadística general, densidades, velocidad | `04b_class_statistics.ipynb` |
| B | Acústica vs movilidad (velocidad, distracción) | `05_acoustic_mobility.ipynb` |
| C | Mapas ponderados, KDE, danger | `06_danger_maps.ipynb` |
| D | Speech↔Ring Tone, artefactos | (sección en `04_reliability_and_classes.ipynb`) |
