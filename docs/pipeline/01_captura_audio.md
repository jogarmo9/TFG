# Paso 1 — Captura de Audio

## Descripción

El sistema recopila audio acústico mediante dos modalidades complementarias: micrófonos fijos instalados en el vehículo y grabación con smartphone. Ambas fuentes producen audio que se procesa con el mismo modelo de inferencia YOLO.

---

## 1.1 Micrófonos Fijos (sistema embebido)

### Hardware
- **Dos micrófonos**: M1 (posición frontal) + M2 (posición trasera) en el interior del vehículo
- Grabación simultánea, misma ventana temporal

### Formato de salida
```
YYYYMMDD_HH_MM_SS_MSMS_MX.wav
Ejemplo: 20260305_15_21_47_0562_M1.wav
```
- Resolución temporal: milisegundos (4 dígitos, divididos /10 para obtener ms reales)
- Sample rate: 16.000 Hz, mono, PCM_16
- Duración nominal: ~5 segundos por fichero (se concatenan en el pipeline)
- Carpeta: `data/audios/`

### Cobertura temporal
- Período: Marzo–Abril 2026
- Rutas: trayectos fijos diarios `PAIPORTA-ETSE` / `ETSE-PAIPORTA`
- ~16.383 ficheros WAV (M1+M2 combinados)

### GPS asociado
- Almacenado en `data/raw/DD-MM-YYYY/*.gpx` (2 ficheros GPX por día: ida + vuelta)
- Precisión temporal del join: ±4 s (timestamps derivados directamente del nombre de fichero)

---

## 1.2 Grabaciones Móvil (smartphone)

### Estructura por sesión
```
data/mobile/NOMBRE_SESION/
├── meta.json          ← sincronización temporal
├── audio.{mp3|mp4|m4a}  ← grabación
└── track.gpx          ← ruta GPS
```

### meta.json
```json
{
  "audio_start_utc": "2026-05-23T14:18:42Z",
  "mic_id": 0,
  "session_id": "PAIPORTA-ALDAIA",
  "notes": "desfase estimado ~3s respecto GPX"
}
```
- `audio_start_utc`: timestamp UTC de inicio de grabación — campo crítico para el join GPS
- GPS y audio grabados con apps separadas → posible desfase; absorbido en `audio_start_utc`
- Fallback si falta `meta.json`: `gen_meta.py` lo genera desde el primer trackpoint GPX

### Sesiones disponibles (Mayo 2026)
| Sesión | Formato | Estado |
|--------|---------|--------|
| ALDAIA-PAIPORTA | MP3 | OK |
| PAIPORTA-ALDAIA | MP3 | OK |
| BURJASOT-SILLA | MP4 | OK |
| GYM-PAIPORTA | MP3 | OK |
| MALVARROSA-PAIPORTA | MP3 | OK |
| MASANASA-SILLA_1 | MP4 | OK |
| MASANASA-SILLA_2_Revisar_GPS | MP4 | GPS a revisar |
| PAIPORTA-GYM | MP3 | OK |
| PAIPORTA-MALVARROSA | MP3 | OK |
| PAIPORTA-SILLA_1 | MP4 | OK |
| PAIPORTA-SILLA_2 | MP3 | OK |
| SILLA-BURJASOT | MP4 | OK |
| SILLA-MASANASA_1 | MP4 | OK |
| SILLA-MASANASA_2 | MP4 | OK |
| SILLA-PAIPORTA_1 | MP3 | OK |

Área geográfica: Valencia metropolitana (Paiporta, Aldaia, Silla, Malvarrosa, Burjasot, Masanasa).

---

## 1.3 Clases Acústicas del Sistema

El modelo detecta 9 clases:

| class_id | Clase | Relevancia |
|----------|-------|-----------|
| 0 | Horn | Alta — peligrosidad vial |
| 1 | Siren | Alta — emergencias |
| 2 | Pets | Media |
| 3 | Physiological | Media |
| 4 | Speech | Alta — indicador de contexto |
| 5 | Ring Tone | Baja |
| 6 | Vibrating | Baja |
| 7 | Notifications | Baja |
| 8 | Cry | Media |

---

## Scripts involucrados

| Script | Rol |
|--------|-----|
| `scripts/gen_meta.py` | Genera `meta.json` desde primer trackpoint GPX si no existe |

## Salidas

| Archivo | Descripción |
|---------|-------------|
| `data/audios/*.wav` | WAVs brutos mic (entrada al pipeline) |
| `data/mobile/*/` | Sesiones móvil con audio + GPX + meta |
| `data/raw/DD-MM-YYYY/*.gpx` | GPS de rutas mic |
