# Pendiente — Figuras y LaTeX

## Ejecutar notebooks (orden importante, kernel `.venv311`)

1. **`notebooks/03_etl.ipynb`**
   - Regenera `trayecto_distribution.png` (fig 4.1): barras side-by-side, figsize 18×6, orden alfabético
   - Regenera `trayecto_distribution_per_min.png` (fig 4.2): mismo orden y paleta que 4.1

2. **`notebooks/04b_class_statistics.ipynb`**
   - Genera `class_distribution_before.png` — sin filtrar (n=15.168)
   - Genera `class_distribution_after.png` — tras umbral dinámico (n=2.904)
   - Genera `source_comparison_rate.png` — tasa de detección por fuente
   - Genera `source_comparison_confidence.png` — confianza media por clase y fuente
   - Genera `source_comparison_classes.png` — distribución porcentual de clases por fuente

3. **`notebooks/08_spectrograms.ipynb`** *(nuevo)*
   - Genera `spectrogram_mic_before.png` — espectrograma mic crudo
   - Genera `spectrogram_mic_after.png` — espectrograma mic preprocesado (Wiener+HPSS)
   - Genera `spectrogram_mobile_before.png` — espectrograma móvil crudo
   - Genera `spectrogram_mobile_after.png` — espectrograma móvil preprocesado (Wiener gated)
   - Parte B requiere `miniaudio` (`pip install miniaudio` en .venv311)

## Copiar PNGs a figs/

Después de ejecutar los notebooks, copiar de `outputs/` a `figs/`:

```
class_distribution_before.png
class_distribution_after.png
source_comparison_rate.png
source_comparison_confidence.png
source_comparison_classes.png
spectrogram_mic_before.png
spectrogram_mic_after.png
spectrogram_mobile_before.png
spectrogram_mobile_after.png
trayecto_distribution.png          (sobreescribe la existente)
trayecto_distribution_per_min.png  (sobreescribe la existente)
```

## Estado del LaTeX (`tex/resultados.tex`)

- ✅ Texto explicativo antes de cada figura (ya estaba escrito)
- ✅ Sección espectrogramas completa con 4 figuras separadas
- ✅ Distribución de clases separada en 2 figuras (before/after)
- ✅ Comparación entre fuentes separada en 3 figuras
- ✅ Figura 4.9 (`stat_distraction_index`) eliminada
- ❌ Las 9 figuras nuevas no existen aún en `figs/` — ejecutar notebooks primero

## Notas

- `trayecto_distribution` (4.1): muestra todos los eventos sin umbral de confianza (pred_geo, n=8.876)
- `trayecto_distribution_per_min` (4.2): mismo orden alfabético que 4.1, misma paleta steelblue/coral
- `class_distribution_sidebyside.png` ya no se usa — reemplazada por before/after
- `source_comparison_full.png` ya no se usa — reemplazada por rate/confidence/classes
- `stat_distraction_index.png` eliminada del LaTeX
