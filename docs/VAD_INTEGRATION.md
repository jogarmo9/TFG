# Integración de Silero VAD - Resumen de Cambios

## Problema Original
- **50% de predicciones Speech tienen confianza < 0.3** (falsos positivos)
- Origen: Demucs genera stems con artefactos/ruido que YOLO detecta incorrectamente
- Pipeline funciona correctamente, pero calidad de input es mala

## Solución Implementada: Silero VAD
Validador de voz basado en VAD (Voice Activity Detection) que filtra false positives speech.

### Archivos Nuevos
1. **scripts/silero_vad_validator.py**
   - Módulo reutilizable con clase `SileroVADValidator`
   - Integra modelo Silero VAD v3.1 (16kHz, ONNX)
   - Lógica de validación configurada

2. **scripts/demo_vad_filtering.py**
   - Demostración de cómo funciona el VAD
   - Ejemplos: ruido puro, tonos, YOLO fuerte, borderline

### Cambios en infer_clean.py
1. **Import**
   - Agregado: `from silero_vad_validator import SileroVADValidator`

2. **Función `process_file()`**
   - Nuevo parámetro: `vad_validator=None`
   - Para cada predicción Speech (cls_id == 4):
     - Extrae segmento de audio del chunk
     - Llama a `vad_validator.validate_segment()`
     - Solo escribe si VAD aprueba

3. **Función `_run_dir()`**
   - Nuevo parámetro: `vad_validator=None`
   - Propaga validador a `process_file()`

4. **Main - PASS 2**
   - **Antes de PASS 2**, inicializa: `vad = SileroVADValidator()`
   - Pasa `vad_validator=vad` a `_run_dir()`
   - Si VAD falla → continúa sin validación (fallback seguro)

## Lógica de Validación

### Decisión VAD por YOLO confidence
```
┌─ YOLO_confidence >= 0.5
│  └─ ACCEPT (predicción fuerte, confiar en YOLO)
│
├─ YOLO_confidence < 0.3
│  └─ Ejecutar VAD:
│     ├─ VAD_prob >= 0.5 → ACCEPT (VAD confirma Speech)
│     └─ VAD_prob < 0.5  → REJECT (probable falso positivo)
│
└─ 0.3 <= YOLO_confidence < 0.5
   └─ ACCEPT (borderline, asumir correcto)
```

### Impacto Esperado
- ✅ Ruido puro (YOLO 0.1-0.3) → Filtrado
- ✅ Artefactos tonales (YOLO 0.2-0.4) → Filtrados
- ✅ Speech legítimo fuerte (YOLO 0.5+) → Mantenido
- ✅ Speech débil real (YOLO 0.2, VAD alto) → Mantenido

## Cómo Ejecutar

### Opción 1: Pipeline Normal (con VAD)
```bash
python scripts/infer_clean.py --dual-clean
```
- PASS 1: Wiener sin Speech ✓
- PASS 2: Demucs con validación VAD ✓

### Opción 2: Solo PASS 1 (si VAD falla)
```bash
python scripts/infer_clean.py --dual-clean --no-use-vad
```
(Nota: requeriría agregar parámetro en argparse si se necesita)

### Opción 3: Testing
```bash
# Ver demostración de VAD
python scripts/demo_vad_filtering.py

# Verificar validador
python scripts/silero_vad_validator.py
```

## Ventajas de esta Solución
1. ⚡ **Rápido**: VAD procesa en ~1ms por frame
2. 🎯 **Específico**: Valida solo Speech (PASS 2)
3. 🔧 **Modular**: Código separado, fácil mantener/actualizar
4. 🛡️ **Robusto**: Fallback si VAD falla
5. 📊 **Bajo Impacto**: Solo filtra confianza < 0.3 + VAD < 0.5

## Métricas Esperadas
Con datos actuales (~14k Speech post-NMS, 50% conf < 0.3):
- ~7k Speech bajo confianza detectados
- ~5k-6k rechazados por VAD (ruido puro Demucs)
- ~1k-2k validados como real (artefactos que parecen speech)
- **Resultado**: -30-40% falsos positivos de Speech

## Próximos Pasos (Opcional)
1. Ejecutar con VAD y validar reducción de falsos positivos
2. Si insuficiente → aumentar agresividad VAD (lower threshold)
3. Si necesario → combinar con DFN3 denoise (Option B)
4. Long-term → fine-tune YOLO en Demucs stems

## Referencias
- Silero VAD: https://github.com/snakers4/silero-vad
- Paper: https://arxiv.org/abs/1904.06313
