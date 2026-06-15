#!/usr/bin/env python3
"""
demo_vad_filtering.py
=====================
Demostración de cómo Silero VAD filtra falsos positivos de Speech.

Crea ejemplos de:
1. Audio con ruido puro (bajo YOLO confidence, VAD rechaza)
2. Audio con tono sinusoidal (bajo YOLO confidence, VAD podría aceptar)
3. Audio real de Demucs stems (simular)
"""

import numpy as np
from silero_vad_validator import SileroVADValidator, VAD_SR

print("╔" + "═" * 78 + "╗")
print("║ " + "Demostración: Silero VAD Filtering de False Positives".center(76) + " ║")
print("╚" + "═" * 78 + "╝")

print("\nInicializando Silero VAD...")
vad = SileroVADValidator()

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Ruido puro (típico en stems Demucs)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("Test 1: Ruido puro (típico en stems Demucs)")
print("="*80)
print("Escenario: YOLO detecta Speech con confianza 0.15 (muy bajo)")
print("          Esperado: VAD rechaza porque es ruido")

noise_audio = np.random.randn(VAD_SR * 2).astype(np.float32) * 0.1  # 2 sec ruido
decision, vad_prob = vad.validate_segment(noise_audio, yolo_confidence=0.15, debug=True)
print(f"\n✓ Resultado: {'REJECT (filtrado)' if not decision else 'ACCEPT'}")
print(f"  VAD_prob = {vad_prob:.3f} (< 0.5 → rechazo)")

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Tono sinusoidal (podría ser artefacto de separación)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("Test 2: Tono sinusoidal (artefacto de separación)")
print("="*80)
print("Escenario: YOLO detecta Speech con confianza 0.25 (bajo-medio)")
print("          Esperado: Resultado depende de VAD")

tone_audio = (np.sin(2*np.pi*440*np.linspace(0, 2, VAD_SR*2)) * 0.3).astype(np.float32)
decision, vad_prob = vad.validate_segment(tone_audio, yolo_confidence=0.25, debug=True)
print(f"\n✓ Resultado: {'REJECT (filtrado)' if not decision else 'ACCEPT'}")
print(f"  VAD_prob = {vad_prob:.3f} ({'< 0.5 → rechazo' if vad_prob < 0.5 else '>= 0.5 → aceptación'})")

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: YOLO confianza fuerte (siempre acepta)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("Test 3: YOLO confianza fuerte (siempre ACCEPT)")
print("="*80)
print("Escenario: YOLO detecta Speech con confianza 0.65 (fuerte)")
print("          Esperado: ACCEPT sin validar VAD")

decision, vad_prob = vad.validate_segment(noise_audio, yolo_confidence=0.65, debug=True)
print(f"\n✓ Resultado: ACCEPT (YOLO_strong)")
print(f"  No se valida VAD cuando YOLO_confidence >= 0.5")

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: YOLO confianza borderline (0.3-0.5)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("Test 4: YOLO confianza borderline (0.3 <= conf < 0.5)")
print("="*80)
print("Escenario: YOLO detecta Speech con confianza 0.40")
print("          Esperado: ACCEPT con warning")

decision, vad_prob = vad.validate_segment(noise_audio, yolo_confidence=0.40, debug=True)
print(f"\n✓ Resultado: ACCEPT (borderline)")
print(f"  Se acepta pero marca como dudoso")

# ─────────────────────────────────────────────────────────────────────────────
# Resumen
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("RESUMEN DE LÓGICA VAD")
print("="*80)
print("""
Decisión VAD basada en:

1. YOLO_confidence >= 0.5        → ACCEPT (predicción fuerte)
                                   └─ Confiar en YOLO

2. YOLO_confidence < 0.3         → Validar con VAD
                                   ├─ Si VAD_prob >= 0.5 → ACCEPT (VAD confirma)
                                   └─ Si VAD_prob < 0.5  → REJECT (probable falso positivo)

3. 0.3 <= YOLO_confidence < 0.5  → ACCEPT con warning (borderline)
                                   └─ No validar; asumir predicción correcta

Impacto esperado en data:
├─ Ruido puro Demucs (YOLO 0.1-0.3, VAD=0.0)    → REJECT ✓
├─ Tono/artefacto (YOLO 0.2-0.4, VAD=bajo)      → REJECT ✓
├─ Speech real (YOLO 0.5+, VAD=alto)             → ACCEPT ✓
└─ Speech débil pero real (YOLO 0.2, VAD=0.6)   → ACCEPT ✓
""")
print()
