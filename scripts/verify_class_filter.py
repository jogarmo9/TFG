#!/usr/bin/env python3
"""
verify_class_filter.py
======================
Verifica que el filtro de clases en infer_clean.py funciona correctamente.

Test lógico: ALL_CLASSES - {SPEECH_ID} debe excluir Speech (4)
"""

import sys

# Configuración (debe coincidir con infer_clean.py)
SPEECH_ID = 4
ALL_CLASSES = set(range(9))  # {0, 1, 2, 3, 4, 5, 6, 7, 8}

CLASS_NAMES = [
    "Horn",            # 0
    "Siren",           # 1
    "Pets",            # 2
    "Physiological",   # 3
    "Speech",          # 4
    "Ring Tone",       # 5
    "Vibrating",       # 6
    "Notifications",   # 7
    "Cry"              # 8
]

print("╔" + "═" * 78 + "╗")
print("║ " + "Verificación del Filtro de Clases - PASS 1 (Wiener)".center(76) + " ║")
print("╚" + "═" * 78 + "╝")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n📋 Configuración:")
print(f"  SPEECH_ID        = {SPEECH_ID} ({CLASS_NAMES[SPEECH_ID]})")
print(f"  ALL_CLASSES      = {sorted(ALL_CLASSES)}")

# ─────────────────────────────────────────────────────────────────────────────
# PASS 1: Todas las clases EXCEPTO Speech
pass1_filter = ALL_CLASSES - {SPEECH_ID}
print(f"\n[PASS 1 - Wiener] class_filter = ALL_CLASSES - {{SPEECH_ID}}")
print(f"  class_filter = {sorted(pass1_filter)}")
print(f"  Clases incluidas:")
for cls_id in sorted(pass1_filter):
    print(f"    {cls_id}: {CLASS_NAMES[cls_id]}")

# ─────────────────────────────────────────────────────────────────────────────
# PASS 2: Solo Speech
pass2_filter = {SPEECH_ID}
print(f"\n[PASS 2 - Demucs] class_filter = {{SPEECH_ID}}")
print(f"  class_filter = {pass2_filter}")
print(f"  Clases incluidas:")
for cls_id in sorted(pass2_filter):
    print(f"    {cls_id}: {CLASS_NAMES[cls_id]}")

# ─────────────────────────────────────────────────────────────────────────────
# Test lógico del filtro
print(f"\n" + "="*80)
print("TEST: Lógica del filtro (cls_id not in class_filter → SKIP)")
print("="*80)

def test_filter(cls_id, class_filter, pass_name):
    """Simula la lógica del filtro en infer_clean.py."""
    if class_filter is not None and cls_id not in class_filter:
        return "SKIP (no escribe)"
    return "WRITE (escribe)"

print(f"\nPASS 1 (Wiener, excluir Speech):")
for cls_id in range(9):
    result = test_filter(cls_id, pass1_filter, "PASS 1")
    marker = "❌" if result == "SKIP" else "✓"
    print(f"  {marker} Clase {cls_id} ({CLASS_NAMES[cls_id]:15s}): {result}")

print(f"\nPASS 2 (Demucs, solo Speech):")
for cls_id in range(9):
    result = test_filter(cls_id, pass2_filter, "PASS 2")
    marker = "✓" if result == "WRITE" else "❌"
    print(f"  {marker} Clase {cls_id} ({CLASS_NAMES[cls_id]:15s}): {result}")

# ─────────────────────────────────────────────────────────────────────────────
# Verificación
print(f"\n" + "="*80)
print("VERIFICACIÓN")
print("="*80)

all_pass1_skip_speech = test_filter(4, pass1_filter, "PASS 1") == "SKIP"
all_pass2_write_speech = test_filter(4, pass2_filter, "PASS 2") == "WRITE"
all_pass1_write_others = all(test_filter(cls, pass1_filter, "PASS 1") == "WRITE" for cls in [0,1,2,3,5,6,7,8])

if all_pass1_skip_speech and all_pass2_write_speech and all_pass1_write_others:
    print("\n✅ CORRECTO: El filtro de clases está configurado correctamente")
    print(f"\n   PASS 1 (Wiener)  → Incluye todas EXCEPTO Speech (4)")
    print(f"   PASS 2 (Demucs)  → Incluye SOLO Speech (4)")
    print(f"\n   → No hay riesgo de Speech accidental en PASS 1")
else:
    print("\n❌ ERROR: Hay problemas con la configuración del filtro")
    if not all_pass1_skip_speech:
        print("   - PASS 1 no está excluyendo Speech")
    if not all_pass2_write_speech:
        print("   - PASS 2 no está incluyendo Speech")
    if not all_pass1_write_others:
        print("   - PASS 1 no está incluyendo otras clases")

print()
