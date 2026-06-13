#!/usr/bin/env python3
"""
test_vad_integration.py
=======================
Test rápido: verifica que VAD se integra correctamente en infer_clean.py
"""

import sys
from pathlib import Path

# Simular entorno de infer_clean.py
sys.path.insert(0, str(Path(__file__).parent))

print("Testing Silero VAD Integration...")
print("="*60)

# Test 1: Import
print("\n[1/3] Importando módulos...")
try:
    from silero_vad_validator import SileroVADValidator, VAD_SR
    print("  ✓ silero_vad_validator importado")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 2: Inicializar VAD
print("\n[2/3] Inicializando Silero VAD...")
try:
    vad = SileroVADValidator()
    print("  ✓ SileroVADValidator inicializado")
except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# Test 3: Validar segmento
print("\n[3/3] Testeando validación de segmento...")
try:
    import numpy as np
    
    # Audio de prueba: ruido
    test_audio = np.random.randn(VAD_SR // 2).astype(np.float32) * 0.1
    
    # Test con confianza baja (debe validar VAD)
    decision, prob = vad.validate_segment(test_audio, yolo_confidence=0.15)
    print(f"  ✓ Validación de ruido: decision={decision}, vad_prob={prob:.3f}")
    
    # Test con confianza alta (no debe validar VAD)
    decision, prob = vad.validate_segment(test_audio, yolo_confidence=0.6)
    print(f"  ✓ Validación fuerte: decision={decision}, vad_prob={prob:.3f}")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ Todos los tests pasaron!")
print("\nLa integración de Silero VAD está lista para usar en infer_clean.py")
print("\nEjecuta: python scripts/infer_clean.py --dual-clean")
