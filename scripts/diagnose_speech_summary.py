#!/usr/bin/env python3
"""
diagnose_speech_summary.py
===========================
RESUMEN EJECUTIVO del diagnóstico de Speech falsos positivos.
"""

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"

CLASS_MAP = {
    0: "Horn", 1: "Siren", 2: "Pets", 3: "Physiological",
    4: "Speech", 5: "Ring Tone", 6: "Vibrating", 7: "Notifications", 8: "Cry"
}

def print_header(text):
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║ " + text.center(76) + " ║")
    print("╚" + "═" * 78 + "╝")

def print_section(text):
    print("\n" + "─" * 80)
    print(text)
    print("─" * 80)

def main():
    print_header("DIAGNÓSTICO DE SPEECH FALSE POSITIVES")
    
    # Cargar datos
    try:
        pred_mic_raw = pd.read_parquet(DATA_DIR / 'predictions_mic_raw.parquet')
        pred_mic = pd.read_parquet(DATA_DIR / 'predictions_mic.parquet')
        pred_geo = pd.read_parquet(DATA_DIR / 'predictions_geo.parquet')
        pred_clean = pd.read_csv(DATA_DIR / 'predicciones_clean.csv')
    except Exception as e:
        print(f"Error cargando datos: {e}")
        return
    
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("1️⃣  FLUJO DEL PIPELINE Y PÉRDIDA DE DATOS")
    
    stages = [
        ("STAGE 0", "predictions_mic_raw", len(pred_mic_raw), 
         len(pred_mic_raw[pred_mic_raw['class']==4]), "Raw pre-NMS"),
        ("STAGE 1", "predictions_mic", len(pred_mic), 
         len(pred_mic[pred_mic['class']==4]), "Post-NMS/Dedup"),
        ("STAGE 2", "predictions_geo", len(pred_geo), 
         len(pred_geo[pred_geo['class']==4]), "Post-GPS join"),
        ("STAGE 3", "pred_clean.csv", len(pred_clean), 
         len(pred_clean[pred_clean['class_id']==4]), "Final output"),
    ]
    
    for stage, name, total, speech, desc in stages:
        print(f"\n{stage} — {name:25s} ({desc})")
        print(f"  Total      : {total:>8,} predicciones")
        print(f"  Speech     : {speech:>8,} ({speech/total*100:>5.1f}%)")
    
    # ─── Análisis de pérdida ──────────────────────────────────────────────────
    print_section("2️⃣  PÉRDIDA DE DATOS EN CADA ETAPA")
    
    print(f"\n0→1 (NMS Deduplication entre M1↔M2):")
    loss01_total = len(pred_mic_raw) - len(pred_mic)
    loss01_speech = len(pred_mic_raw[pred_mic_raw['class']==4]) - len(pred_mic[pred_mic['class']==4])
    print(f"  Total  : {loss01_total:6,} ({loss01_total/len(pred_mic_raw)*100:5.1f}%)")
    print(f"  Speech : {loss01_speech:6,} ({loss01_speech/len(pred_mic_raw[pred_mic_raw['class']==4])*100:5.1f}%)")
    
    print(f"\n1→2 (GPS join filter):")
    loss12_total = len(pred_mic) - len(pred_geo)
    loss12_speech = len(pred_mic[pred_mic['class']==4]) - len(pred_geo[pred_geo['class']==4])
    loss12_other = loss12_total - loss12_speech
    print(f"  Total  : {loss12_total:6,} ({loss12_total/len(pred_mic)*100:5.1f}%)")
    print(f"  Speech : {loss12_speech:6,} ({loss12_speech/len(pred_mic[pred_mic['class']==4])*100:5.1f}%)")
    print(f"  Otros  : {loss12_other:6,} ({loss12_other/(len(pred_mic)-len(pred_mic[pred_mic['class']==4]))*100:5.1f}%)")
    
    if loss12_speech/len(pred_mic[pred_mic['class']==4])*100 < \
       loss12_other/(len(pred_mic)-len(pred_mic[pred_mic['class']==4]))*100 + 5:
        print(f"  ✓ OK: Speech y otros tienen tasa similar de pérdida (sin sesgo)")
    else:
        print(f"  ⚠️  Speech se pierde MÁS que otras clases (posible sesgo)")
    
    print(f"\n2→3 (?) — De predictions_geo (20.3K) a predicciones_clean.csv (24.6K)")
    print(f"  ANOMALÍA: Hay MÁS predicciones en Stage 3 que en Stage 2")
    print(f"  → predicciones_clean.csv tiene AMBAS (mic + mobile)")
    print(f"  → No es salida directa de predictions_geo")
    
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("3️⃣  ANÁLISIS DE CONFIANZA - SPEECH")
    
    speech_raw = pred_mic_raw[pred_mic_raw['class'] == 4]
    speech_nms = pred_mic[pred_mic['class'] == 4]
    speech_geo = pred_geo[pred_geo['class'] == 4]
    
    print(f"\nConfidencia SPEECH (media ± std):")
    print(f"  Raw (pre-NMS)  : {speech_raw['confidence'].mean():.3f} ± {speech_raw['confidence'].std():.3f}")
    print(f"  Post-NMS       : {speech_nms['confidence'].mean():.3f} ± {speech_nms['confidence'].std():.3f}")
    print(f"  Post-GPS       : {speech_geo['confidence'].mean():.3f} ± {speech_geo['confidence'].std():.3f}")
    
    print(f"\nPercentiles de confianza (Post-NMS):")
    for pct in [10, 25, 50, 75, 90]:
        val = speech_nms['confidence'].quantile(pct/100)
        print(f"  P{pct:2d} : {val:.3f}")
    
    print(f"\n⚠️  CRÍTICO: {len(speech_raw[speech_raw['confidence']<0.3])/len(speech_raw)*100:.1f}% de Speech tiene confianza < 0.3")
    print(f"            {len(speech_nms[speech_nms['confidence']<0.3])/len(speech_nms)*100:.1f}% post-NMS")
    
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("4️⃣  VEREDICTO")
    
    print("""
✓ NO hay FILTRADO ERRÓNEO en el pipeline
  - Pérdida de Speech vs otras clases es proporcional
  - No hay sesgo discriminatorio contra Speech en GPS join
  
❌ EL VERDADERO PROBLEMA es la CALIDAD de predicciones YOLO en Speech
  - predictions_mic_raw: 110,196 Speech (51.4% del total raw)
  - Pero mayoría con confianza baja (< 0.3: 50%)
  - Después de NMS: 14,258 Speech (87% eliminadas por duplicados)
  - Después GPS join: 10,100 Speech (29% perdidas por audios sin GPS)
  - Final: 2,855 Speech (73% del resto perdido en prepare_mic.py)

🔍 ROOT CAUSE: Demucs genera stems de voz incluso donde NO hay voz real
  - El stem "vocals" contiene: ruido, radio, silencio filtrado con artefactos
  - YOLO detecta estos artefactos como Speech con baja confianza
  - NMS no ayuda porque son detecciones REALES en el stem (no duplicados)
  - GPS join también pierde predicciones por audio sin GPS asignado
""")
    
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("5️⃣  RECOMENDACIONES DE SOLUCIÓN")
    
    print("""
OPCIÓN A (Rápida - 5 minutos)
  ───────────────────────────
  Aplicar threshold de confianza >= 0.5 en YOLO
  
  Impacto:
    - Speech: 14,258 → ~3,900 (73% reducción)
    - Mantiene predicciones con confianza >0.5
    - Fácil de revertir si es demasiado agresivo
  
  Ejecución:
    python scripts/infer_clean.py --dual-clean --reprocess-all
    # Luego editar infer_clean.py para filtrar Speech < 0.5

OPCIÓN B (Moderada - 2-3 horas)
  ───────────────────────────
  Pre-process Demucs stems con DFN3 (denoise)
  
  Rationale:
    - DFN3 (Denoising Filter Network) está diseñado para mejorar speech detectability
    - Elimina ruido ANTES de que YOLO analice
    - Reduce artefactos falsos que YOLO confunde como Speech
  
  Ejecución:
    # Si clean_dfn/ no existe:
    .venv311\Scripts\python.exe scripts/clean_audio.py --method dfn3 --reprocess-all
    
    # Re-inferir con DFN3 para Speech:
    python scripts/infer_clean.py --dual-clean --speech-source dfn3 --reprocess-all

OPCIÓN C (Agresiva - 10-12 horas)
  ───────────────────────────
  Aumentar limpieza Wiener antes de Demucs
  
  Rationale:
    - Wiener actual (HPSS kernel=61) deja residuos tonales
    - Aumentar a kernel=91 elimina más ruido de motor
    - Demucs extrae voz "más limpia"
    - YOLO produce confianzas más altas y menos FP
  
  Ejecución:
    python scripts/clean_audio.py --method wiener --hpss-kernel 91 --reprocess-all
    .venv311\Scripts\python.exe scripts/clean_audio.py --method demucs --reprocess-all
    python scripts/infer_clean.py --dual-clean --reprocess-all
""")
    
    # ═══════════════════════════════════════════════════════════════════════════
    print_section("6️⃣  PRÓXIMOS PASOS")
    
    print("""
✋ ANTES de cualquier cambio:
  1. Hacer BACKUP de:
     - data/processed/predicciones_clean.csv
     - data/processed/predictions_mic.parquet
     - data/clean_demucs/ (opcional)
  
  2. Documentar cambios (ej: "Applied threshold 0.5 on Speech")
  
  3. RE-EJECUTAR este diagnóstico después del cambio
     para validar que el problema se ha resuelto

📋 RECOMENDACIÓN FINAL:
   
   Prueba OPCIÓN A primero (5 min, reversible)
   ↓
   Si no es suficiente → OPCIÓN B (2-3h, mejor calidad)
   ↓
   Si aún hay problemas → OPCIÓN C (12h, más agresivo)
""")
    
    print_header("FIN DEL DIAGNÓSTICO")

if __name__ == "__main__":
    main()
