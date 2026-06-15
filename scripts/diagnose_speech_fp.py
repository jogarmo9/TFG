#!/usr/bin/env python3
"""
diagnose_speech_fp.py
====================
Diagnóstico rápido de falsos positivos de Speech en el pipeline ETL.

Compara predicciones en cada etapa:
  1. predictions_mic.parquet (raw → YOLO pre-GPS)
  2. predictions_geo.parquet (post-GPS join)
  3. predicciones_clean.csv (entrada a prepare_mic.py)

Objetivo: Identificar dónde se pierden predicciones y si Speech es filtrado erróneamente.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"

CLASS_MAP = {
    0: "Horn", 1: "Siren", 2: "Pets", 3: "Physiological",
    4: "Speech", 5: "Ring Tone", 6: "Vibrating", 7: "Notifications", 8: "Cry"
}

def load_data():
    """Carga los tres datasets clave."""
    files_needed = {
        'predictions_mic.parquet': 'predictions_mic (raw pre-GPS)',
        'predictions_geo.parquet': 'predictions_geo (post-GPS)',
        'predicciones_clean.csv': 'predicciones_clean.csv (origen del ETL)',
    }
    
    missing = []
    for fname, desc in files_needed.items():
        path = DATA_DIR / fname
        if not path.exists():
            missing.append(f"  ❌ {fname} ({desc})")
    
    if missing:
        print("⚠️  Archivos FALTANTES:")
        for m in missing:
            print(m)
        print("\nEjecuta primero:")
        print("  python scripts/prepare_mic.py  (genera predictions_geo + predictions_mic)")
        print("  python scripts/infer_clean.py --dual-clean  (genera predicciones_clean.csv)")
        return None, None, None
    
    print("📂 Cargando datasets...")
    pred_mic_raw = pd.read_parquet(DATA_DIR / 'predictions_mic.parquet')
    pred_geo = pd.read_parquet(DATA_DIR / 'predictions_geo.parquet')
    pred_clean_csv = pd.read_csv(DATA_DIR / 'predicciones_clean.csv')
    
    print(f"  ✓ predictions_mic.parquet   : {len(pred_mic_raw)} filas")
    print(f"  ✓ predictions_geo.parquet   : {len(pred_geo)} filas")
    print(f"  ✓ predicciones_clean.csv    : {len(pred_clean_csv)} filas")
    
    return pred_mic_raw, pred_geo, pred_clean_csv

def audit_pipeline(pred_mic_raw, pred_geo, pred_clean_csv):
    """Realiza audit de pérdida de datos en el pipeline."""
    
    print("\n" + "="*80)
    print("AUDIT DE PIPELINE: PÉRDIDA DE PREDICCIONES")
    print("="*80)
    
    # Agregar nombres de clase
    pred_mic_raw['class_name'] = pred_mic_raw['class'].map(CLASS_MAP)
    pred_geo['class_name'] = pred_geo['class'].map(CLASS_MAP)
    pred_clean_csv['class_name'] = pred_clean_csv['class_id'].map(CLASS_MAP)
    
    # ─── STAGE 1: Raw (predicciones_mic.parquet) ─────────────────────────────
    stage1_total = len(pred_mic_raw)
    stage1_speech = len(pred_mic_raw[pred_mic_raw['class'] == 4])
    
    print("\n[STAGE 1] predicciones_mic.parquet (Raw pre-GPS):")
    print(f"  Total          : {stage1_total:,} predicciones")
    print(f"  Speech         : {stage1_speech:,} ({stage1_speech/stage1_total*100:.1f}%)")
    print(f"  Outras clases  : {stage1_total - stage1_speech:,}")
    
    # ─── STAGE 2: After GPS join (predictions_geo.parquet) ──────────────────
    stage2_total = len(pred_geo)
    stage2_speech = len(pred_geo[pred_geo['class'] == 4])
    
    loss_abs = stage1_total - stage2_total
    loss_pct = loss_abs / stage1_total * 100
    
    print(f"\n[STAGE 2] predictions_geo.parquet (Post-GPS join):")
    print(f"  Total          : {stage2_total:,} predicciones")
    print(f"  Speech         : {stage2_speech:,} ({stage2_speech/stage2_total*100:.1f}%)")
    print(f"  PÉRDIDA        : {loss_abs:,} ({loss_pct:.1f}%) ← GPS filter")
    
    # Analizar pérdida de Speech vs otras clases
    speech_loss = stage1_speech - stage2_speech
    other_loss = (stage1_total - stage1_speech) - (stage2_total - stage2_speech)
    speech_loss_pct = speech_loss / stage1_speech * 100 if stage1_speech > 0 else 0
    other_loss_pct = other_loss / (stage1_total - stage1_speech) * 100 if (stage1_total - stage1_speech) > 0 else 0
    
    print(f"\n  [Desglose de PÉRDIDA por clase]")
    print(f"    Speech       : {speech_loss:,} perdidas ({speech_loss_pct:.1f}% del Speech raw)")
    print(f"    Otros        : {other_loss:,} perdidas ({other_loss_pct:.1f}% del resto raw)")
    
    if speech_loss_pct > other_loss_pct + 5:
        print(f"    ⚠️  ALERT: Speech se pierde MÁS que otras clases (Δ = {speech_loss_pct - other_loss_pct:.1f}%)")
        print(f"       → Revisa si el GPS join es discriminatorio con Speech")
    
    # ─── STAGE 3: Final CSV (predicciones_clean.csv) ──────────────────────────
    stage3_total = len(pred_clean_csv)
    stage3_speech = len(pred_clean_csv[pred_clean_csv['class_id'] == 4])
    
    loss2_abs = stage2_total - stage3_total
    loss2_pct = loss2_abs / stage2_total * 100 if stage2_total > 0 else 0
    
    print(f"\n[STAGE 3] predicciones_clean.csv (Final output):")
    print(f"  Total          : {stage3_total:,} predicciones")
    print(f"  Speech         : {stage3_speech:,} ({stage3_speech/stage3_total*100:.1f}%)")
    print(f"  PÉRDIDA        : {loss2_abs:,} ({loss2_pct:.1f}%) ← ??? (debería ser 0)")
    
    if loss2_abs > 0:
        print(f"    ⚠️  ALERT: Se pierden {loss2_abs} predicciones entre STAGE2→3")
        print(f"       → Esto NO debería suceder (son los mismos datos)")
        print(f"       → Revisa: ¿se está regenerando el CSV? ¿hay filtros aplicados?")
        
        speech_loss2 = stage2_speech - stage3_speech
        if speech_loss2 > 0:
            print(f"       → {speech_loss2} Speech perdidas ← CRÍTICO, revisar infer_clean.py")
    
    # ─── SUMMARY ──────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("RESUMEN")
    print("="*80)
    
    total_loss_abs = stage1_total - stage3_total
    total_loss_pct = total_loss_abs / stage1_total * 100 if stage1_total > 0 else 0
    
    print(f"\nSpeech detectado:")
    print(f"  Inicio (Stage 1) : {stage1_speech:,}")
    print(f"  Final (Stage 3)  : {stage3_speech:,}")
    print(f"  Pérdida total    : {stage1_speech - stage3_speech:,} ({(stage1_speech - stage3_speech)/stage1_speech*100:.1f}%)")
    
    print(f"\nDataset completo:")
    print(f"  Inicio (Stage 1) : {stage1_total:,} predicciones")
    print(f"  Final (Stage 3)  : {stage3_total:,} predicciones")
    print(f"  Pérdida total    : {total_loss_abs:,} ({total_loss_pct:.1f}%)")
    
    # ─── RECOMENDACIONES ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("RECOMENDACIONES")
    print("="*80)
    
    if loss2_abs == 0:
        print("✓ OK: Los datos entre pred_geo y predicciones_clean.csv son consistentes")
    else:
        print(f"❌ PROBLEMA: Se pierden {loss2_abs} predicciones entre pred_geo → CSV final")
        print("   → Revisa si hay filtros aplicados en prepare_mic.py o infer_clean.py")
    
    if speech_loss_pct > other_loss_pct + 5:
        print(f"❌ Speech se pierde MÁS ({speech_loss_pct:.1f}%) que otras clases ({other_loss_pct:.1f}%)")
        print("   → GPS join podría estar sesgado contra Speech")
        print("   → O bien: Demucs genera Speech en chunks sin voz real → bajo GPS matching")
    else:
        print(f"✓ OK: Pérdida de Speech ({speech_loss_pct:.1f}%) es similar a otras clases ({other_loss_pct:.1f}%)")
    
    # ─── DISTRIBUCIÓN POR MIC ─────────────────────────────────────────────────
    print("\n" + "="*80)
    print("ANÁLISIS POR MICRÓFONO")
    print("="*80)
    
    if 'microfono_id' in pred_geo.columns:
        print(f"\nSpeech por micrófono (post-GPS):")
        mic_speech = pred_geo[pred_geo['class'] == 4].groupby('microfono_id').size()
        for mic_id, count in sorted(mic_speech.items()):
            total_mic = len(pred_geo[pred_geo['microfono_id'] == mic_id])
            pct = count / total_mic * 100 if total_mic > 0 else 0
            print(f"  M{mic_id}: {count:6,} Speech / {total_mic:6,} total ({pct:5.1f}%)")
    
    # ─── DISTRIBUCIÓN POR FECHA ───────────────────────────────────────────────
    print(f"\nSpeech por fecha (post-GPS):")
    if 'date' in pred_geo.columns:
        date_speech = pred_geo[pred_geo['class'] == 4].groupby('date').size()
        for date, count in sorted(date_speech.items(), reverse=True)[:10]:
            total_date = len(pred_geo[pred_geo['date'] == date])
            pct = count / total_date * 100 if total_date > 0 else 0
            print(f"  {date}: {count:6,} Speech / {total_date:6,} total ({pct:5.1f}%)")
    
    # ─── CONFIANZA ─────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("DISTRIBUCIÓN DE CONFIANZA - SPEECH")
    print("="*80)
    
    speech_geo = pred_geo[pred_geo['class'] == 4]
    if len(speech_geo) > 0:
        print(f"\nEstadísticas de confianza Speech (pred_geo):")
        print(f"  Media     : {speech_geo['confidence'].mean():.3f}")
        print(f"  Mediana   : {speech_geo['confidence'].median():.3f}")
        print(f"  Mín       : {speech_geo['confidence'].min():.3f}")
        print(f"  Máx       : {speech_geo['confidence'].max():.3f}")
        
        ranges = [(0, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
        print(f"\n  Rango de confianza:")
        for lo, hi in ranges:
            count = len(speech_geo[(speech_geo['confidence'] >= lo) & (speech_geo['confidence'] < hi)])
            pct = count / len(speech_geo) * 100
            print(f"    [{lo:.1f}-{hi:.1f}) : {count:6,} ({pct:5.1f}%)")
    
    print("\n✅ Diagnóstico completado.\n")

def main():
    pred_mic_raw, pred_geo, pred_clean_csv = load_data()
    
    if any(x is None for x in [pred_mic_raw, pred_geo, pred_clean_csv]):
        sys.exit(1)
    
    audit_pipeline(pred_mic_raw, pred_geo, pred_clean_csv)

if __name__ == "__main__":
    main()
