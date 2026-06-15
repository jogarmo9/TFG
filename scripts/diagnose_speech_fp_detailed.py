#!/usr/bin/env python3
"""
diagnose_speech_fp_detailed.py
==============================
Análisis DETALLADO del pipeline de Speech. 

Explora:
1. Diferencia entre predictions_mic (raw) y predictions_clean.csv
2. Impacto del cross-NMS entre micrófonos
3. Validar si prepare_mic.py está filtrando erróneamente
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

def analyze_mic_processing():
    """Analiza qué pasa entre predictions_clean.csv y predictions_mic.parquet."""
    
    print("\n" + "="*80)
    print("ANÁLISIS: Transformación predictions_clean.csv → predictions_mic.parquet")
    print("="*80)
    
    # Cargar
    if not (DATA_DIR / 'predictions_mic.parquet').exists():
        print("❌ predictions_mic.parquet no encontrado")
        return
    
    if not (DATA_DIR / 'predicciones_clean.csv').exists():
        print("❌ predicciones_clean.csv no encontrado")
        return
    
    pred_clean = pd.read_csv(DATA_DIR / 'predicciones_clean.csv')
    pred_mic = pd.read_parquet(DATA_DIR / 'predictions_mic.parquet')
    
    pred_clean['class_name'] = pred_clean['class_id'].map(CLASS_MAP)
    pred_mic['class_name'] = pred_mic['class'].map(CLASS_MAP)
    
    print(f"\npredicciones_clean.csv (entrada prepare_mic.py):")
    print(f"  Total       : {len(pred_clean):,}")
    print(f"  Speech      : {len(pred_clean[pred_clean['class_id']==4]):,} ({len(pred_clean[pred_clean['class_id']==4])/len(pred_clean)*100:.1f}%)")
    
    # Solo extraer mic de predictions_clean
    pred_clean_mic = pred_clean[pred_clean['source'] == 'mic'].copy()
    print(f"  Solo MIC    : {len(pred_clean_mic):,}")
    print(f"  Mic Speech  : {len(pred_clean_mic[pred_clean_mic['class_id']==4]):,}")
    
    print(f"\npredictions_mic.parquet (salida prepare_mic.py):")
    print(f"  Total       : {len(pred_mic):,}")
    print(f"  Speech      : {len(pred_mic[pred_mic['class']==4]):,} ({len(pred_mic[pred_mic['class']==4])/len(pred_mic)*100:.1f}%)")
    
    # Diferencia
    diff_total = len(pred_clean_mic) - len(pred_mic)
    diff_speech = len(pred_clean_mic[pred_clean_mic['class_id']==4]) - len(pred_mic[pred_mic['class']==4])
    
    print(f"\nDIFERENCIA (Clean → Mic):")
    print(f"  Total perdido    : {diff_total:,} ({diff_total/len(pred_clean_mic)*100:.1f}%)")
    print(f"  Speech perdido    : {diff_speech:,} ({diff_speech/len(pred_clean_mic[pred_clean_mic['class_id']==4])*100:.1f}% del Speech)")
    
    if diff_total > 0:
        print(f"\n  Causas posibles:")
        print(f"    1. Cross-NMS entre M1 y M2 (elimina duplicados)")
        print(f"    2. Filtro de confianza aplicado en prepare_mic.py")
        print(f"    3. Time corrections aplicadas durante conversión de zonas horarias")
        print(f"    4. Filtro de duración (duration_s < threshold)")
    
    # ─── Analizar por micrófono ──────────────────────────────────────────────
    print(f"\nPor micrófono (predictions_mic):")
    for mic_id in sorted(pred_mic['microfono_id'].unique()):
        mic_data = pred_mic[pred_mic['microfono_id'] == mic_id]
        mic_speech = len(mic_data[mic_data['class'] == 4])
        print(f"  M{mic_id}: {len(mic_data):6,} total | {mic_speech:5,} Speech ({mic_speech/len(mic_data)*100:5.1f}%)")
    
    # ─── Estadísticas de confianza en predicciones_clean (MIC) ────────────────
    print(f"\nConfianza SPEECH en predictions_clean.csv (MIC input):")
    clean_mic_speech = pred_clean_mic[pred_clean_mic['class_id'] == 4]
    if len(clean_mic_speech) > 0:
        print(f"  Media       : {clean_mic_speech['confidence'].mean():.3f}")
        print(f"  < 0.3       : {len(clean_mic_speech[clean_mic_speech['confidence']<0.3]):,} ({len(clean_mic_speech[clean_mic_speech['confidence']<0.3])/len(clean_mic_speech)*100:.1f}%)")
        print(f"  >= 0.5      : {len(clean_mic_speech[clean_mic_speech['confidence']>=0.5]):,} ({len(clean_mic_speech[clean_mic_speech['confidence']>=0.5])/len(clean_mic_speech)*100:.1f}%)")
    
    # ─── Estadísticas de confianza en predictions_mic (MIC output) ──────────
    print(f"\nConfianza SPEECH en predictions_mic.parquet (MIC output):")
    mic_speech_data = pred_mic[pred_mic['class'] == 4]
    if len(mic_speech_data) > 0:
        print(f"  Media       : {mic_speech_data['confidence'].mean():.3f}")
        print(f"  < 0.3       : {len(mic_speech_data[mic_speech_data['confidence']<0.3]):,} ({len(mic_speech_data[mic_speech_data['confidence']<0.3])/len(mic_speech_data)*100:.1f}%)")
        print(f"  >= 0.5      : {len(mic_speech_data[mic_speech_data['confidence']>=0.5]):,} ({len(mic_speech_data[mic_speech_data['confidence']>=0.5])/len(mic_speech_data)*100:.1f}%)")

def analyze_speech_raw_vs_nms():
    """Compara speech en predictions_mic.parquet (NMS) vs predictions_mic_raw.parquet (sin NMS)."""
    
    print("\n" + "="*80)
    print("ANÁLISIS: Impacto de NMS y Cross-NMS en Speech")
    print("="*80)
    
    if not (DATA_DIR / 'predictions_mic_raw.parquet').exists():
        print("⚠️  predictions_mic_raw.parquet no encontrado (no se puede comparar NMS impact)")
        print("   Esta sería la versión SIN deduplicación M1↔M2")
        return
    
    pred_mic_raw = pd.read_parquet(DATA_DIR / 'predictions_mic_raw.parquet')
    pred_mic_nms = pd.read_parquet(DATA_DIR / 'predictions_mic.parquet')
    
    print(f"\npredictions_mic_raw.parquet (ANTES de NMS/dedup):")
    print(f"  Total       : {len(pred_mic_raw):,}")
    print(f"  Speech      : {len(pred_mic_raw[pred_mic_raw['class']==4]):,}")
    
    print(f"\npredictions_mic.parquet (DESPUÉS de NMS/dedup):")
    print(f"  Total       : {len(pred_mic_nms):,}")
    print(f"  Speech      : {len(pred_mic_nms[pred_mic_nms['class']==4]):,}")
    
    nms_loss = len(pred_mic_raw) - len(pred_mic_nms)
    speech_loss = len(pred_mic_raw[pred_mic_raw['class']==4]) - len(pred_mic_nms[pred_mic_nms['class']==4])
    
    print(f"\nPérdida por NMS:")
    print(f"  Total       : {nms_loss:,} ({nms_loss/len(pred_mic_raw)*100:.1f}%)")
    print(f"  Speech      : {speech_loss:,} ({speech_loss/len(pred_mic_raw[pred_mic_raw['class']==4])*100:.1f}%)")

def compare_trayectos():
    """Compara cantidad de predicciones por trayecto (especialmente Speech)."""
    
    print("\n" + "="*80)
    print("ANÁLISIS: Speech por Trayecto")
    print("="*80)
    
    if not (DATA_DIR / 'predictions_geo.parquet').exists():
        print("❌ predictions_geo.parquet no encontrado")
        return
    
    pred_geo = pd.read_parquet(DATA_DIR / 'predictions_geo.parquet')
    
    print(f"\nTrayectos con Speech (top 15):")
    traj_speech = pred_geo[pred_geo['class'] == 4].groupby('trayecto').size().sort_values(ascending=False)
    
    for traj, count in traj_speech.head(15).items():
        total = len(pred_geo[pred_geo['trayecto'] == traj])
        pct = count / total * 100 if total > 0 else 0
        src = pred_geo[pred_geo['trayecto'] == traj]['source'].iloc[0] if len(pred_geo[pred_geo['trayecto'] == traj]) > 0 else '?'
        print(f"  {traj:40s} [{src:6s}]: {count:5,} / {total:5,} ({pct:5.1f}%)")

def main():
    analyze_mic_processing()
    analyze_speech_raw_vs_nms()
    compare_trayectos()
    print("\n✅ Análisis detallado completado.\n")

if __name__ == "__main__":
    main()
