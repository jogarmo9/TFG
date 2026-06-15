#!/usr/bin/env python3
"""
Analizar distribución de Ring Tone y Physiological en Mic pipeline
"""

import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

csv_path = Path("data/processed/predicciones_clean.csv")

if not csv_path.exists():
    print(f"❌ {csv_path} no encontrado")
    exit(1)

# Leer CSV
rows = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"✓ Cargadas {len(rows):,} predicciones")

# Filtrar por clases de interés
CLASS_NAMES = {
    3: "Physiological",
    5: "Ring Tone",
}

class_data = {cls_id: [] for cls_id in CLASS_NAMES.keys()}

for row in rows:
    try:
        cls_id = int(float(row["class_id"]))
        conf = float(row["confidence"])
        
        if cls_id in CLASS_NAMES:
            class_data[cls_id].append({
                "conf": conf,
                "source": row.get("source", "unknown"),
                "raw": row
            })
    except:
        pass

print(f"\n{'═'*80}")
print("DISTRIBUCIÓN DE RING TONE Y PHYSIOLOGICAL")
print(f"{'═'*80}\n")

for cls_id, cls_name in sorted(CLASS_NAMES.items()):
    data = class_data[cls_id]
    count = len(data)
    
    if count == 0:
        print(f"❌ {cls_name}: 0 predicciones")
        continue
    
    confs = [d["conf"] for d in data]
    
    print(f"📊 {cls_name} (Clase {cls_id}): {count:,} predicciones")
    print(f"   Confianza: min={min(confs):.3f}, mean={np.mean(confs):.3f}, max={max(confs):.3f}")
    print(f"   Mediana: {np.median(confs):.3f}, Std: {np.std(confs):.3f}")
    
    # Bins de confianza
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(confs, bins=bins)
    
    print(f"\n   Distribución por confianza:")
    for i in range(len(bins)-1):
        bin_start, bin_end = bins[i], bins[i+1]
        count_in_bin = hist[i]
        pct = (count_in_bin / len(confs)) * 100
        bar = "█" * int(pct / 2)
        print(f"     {bin_start:.1f}-{bin_end:.1f}: {count_in_bin:>5} ({pct:>5.1f}%) {bar}")
    
    # Source
    source_counts = defaultdict(int)
    for d in data:
        source_counts[d["source"]] += 1
    
    print(f"\n   Por source:")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"     {src}: {cnt:,} ({(cnt/count)*100:.1f}%)")
    
    print()

# Análisis cruzado
print(f"{'═'*80}")
print("ANÁLISIS CRUZADO")
print(f"{'═'*80}\n")

ringtone_data = class_data[5]
physio_data = class_data[3]

print(f"Ring Tone vs Physiological:")
print(f"  Ring Tone:      {len(ringtone_data):>6,} predicciones")
print(f"  Physiological:  {len(physio_data):>6,} predicciones")
print(f"  Ratio: 1:{len(physio_data)/max(1, len(ringtone_data)):.2f}")

# Confianza promedio
if len(ringtone_data) > 0:
    rt_conf = np.mean([d["conf"] for d in ringtone_data])
    print(f"\n  Ring Tone conf media:      {rt_conf:.3f}")
else:
    rt_conf = 0
    
if len(physio_data) > 0:
    ph_conf = np.mean([d["conf"] for d in physio_data])
    print(f"  Physiological conf media:  {ph_conf:.3f}")
else:
    ph_conf = 0

print(f"\n  Diferencia: {abs(rt_conf - ph_conf):.3f}")

# Tabla comparativa
print(f"\n{'─'*80}")
print("TABLA COMPARATIVA")
print(f"{'─'*80}\n")

print(f"{'Clase':<20} {'Count':>10} {'Min Conf':>10} {'Mean Conf':>10} {'Max Conf':>10}")
print(f"{'-'*60}")

for cls_id in [3, 5]:
    data = class_data[cls_id]
    if len(data) > 0:
        confs = [d["conf"] for d in data]
        print(f"{CLASS_NAMES[cls_id]:<20} {len(data):>10,} {min(confs):>10.3f} {np.mean(confs):>10.3f} {max(confs):>10.3f}")

print(f"\n{'═'*80}")
print("🔍 INTERPRETACIÓN")
print(f"{'═'*80}\n")

print("""
Ring Tone (Clase 5):
  • Debería detectar: Tonos de llamada, notificaciones, alarmas
  • Posibles falsos positivos: Tonos frecuentes, bips de aparatos
  • Esperado en Mic: MEDIO-BAJO (ambiente móvil ocasional)

Physiological (Clase 3):
  • Debería detectar: Tos, estornudo, respiración fuerte, jadeos
  • Posibles falsos positivos: Ruidos de motor, fricción
  • Esperado en Mic: BAJO-MUY BAJO (aunque hay personas dentro del coche)
  
Vibraciones → ¿Qué clase?: Probablemente "Physiological" o "Vibrating" (6)
Intermitentes → ¿Qué clase?: Probablemente "Ring Tone" (5) o "Notifications" (7)
""")

print()
