#!/usr/bin/env python3
"""
test_wiener_pass1_speech.py
===========================
Test aislado: ¿Genera Speech el PASS 1 (Wiener, clases≠Speech)?

Ejecuta YOLO solo sobre data/clean/ (Wiener) sin filtrar Speech
para ver cuántas predicciones de Speech Wiener produce.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import librosa as lb
import onnxruntime as ort
import csv
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN (debe coincidir con infer_clean.py)
# ─────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
MODEL_PATH = ROOT / "models" / "YOLOv5n_original.onnx"
OUTPUT_CSV = ROOT / "data" / "processed" / "test_wiener_pass1.csv"

CONF_THRESH = 0.1
IOU_THRESH = 0.7

SR = 16_000
CHUNK_SEC = 10
CHUNK_SAMP = SR * CHUNK_SEC

N_FFT = 2048
HOP_LENGTH = 256
WIN_LENGTH = 2048
N_MELS = 128
FMIN = 0
FMAX = 8000
MEL_DB_MIN = -50
MEL_DB_MAX = 80
NORM_MIN = -38.5
NORM_MAX = 41.37

YOLO_W = 640
PAD_COLS = 7
PAD_VALUE = 0.447058824

CLASSES = [
    "Horn", "Siren", "Pets", "Physiological",
    "Speech", "Ring Tone", "Vibrating", "Notifications", "Cry"
]

CLASS_MAP = {i: name for i, name in enumerate(CLASSES)}

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTIONS (from infer_clean.py)
# ─────────────────────────────────────────────────────────────────────────────

def wave_to_mel(audio_chunk: np.ndarray) -> np.ndarray:
    """Convert audio waveform to mel spectrogram."""
    S = np.abs(lb.stft(audio_chunk, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       win_length=WIN_LENGTH)) ** 2
    mel = lb.feature.melspectrogram(S=S, sr=SR, n_mels=N_MELS,
                                    fmin=FMIN, fmax=FMAX)
    mel_db = lb.power_to_db(mel, ref=np.max)
    mel_norm = 2 * (mel_db - MEL_DB_MIN) / (MEL_DB_MAX - MEL_DB_MIN) - 1
    mel_norm = np.clip(mel_norm, -1, 1)
    return mel_norm

def mel_to_input(mel: np.ndarray) -> np.ndarray:
    """Convert mel spec to YOLO input tensor."""
    if mel.shape[1] < YOLO_W:
        mel = np.pad(mel, ((0, 0), (PAD_COLS, PAD_COLS + YOLO_W - mel.shape[1])),
                     constant_values=PAD_VALUE)
    else:
        mel = mel[:, :YOLO_W]
    mel = np.pad(mel, ((0, 0), (PAD_COLS, PAD_COLS)), constant_values=PAD_VALUE)
    inp = np.stack([mel, mel, mel], axis=0)
    return inp.astype(np.float32).reshape(1, 3, N_MELS, YOLO_W + 2 * PAD_COLS)

def infer_chunk(session, chunk: np.ndarray, conf_thresh: float) -> list:
    """Returns raw detections: [x1_sec, x2_sec, class_id, confidence]"""
    mel = wave_to_mel(chunk)
    inp = mel_to_input(mel)
    outs = session.run(None, {"images": inp})
    
    preds = outs[0].transpose()[:, :, 0]
    xc = preds[:, 0]
    w = preds[:, 2]
    cls = np.argmax(preds[:, 4:], axis=-1)
    probs = np.max(preds[:, 4:], axis=-1)
    
    inner = YOLO_W - 2 * PAD_COLS
    x1 = np.clip((xc - w / 2 - PAD_COLS) * CHUNK_SEC / inner, 0, CHUNK_SEC)
    x2 = np.clip((xc + w / 2 - PAD_COLS) * CHUNK_SEC / inner, 0, CHUNK_SEC)
    
    boxes = [[float(x1[j]), float(x2[j]), int(cls[j]), float(probs[j])]
             for j in range(len(probs)) if probs[j] >= conf_thresh and x2[j] > x1[j]]
    return boxes

def _iou_1d(b1, b2):
    """IoU between two 1-D intervals [x1, x2]."""
    inter = max(0, min(b1[1], b2[1]) - max(b1[0], b2[0]))
    union = (b1[1] - b1[0]) + (b2[1] - b2[0]) - inter
    return inter / union if union > 0 else 0

def nms(boxes: list, iou_thresh: float) -> list:
    """Greedy NMS."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[3], reverse=True)
    keep = []
    for box in boxes:
        if all(_iou_1d(box, kb) < iou_thresh for kb in keep):
            keep.append(box)
    return keep

# ─────────────────────────────────────────────────────────────────────────────
# TEST: Run YOLO on Wiener WITHOUT filtering Speech
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("╔" + "═" * 78 + "╗")
    print("║ " + "TEST PASS 1 (Wiener): ¿Genera Speech?".center(76) + " ║")
    print("╚" + "═" * 78 + "╝")
    
    # Validar archivos
    if not CLEAN_DIR.exists():
        print(f"❌ {CLEAN_DIR} no encontrado")
        sys.exit(1)
    
    if not MODEL_PATH.exists():
        print(f"❌ {MODEL_PATH} no encontrado")
        sys.exit(1)
    
    # Cargar modelo
    print(f"\n📦 Cargando modelo YOLO...")
    session = ort.InferenceSession(str(MODEL_PATH),
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print(f"  Provider: {session.get_providers()[0]}")
    
    # Procesar archivos WAV en data/clean/
    wav_files = sorted(CLEAN_DIR.glob("*.wav"))
    print(f"\n📂 Encontrados {len(wav_files)} archivos WAV en {CLEAN_DIR.name}/")
    
    if not wav_files:
        print("❌ No hay archivos WAV")
        sys.exit(1)
    
    # CSV output
    output_rows = []
    total_predictions = 0
    total_speech = 0
    speech_by_confidence = {
        'low': 0,      # < 0.3
        'medium': 0,   # 0.3-0.5
        'high': 0      # >= 0.5
    }
    
    print(f"\n🔄 Procesando archivos...")
    for idx, wav_path in enumerate(wav_files, 1):
        if idx % 100 == 0:
            print(f"  [{idx}/{len(wav_files)}]")
        
        try:
            audio, _ = lb.load(str(wav_path), sr=SR, mono=True)
            n_chunks = max(1, int(np.ceil(len(audio) / CHUNK_SAMP)))
            
            for i in range(n_chunks):
                chunk = audio[i * CHUNK_SAMP : (i + 1) * CHUNK_SAMP]
                if len(chunk) < CHUNK_SAMP:
                    chunk = np.pad(chunk, (0, CHUNK_SAMP - len(chunk)))
                
                raw_boxes = infer_chunk(session, chunk, CONF_THRESH)
                detections = nms(raw_boxes, IOU_THRESH)
                
                for x1, x2, cls_id, conf in detections:
                    total_predictions += 1
                    
                    if cls_id == 4:  # Speech
                        total_speech += 1
                        if conf < 0.3:
                            speech_by_confidence['low'] += 1
                        elif conf < 0.5:
                            speech_by_confidence['medium'] += 1
                        else:
                            speech_by_confidence['high'] += 1
                        
                        output_rows.append({
                            'file': wav_path.name,
                            'class': CLASS_MAP[cls_id],
                            'confidence': f'{conf:.3f}',
                            'x1_sec': f'{x1:.2f}',
                            'x2_sec': f'{x2:.2f}',
                        })
        
        except Exception as e:
            print(f"  ⚠️  Error en {wav_path.name}: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # REPORTE
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "="*80)
    print("RESULTADOS: PASS 1 (Wiener, data/clean/)")
    print("="*80)
    
    print(f"\n📊 Estadísticas Generales:")
    print(f"  Total predicciones    : {total_predictions:,}")
    print(f"  Speech encontrado     : {total_speech:,} ({total_speech/max(1,total_predictions)*100:.1f}%)")
    print(f"  Otras clases          : {total_predictions - total_speech:,}")
    
    if total_speech > 0:
        print(f"\n⚠️  SPEECH DETECTADO EN PASS 1 (Wiener):")
        print(f"\n  Distribución de confianza:")
        print(f"    < 0.3   : {speech_by_confidence['low']:,} ({speech_by_confidence['low']/total_speech*100:.1f}%)")
        print(f"    0.3-0.5 : {speech_by_confidence['medium']:,} ({speech_by_confidence['medium']/total_speech*100:.1f}%)")
        print(f"    >= 0.5  : {speech_by_confidence['high']:,} ({speech_by_confidence['high']/total_speech*100:.1f}%)")
        
        print(f"\n  ❌ CRÍTICO: Wiener genera {total_speech} predicciones de Speech")
        print(f"     → Pero PASS 1 debería filtrar Speech (class_filter=ALL_CLASSES - {{SPEECH_ID}})")
        print(f"     → Estas Speech NO deberían estar en predicciones_clean.csv")
        
        # Guardar ejemplos
        if output_rows:
            df = pd.DataFrame(output_rows)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\n  Top 20 detecciones guardadas en: {OUTPUT_CSV}")
            print(f"\n  Ejemplos:")
            for idx, row in df.head(20).iterrows():
                print(f"    {row['file']:40s} | Speech conf={row['confidence']} | {row['x1_sec']}-{row['x2_sec']}s")
    
    else:
        print(f"\n✅ OK: Wiener NO genera Speech")
        print(f"   → PASS 1 está funcionando correctamente")
        print(f"   → El filtro class_filter={{SPEECH_ID}} está evitando Speech")
        print(f"   → Todas las Speech vienen de PASS 2 (Demucs)")

if __name__ == "__main__":
    main()
