"""
infer_clean.py
==============
Runs the YOLOv5n ONNX model on every cleaned WAV file inside ./clean/
and writes the predictions to predicciones_clean.csv.

Output CSV format (one row per detection):
  mic_id, timestamp_onset, timestamp_offset, class_id, confidence

Filename convention expected:
  YYYYMMDD_HH_MM_SS_MSMS_MX.wav
  e.g. 20260305_15_21_47_0562_M1.wav
"""

import argparse
import os
import re
import csv
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import librosa as lb
import onnxruntime as ort

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
_ROOT           = Path(__file__).parent.parent
CLEAN_DIR       = _ROOT / "data" / "clean"
CLEAN_DFN_DIR   = _ROOT / "data" / "clean_dfn"
MODEL_PATH      = _ROOT / "models" / "YOLOv5n_original.onnx"
OUTPUT_CSV      = _ROOT / "data" / "processed" / "predicciones_clean.csv"
OUTPUT_CSV_RAW  = _ROOT / "data" / "processed" / "predicciones_clean_raw.csv"

SPEECH_ID   = 4
ALL_CLASSES = set(range(9))

SR          = 16_000        # model sample rate
CHUNK_SEC   = 10            # seconds per inference window
CHUNK_SAMP  = SR * CHUNK_SEC

CONF_THRESH = 0.1           # minimum confidence to keep a box
IOU_THRESH  = 0.7           # NMS IoU threshold

# Mel spectrogram parameters (must match training)
N_FFT       = 2048
HOP_LENGTH  = 256
WIN_LENGTH  = 2048
N_MELS      = 128
FMIN        = 0
FMAX        = 8000
MEL_DB_MIN  = -50
MEL_DB_MAX  = 80
NORM_MIN    = -38.5
NORM_MAX    = 41.37

# YOLO input shape: [1, 3, 128, 640]
YOLO_W      = 640
PAD_COLS    = 7             # columns of padding on each side (→ inner width = 626)
PAD_VALUE   = 0.447058824

CLASSES = [
    "Horn", "Siren", "Pets", "Physiological",
    "Speech", "Ring Tone", "Vibrating", "Notifications", "Cry"
]

# ──────────────────────────────────────────────────────────────
# FILENAME PARSER
# ──────────────────────────────────────────────────────────────
_FN_RE = re.compile(
    r"^(\d{4})(\d{2})(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{4})_M(\d+)\.wav$",
    re.IGNORECASE,
)

def parse_filename(filename: str):
    """
    Returns (mic_id: int, start_dt: datetime) or raises ValueError.
    Pattern: YYYYMMDD_HH_MM_SS_MSMS_MX.wav
    """
    m = _FN_RE.match(filename)
    if not m:
        raise ValueError(f"Nombre de fichero no reconocido: {filename!r}")

    year, month, day, hour, minute, second, ms4, mic = m.groups()
    # ms4 is 4 digits → divide by 10 to get milliseconds (first 3 digits)
    ms = int(ms4) // 10
    dt = datetime(int(year), int(month), int(day),
                  int(hour), int(minute), int(second),
                  ms * 1000)          # microseconds = ms * 1000
    return int(mic), dt


# ──────────────────────────────────────────────────────────────
# AUDIO → MEL SPECTROGRAM
# ──────────────────────────────────────────────────────────────
def wave_to_mel(audio: np.ndarray) -> np.ndarray:
    """Converts a mono float32 waveform (16 kHz) to a normalised Mel spectrogram."""
    X = np.abs(lb.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       win_length=WIN_LENGTH, window="hann", center=True))
    mel = lb.feature.melspectrogram(sr=SR, S=X, n_fft=N_FFT,
                                    hop_length=HOP_LENGTH, power=1.0,
                                    n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
                                    htk=True, norm=None)
    mel = lb.core.amplitude_to_db(mel)
    mel = np.clip(mel, MEL_DB_MIN, MEL_DB_MAX)
    mel = (mel - NORM_MIN) / (NORM_MAX - NORM_MIN)
    return mel


def mel_to_input(mel: np.ndarray) -> np.ndarray:
    """Pads and shapes mel into YOLO input tensor [1, 3, 128, 640]."""
    # Pad time axis to YOLO_W columns
    current_cols = mel.shape[1]
    total_pad = YOLO_W - current_cols
    pad_left   = PAD_COLS
    pad_right  = max(0, total_pad - pad_left)

    mel_padded = np.pad(mel,
                        [(0, 0), (pad_left, pad_right)],
                        mode="constant",
                        constant_values=PAD_VALUE)
    # Truncate if any overshoot
    mel_padded = mel_padded[:, :YOLO_W]

    # [1, 128, 640] → [1, 3, 128, 640]
    tensor = np.expand_dims(mel_padded, 0)
    tensor = np.concatenate([tensor, tensor, tensor], axis=0)
    tensor = np.expand_dims(tensor, 0).astype(np.float32)
    return tensor


# ──────────────────────────────────────────────────────────────
# NMS
# ──────────────────────────────────────────────────────────────
def _iou_1d(b1, b2):
    """IoU between two 1-D intervals [x1, x2]."""
    inter = max(0, min(b1[1], b2[1]) - max(b1[0], b2[0]))
    union = (b1[1] - b1[0]) + (b2[1] - b2[0]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes: list, iou_thresh: float) -> list:
    """Greedy NMS. boxes = list of [x1, x2, cls, conf]."""
    boxes = sorted(boxes, key=lambda b: b[3], reverse=True)
    result = []
    while boxes:
        best = boxes[0]
        result.append(best)
        boxes = [b for b in boxes[1:] if _iou_1d(best, b) < iou_thresh]
    return result


# ──────────────────────────────────────────────────────────────
# INFERENCE ON ONE CHUNK
# ──────────────────────────────────────────────────────────────
def infer_chunk(session, chunk: np.ndarray, conf_thresh: float) -> list:
    """
    Returns a list of raw detections: [x1_sec, x2_sec, class_id, confidence]
    where x1/x2 are offsets within the 10-second chunk.
    """
    mel   = wave_to_mel(chunk)
    inp   = mel_to_input(mel)
    outs  = session.run(None, {"images": inp})

    # Shape: [1, 13, 6400] → transpose → [6400, 13]
    preds = outs[0].transpose()[:, :, 0]

    xc    = preds[:, 0]
    w     = preds[:, 2]
    cls   = np.argmax(preds[:, 4:], axis=-1)
    probs = np.max(preds[:, 4:], axis=-1)

    # Decode x1/x2 from YOLO column coordinates to seconds
    inner = YOLO_W - 2 * PAD_COLS           # 626 usable columns
    x1 = np.clip((xc - w / 2 - PAD_COLS) * CHUNK_SEC / inner, 0, CHUNK_SEC)
    x2 = np.clip((xc + w / 2 - PAD_COLS) * CHUNK_SEC / inner, 0, CHUNK_SEC)

    boxes = [
        [float(x1[i]), float(x2[i]), int(cls[i]), float(probs[i])]
        for i in range(len(probs))
        if probs[i] >= conf_thresh and x2[i] > x1[i]
    ]
    return boxes  # raw, pre-NMS — caller applies NMS


# ──────────────────────────────────────────────────────────────
# PROCESS ONE FILE
# ──────────────────────────────────────────────────────────────
def process_file(session, wav_path: Path, writer, raw_writer, mic_id: int, file_start: datetime,
                 source_file: str, session_id: str, class_filter=None):
    """
    Chunks the audio, runs inference, and writes rows to both CSV writers.
    class_filter: set of class_ids to keep, or None for all classes.
    """
    audio, _ = lb.load(str(wav_path), sr=SR, mono=True)

    n_samples = len(audio)
    n_chunks  = max(1, int(np.ceil(n_samples / CHUNK_SAMP)))

    for i in range(n_chunks):
        chunk = audio[i * CHUNK_SAMP : (i + 1) * CHUNK_SAMP]

        if len(chunk) < CHUNK_SAMP:
            chunk = np.pad(chunk, (0, CHUNK_SAMP - len(chunk)))

        chunk_offset = timedelta(seconds=i * CHUNK_SEC)
        raw_boxes    = infer_chunk(session, chunk, CONF_THRESH)
        detections   = nms(raw_boxes, IOU_THRESH)

        for x1, x2, cls_id, conf in raw_boxes:
            if class_filter is not None and cls_id not in class_filter:
                continue
            onset  = file_start + chunk_offset + timedelta(seconds=x1)
            offset = file_start + chunk_offset + timedelta(seconds=x2)
            raw_writer.writerow([mic_id, onset.isoformat(), offset.isoformat(),
                                 float(cls_id), conf, source_file, session_id, "mic"])

        for x1, x2, cls_id, conf in detections:
            if class_filter is not None and cls_id not in class_filter:
                continue
            onset  = file_start + chunk_offset + timedelta(seconds=x1)
            offset = file_start + chunk_offset + timedelta(seconds=x2)
            writer.writerow([mic_id, onset.isoformat(), offset.isoformat(),
                             float(cls_id), conf, source_file, session_id, "mic"])


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def _run_dir(infer_session, wav_dir: Path, writer, raw_writer,
             class_filter=None, already_done: set = None) -> tuple[int, int]:
    """Procesa todos los WAVs en wav_dir, escribe a ambos writers. Retorna (ok, fail)."""
    wav_files = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        print(f"  [WARN] No se encontraron .wav en {wav_dir}")
        return 0, 0

    ok = fail = 0
    for wav_path in wav_files:
        filename = wav_path.name
        if already_done and filename in already_done:
            print(f"  [SKIP] {filename}")
            continue
        try:
            mic_id, file_start = parse_filename(filename)
        except ValueError as e:
            print(f"  [SKIP] {filename} – {e}")
            fail += 1
            continue
        try:
            session_id = f"{file_start.year}{file_start.month:02d}{file_start.day:02d}"
            process_file(infer_session, wav_path, writer, raw_writer,
                         mic_id, file_start, filename, session_id,
                         class_filter=class_filter)
            label = "" if class_filter is None else f" (clases {sorted(class_filter)})"
            print(f"  [OK] {filename}{label}")
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {filename} – {e}")
            fail += 1
    return ok, fail


def main():
    parser = argparse.ArgumentParser(description="Inferencia YOLO sobre WAVs limpios")
    parser.add_argument("--reprocess-all", action="store_true",
                        help="Sobreescribe el CSV y reprocesa todos los archivos")
    parser.add_argument("--dual-clean", action="store_true",
                        help="Modo dual: Wiener (data/clean/) para clases≠Speech, "
                             "DFN3 (data/clean_dfn/) para Speech. Siempre sobreescribe CSV.")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        import urllib.request
        _MODEL_URL = "https://media.githubusercontent.com/media/ccastore/sed_dis/main/models/YOLOv5n_original.onnx"
        print(f"[INFO] Modelo no encontrado. Descargando desde {_MODEL_URL} ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH)
        print(f"[INFO] Modelo guardado en {MODEL_PATH}")

    if not CLEAN_DIR.exists():
        sys.exit(f"[ERROR] Carpeta de audios limpios no encontrada: {CLEAN_DIR}")

    print(f"[INFO] Cargando modelo: {MODEL_PATH}")
    available = ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]
    infer_session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    print(f"[INFO] ONNX provider: {infer_session.get_providers()[0]}")

    header = ["mic_id", "timestamp_onset", "timestamp_offset",
              "class_id", "confidence", "source_file", "session_id", "source"]

    if args.dual_clean:
        if not CLEAN_DFN_DIR.exists():
            sys.exit(f"[ERROR] Carpeta DFN3 no encontrada: {CLEAN_DFN_DIR}\n"
                     f"       Ejecutar primero: python scripts/clean_audio_prueba.py --method dfn3")
        print("[INFO] Modo DUAL-CLEAN: Wiener→clases≠Speech | DFN3→Speech")
        print(f"[INFO] Salida CSV: {OUTPUT_CSV} (modo: sobreescribir)")
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fcsv, \
             open(OUTPUT_CSV_RAW, "w", newline="", encoding="utf-8") as fcsv_raw:
            writer     = csv.writer(fcsv)
            raw_writer = csv.writer(fcsv_raw)
            writer.writerow(header)
            raw_writer.writerow(header)

            print(f"\n[PASS 1] Wiener ({CLEAN_DIR.name}/) → todas las clases excepto Speech")
            ok1, fail1 = _run_dir(infer_session, CLEAN_DIR, writer, raw_writer,
                                  class_filter=ALL_CLASSES - {SPEECH_ID})

            print(f"\n[PASS 2] DFN3 ({CLEAN_DFN_DIR.name}/) → Speech únicamente")
            ok2, fail2 = _run_dir(infer_session, CLEAN_DFN_DIR, writer, raw_writer,
                                  class_filter={SPEECH_ID})

        print(f"\n{'='*50}")
        print(f"Procesados OK : {ok1 + ok2}  (Wiener: {ok1} | DFN3: {ok2})")
        print(f"Fallidos      : {fail1 + fail2}")
        print(f"Predicciones guardadas en: {OUTPUT_CSV}")

    else:
        wav_files = sorted(CLEAN_DIR.glob("*.wav"))
        if not wav_files:
            sys.exit(f"[ERROR] No se encontraron .wav en {CLEAN_DIR}")

        already_done = set()
        write_mode = "w"
        if not args.reprocess_all and OUTPUT_CSV.exists():
            with open(OUTPUT_CSV, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                hdr = next(reader, [])
                if "source_file" in hdr:
                    sf_idx = hdr.index("source_file")
                    for row in reader:
                        if len(row) > sf_idx:
                            already_done.add(row[sf_idx])
                    write_mode = "a"

        pending = {f.name for f in wav_files} - already_done
        if not pending:
            print("[INFO] Todos los archivos ya procesados. Usa --reprocess-all para forzar.")
            return
        print(f"[INFO] Ya procesados: {len(already_done)} | Pendientes: {len(pending)}")
        print(f"[INFO] Salida CSV: {OUTPUT_CSV} (modo: {'sobreescribir' if write_mode == 'w' else 'añadir'})")

        with open(OUTPUT_CSV, write_mode, newline="", encoding="utf-8") as fcsv, \
             open(OUTPUT_CSV_RAW, write_mode, newline="", encoding="utf-8") as fcsv_raw:
            writer     = csv.writer(fcsv)
            raw_writer = csv.writer(fcsv_raw)
            if write_mode == "w":
                writer.writerow(header)
                raw_writer.writerow(header)

            ok, fail = _run_dir(infer_session, CLEAN_DIR, writer, raw_writer,
                                class_filter=None, already_done=already_done)

        print(f"\n{'='*50}")
        print(f"Procesados OK : {ok}")
        print(f"Fallidos      : {fail}")
        print(f"Predicciones guardadas en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
