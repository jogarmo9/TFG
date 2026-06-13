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

# Import Silero VAD validator
from silero_vad_validator import SileroVADValidator

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
_ROOT           = Path(__file__).parent.parent
CLEAN_DIR        = _ROOT / "data" / "clean"
CLEAN_DFN_DIR    = _ROOT / "data" / "clean_dfn"
CLEAN_DEMUCS_DIR = _ROOT / "data" / "clean_demucs"
CLEAN_CAND_DIR   = _ROOT / "data" / "clean_cand"        # Wiener-solo-sin-HPSS (Etapa A)
MODEL_PATH      = _ROOT / "models" / "YOLOv5n_original.onnx"
OUTPUT_CSV      = _ROOT / "data" / "processed" / "predicciones_clean.csv"
OUTPUT_CSV_RAW  = _ROOT / "data" / "processed" / "predicciones_clean_raw.csv"
CANDIDATES_FILE = _ROOT / "data" / "processed" / "speech_candidates.txt"

CAND_CONF   = 0.06          # umbral (bajo) para el prefiltro de candidatos Speech

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
                 source_file: str, session_id: str, class_filter=None, conf_thresh: float = CONF_THRESH,
                 vad_validator=None, chunk_audio_data=None):
    """
    Chunks the audio, runs inference, and writes rows to both CSV writers.
    class_filter: set of class_ids to keep, or None for all classes.
    conf_thresh: umbral de confianza (default CONF_THRESH=0.1; mas bajo para candidatos).
    vad_validator: optional SileroVADValidator for validating Speech predictions
    chunk_audio_data: optional dict to return audio chunks for external validation
    """
    audio, _ = lb.load(str(wav_path), sr=SR, mono=True)

    n_samples = len(audio)
    n_chunks  = max(1, int(np.ceil(n_samples / CHUNK_SAMP)))
    
    # Store audio chunks for VAD validation if needed
    audio_chunks_by_idx = {}

    for i in range(n_chunks):
        chunk = audio[i * CHUNK_SAMP : (i + 1) * CHUNK_SAMP]

        if len(chunk) < CHUNK_SAMP:
            chunk = np.pad(chunk, (0, CHUNK_SAMP - len(chunk)))
        
        # Store for VAD validation if needed
        audio_chunks_by_idx[i] = chunk

        chunk_offset = timedelta(seconds=i * CHUNK_SEC)
        raw_boxes    = infer_chunk(session, chunk, conf_thresh)
        detections   = nms(raw_boxes, IOU_THRESH)

        for x1, x2, cls_id, conf in raw_boxes:
            if class_filter is not None and cls_id not in class_filter:
                continue
            
            # VAD validation for Speech predictions
            should_write = True
            if vad_validator is not None and cls_id == SPEECH_ID:
                # Extract the relevant part of the audio chunk
                start_sample = int(x1 * SR)
                end_sample = int(x2 * SR)
                audio_segment = chunk[start_sample:end_sample]
                
                if len(audio_segment) > 0:
                    should_write, vad_prob = vad_validator.validate_segment(audio_segment, yolo_confidence=conf)
                else:
                    should_write = False
            
            if not should_write:
                continue
                
            onset  = file_start + chunk_offset + timedelta(seconds=x1)
            offset = file_start + chunk_offset + timedelta(seconds=x2)
            raw_writer.writerow([mic_id, onset.isoformat(), offset.isoformat(),
                                 float(cls_id), conf, source_file, session_id, "mic"])

        for x1, x2, cls_id, conf in detections:
            if class_filter is not None and cls_id not in class_filter:
                continue
            
            # VAD validation for Speech predictions
            should_write = True
            if vad_validator is not None and cls_id == SPEECH_ID:
                # Extract the relevant part of the audio chunk
                start_sample = int(x1 * SR)
                end_sample = int(x2 * SR)
                audio_segment = chunk[start_sample:end_sample]
                
                if len(audio_segment) > 0:
                    should_write, vad_prob = vad_validator.validate_segment(audio_segment, yolo_confidence=conf)
                else:
                    should_write = False
            
            if not should_write:
                continue
                
            onset  = file_start + chunk_offset + timedelta(seconds=x1)
            offset = file_start + chunk_offset + timedelta(seconds=x2)
            writer.writerow([mic_id, onset.isoformat(), offset.isoformat(),
                             float(cls_id), conf, source_file, session_id, "mic"])


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def _in_date_range(name: str, date_from: str, date_to: str) -> bool:
    """Filtra por la fecha YYYYMMDD embebida en el nombre (inclusive)."""
    d = name[:8]
    if not d.isdigit():
        return True
    if date_from and d < date_from:
        return False
    if date_to and d > date_to:
        return False
    return True


def _run_dir(infer_session, wav_dir: Path, writer, raw_writer,
             class_filter=None, already_done: set = None,
             conf_thresh: float = CONF_THRESH, only_files: set = None,
             date_from: str = None, date_to: str = None, vad_validator=None) -> tuple[int, int]:
    """Procesa todos los WAVs en wav_dir, escribe a ambos writers. Retorna (ok, fail).

    conf_thresh: umbral de confianza propagado a process_file.
    only_files:  si se pasa, procesa solo los WAVs cuyo nombre este en este set
                 (subconjunto de candidatos Speech).
    date_from/date_to: filtro YYYYMMDD inclusive por fecha en el nombre.
    vad_validator: optional SileroVADValidator for validating Speech predictions
    """
    wav_files = sorted(wav_dir.glob("*.wav"))
    if only_files is not None:
        wav_files = [w for w in wav_files if w.name in only_files]
    if date_from or date_to:
        wav_files = [w for w in wav_files if _in_date_range(w.name, date_from, date_to)]
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
                         class_filter=class_filter, conf_thresh=conf_thresh, vad_validator=vad_validator)
            label = "" if class_filter is None else f" (clases {sorted(class_filter)})"
            print(f"  [OK] {filename}{label}")
            ok += 1
        except Exception as e:
            print(f"  [ERROR] {filename} – {e}")
            fail += 1
    return ok, fail


# ──────────────────────────────────────────────────────────────
# ETAPA A — construir lista de candidatos Speech (sin escribir CSV)
# ──────────────────────────────────────────────────────────────
def _build_candidates(infer_session, cand_dir: Path, out_path: Path,
                      conf_thresh: float) -> int:
    """Corre YOLO clase 4 sobre cand_dir y vuelca los nombres de fichero que
    disparan >=1 deteccion Speech a out_path (uno por linea). Retorna N candidatos.

    Prefiltro recall-safe: cand_dir debe ser Wiener-solo-sin-HPSS (preserva
    fricativas; el ruido musical sobre-produce Speech -> alta sensibilidad).
    """
    wav_files = sorted(cand_dir.glob("*.wav"))
    if not wav_files:
        sys.exit(f"[ERROR] No se encontraron .wav en {cand_dir}")

    print(f"[BUILD-CANDIDATES] {len(wav_files)} WAVs en {cand_dir.name}/ | "
          f"clase=Speech({SPEECH_ID}) | conf>={conf_thresh}")
    candidates = []
    for n, wav_path in enumerate(wav_files, 1):
        try:
            audio, _ = lb.load(str(wav_path), sr=SR, mono=True)
            n_chunks = max(1, int(np.ceil(len(audio) / CHUNK_SAMP)))
            hit = False
            for i in range(n_chunks):
                chunk = audio[i * CHUNK_SAMP : (i + 1) * CHUNK_SAMP]
                if len(chunk) < CHUNK_SAMP:
                    chunk = np.pad(chunk, (0, CHUNK_SAMP - len(chunk)))
                for _x1, _x2, cls_id, _conf in infer_chunk(infer_session, chunk, conf_thresh):
                    if cls_id == SPEECH_ID:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                candidates.append(wav_path.name)
        except Exception as e:
            print(f"  [ERROR] {wav_path.name} – {e}")
        if n % 500 == 0:
            print(f"  ... {n}/{len(wav_files)} | candidatos: {len(candidates)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(candidates) + ("\n" if candidates else ""))
    print(f"[BUILD-CANDIDATES] {len(candidates)}/{len(wav_files)} candidatos "
          f"({len(candidates)/len(wav_files)*100:.1f}%) -> {out_path}")
    return len(candidates)


def main():
    parser = argparse.ArgumentParser(description="Inferencia YOLO sobre WAVs limpios")
    parser.add_argument("--reprocess-all", action="store_true",
                        help="Sobreescribe el CSV y reprocesa todos los archivos")
    parser.add_argument("--dual-clean", action="store_true",
                        help="Modo dual: Wiener (data/clean/) para clases!=Speech, "
                             "y separacion/limpieza para Speech (ver --speech-source). "
                             "Siempre sobreescribe CSV.")
    parser.add_argument("--speech-source", choices=["demucs", "dfn3"], default="demucs",
                        help="Fuente del pass 2 (Speech): 'demucs' (data/clean_demucs/, default) "
                             "o 'dfn3' (data/clean_dfn/).")
    parser.add_argument("--build-candidates", action="store_true",
                        help="Etapa A: corre YOLO Speech sobre --cand-dir y vuelca los ficheros "
                             "que disparan clase 4 a speech_candidates.txt (no escribe CSV).")
    parser.add_argument("--cand-dir", default=str(CLEAN_CAND_DIR),
                        help="Carpeta para --build-candidates (default: data/clean_cand, "
                             "debe ser Wiener-solo-sin-HPSS).")
    parser.add_argument("--cand-conf", type=float, default=CAND_CONF,
                        help=f"Umbral de confianza del prefiltro de candidatos (default: {CAND_CONF}).")
    parser.add_argument("--use-candidates", action="store_true",
                        help="En --dual-clean: el pass 2 (Speech) procesa solo los ficheros "
                             "listados en speech_candidates.txt.")
    parser.add_argument("--date-from", default=None,
                        help="Procesar solo ficheros desde esta fecha (YYYYMMDD, inclusive).")
    parser.add_argument("--date-to", default=None,
                        help="Procesar solo ficheros hasta esta fecha (YYYYMMDD, inclusive).")
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

    # ── ETAPA A: construir candidatos Speech y salir ──────────────────────────
    if args.build_candidates:
        cand_dir = Path(args.cand_dir)
        if not cand_dir.exists():
            sys.exit(f"[ERROR] Carpeta de candidatos no encontrada: {cand_dir}\n"
                     f"       Generar Wiener-solo-sin-HPSS primero:\n"
                     f"       python scripts/clean_audio.py --method wiener --passes 2 "
                     f"--hpss-kernel 0 --clean-dir {cand_dir}")
        _build_candidates(infer_session, cand_dir, CANDIDATES_FILE, args.cand_conf)
        return

    header = ["mic_id", "timestamp_onset", "timestamp_offset",
              "class_id", "confidence", "source_file", "session_id", "source"]

    if args.dual_clean:
        if args.speech_source == "demucs":
            speech_dir, speech_tag = CLEAN_DEMUCS_DIR, "Demucs (stem voz)"
        else:
            speech_dir, speech_tag = CLEAN_DFN_DIR, "DFN3"
        if not speech_dir.exists():
            sys.exit(f"[ERROR] Carpeta Speech no encontrada: {speech_dir}\n"
                     f"       Ejecutar primero: python scripts/clean_audio.py --method {args.speech_source}")
        only_files = None
        if args.use_candidates:
            if not CANDIDATES_FILE.exists():
                sys.exit(f"[ERROR] {CANDIDATES_FILE} no existe. Ejecutar primero:\n"
                         f"       python scripts/infer_clean.py --build-candidates")
            with open(CANDIDATES_FILE, "r", encoding="utf-8") as f:
                only_files = {ln.strip() for ln in f if ln.strip()}
            print(f"[INFO] use-candidates: pass 2 limitado a {len(only_files)} ficheros "
                  f"de {CANDIDATES_FILE.name}")

        print(f"[INFO] Modo DUAL-CLEAN: Wiener→clases≠Speech | {speech_tag}→Speech")
        print(f"[INFO] Salida CSV: {OUTPUT_CSV} (modo: sobreescribir)")
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fcsv, \
             open(OUTPUT_CSV_RAW, "w", newline="", encoding="utf-8") as fcsv_raw:
            writer     = csv.writer(fcsv)
            raw_writer = csv.writer(fcsv_raw)
            writer.writerow(header)
            raw_writer.writerow(header)

            print(f"\n[PASS 1] Wiener ({CLEAN_DIR.name}/) → clases≠Speech")
            ok1, fail1 = _run_dir(infer_session, CLEAN_DIR, writer, raw_writer,
                                  class_filter=ALL_CLASSES - {SPEECH_ID},
                                  date_from=args.date_from, date_to=args.date_to)

            print(f"\n[PASS 2] {speech_tag} ({speech_dir.name}/) → Speech únicamente")
            
            # Initialize Silero VAD validator for Speech predictions
            print(f"[PASS 2] Inicializando validador Silero VAD...")
            try:
                vad_validator = SileroVADValidator()
            except Exception as e:
                print(f"[WARN] No se pudo cargar Silero VAD: {e}")
                print(f"[WARN] Continuando sin validación VAD (todos los Speech se guardarán)")
                vad_validator = None
            
            ok2, fail2 = _run_dir(infer_session, speech_dir, writer, raw_writer,
                                  class_filter={SPEECH_ID}, only_files=only_files,
                                  date_from=args.date_from, date_to=args.date_to, vad_validator=vad_validator)

        print(f"\n{'='*50}")
        print(f"Procesados OK : {ok1 + ok2}  (Wiener: {ok1} | {speech_tag}: {ok2})")
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
                                class_filter=None, already_done=already_done,
                                date_from=args.date_from, date_to=args.date_to)

        print(f"\n{'='*50}")
        print(f"Procesados OK : {ok}")
        print(f"Fallidos      : {fail}")
        print(f"Predicciones guardadas en: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
