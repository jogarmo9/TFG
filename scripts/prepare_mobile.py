"""
prepare_mobile.py
=================
Preprocesa sesiones móvil para inferencia YOLO.

Para cada carpeta en data/mobile/*/:
  1. Lee meta.json → audio_start_utc
     Fallback: mtime del audio + advertencia
  2. Convierte audio a WAV 16kHz mono
     MP3: miniaudio.mp3_read_file_f32() (sin ffmpeg)
     Otros: librosa.load(sr=16000, mono=True)
  3. Aplica Wiener Gated chunk a chunk:
       SNR raw > SNR_GATE_DB → chunk raw
       SNR raw ≤ SNR_GATE_DB → chunk Wiener Mobile (hp=100, nr=0.50, ×2)
  4. Inferencia YOLO inline: chunks 10s → mel → conf ≥ 0.1 → NMS 1D IoU ≥ 0.7
  5. Append a data/processed/predictions_mobile.parquet
  6. Parsea track.gpx → append a data/processed/tracks_mobile.parquet

Uso:
  python scripts/prepare_mobile.py
  python scripts/prepare_mobile.py --session data/mobile/PAIPORTA-ALDAIA
  python scripts/prepare_mobile.py --snr-gate 6.0       # ajustar umbral SNR
  python scripts/prepare_mobile.py --reprocess-all      # borra parquets y regenera todo
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import gpxpy
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from clean_audio import clean_audio as _wiener_clean

ROOT = Path(__file__).parent.parent
MOBILE_DIR = ROOT / "data" / "mobile"
PROCESSED_DIR = ROOT / "data" / "processed"

AUDIO_EXTS = [".wav", ".mp3", ".mp4", ".m4a", ".ogg", ".flac"]

# Parámetros Wiener Mobile (optimizados por grid search en notebook 04)
_WIENER_MOBILE_PARAMS = dict(hp_cutoff=100, nr_strength=0.50, lp_cutoff=8000, passes=2)

# Umbral SNR-gate: chunks con SNR raw > este valor no se filtran
_SNR_GATE_DB = 8.0


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def find_audio(session_dir: Path) -> Path | None:
    for ext in AUDIO_EXTS:
        matches = list(session_dir.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def find_gpx(session_dir: Path) -> Path | None:
    matches = list(session_dir.glob("*.gpx"))
    return matches[0] if matches else None


def load_meta(session_dir: Path, audio_path: Path, gpx_path: Path = None) -> dict:
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["audio_start_utc"] = datetime.fromisoformat(
            meta["audio_start_utc"].replace("Z", "+00:00")
        )
        return meta

    # Sin meta.json: intentar extraer audio_start del primer trackpoint GPX
    if gpx_path is not None:
        with open(gpx_path, encoding="utf-8") as f:
            gpx = gpxpy.parse(f)
        first_time = None
        for track in gpx.tracks:
            for seg in track.segments:
                for pt in seg.points:
                    if pt.time:
                        first_time = pt.time
                        break
                if first_time:
                    break
            if first_time:
                break
        if first_time is not None:
            if first_time.tzinfo is None:
                first_time = first_time.replace(tzinfo=timezone.utc)
            meta = {
                "audio_start_utc": first_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "mic_id": 0,
                "session_id": session_dir.name,
                "notes": "Auto-generado: audio_start_utc = primer trackpoint GPX. Verificar desfase real.",
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"  [INFO] meta.json auto-generado desde GPX: {first_time.isoformat()}")
            meta["audio_start_utc"] = first_time
            return meta

    # Último recurso: mtime del audio
    mtime = os.path.getmtime(audio_path)
    audio_start = datetime.fromtimestamp(mtime, tz=timezone.utc)
    print(f"  [WARN] Sin meta.json en {session_dir.name}. Usando mtime del audio: {audio_start.isoformat()}")
    return {
        "audio_start_utc": audio_start,
        "mic_id": 0,
        "session_id": session_dir.name,
        "notes": "fallback: mtime",
    }


def convert_to_wav(audio_path: Path, out_path: Path):
    """Convierte audio a WAV 16kHz mono."""
    import subprocess
    import numpy as np
    import soundfile as sf

    ext = audio_path.suffix.lower()

    if ext == ".wav":
        shutil.copy2(audio_path, out_path)
        return

    if ext == ".mp3":
        import miniaudio
        import librosa
        decoded = miniaudio.mp3_read_file_f32(str(audio_path))
        audio = np.array(decoded.samples, dtype=np.float32)
        if decoded.nchannels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
        if decoded.sample_rate != 16_000:
            audio = librosa.resample(audio, orig_sr=decoded.sample_rate, target_sr=16_000, res_type='kaiser_best')
        sf.write(str(out_path), audio, 16_000, subtype="PCM_16")
        return

    if ext in (".mp4", ".m4a"):
        import shutil as _shutil
        _FFMPEG_FALLBACK = (
            r"C:\Users\Yurest\AppData\Local\Microsoft\WinGet\Packages"
            r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
            r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe"
        )
        ffmpeg_bin = _shutil.which("ffmpeg") or _FFMPEG_FALLBACK
        result = subprocess.run(
            [ffmpeg_bin, "-y", "-i", str(audio_path),
             "-ar", "16000", "-ac", "1", "-f", "wav", str(out_path)],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg falló convirtiendo {audio_path.name}.\n"
                f"ffmpeg usado: {ffmpeg_bin}\n"
                f"Stderr: {result.stderr.decode(errors='replace')}"
            )
        return

    # Fallback: librosa para ogg, flac, etc.
    import librosa
    audio, _ = librosa.load(str(audio_path), sr=16_000, mono=True)
    sf.write(str(out_path), audio, 16_000, subtype="PCM_16")


def audio_start_to_filename(audio_start: datetime, mic_id: int) -> str:
    """YYYYMMDD_HH_MM_SS_0000_M{mic_id}.wav"""
    return (
        f"{audio_start.strftime('%Y%m%d_%H_%M_%S')}_0000_M{mic_id}.wav"
    )


def parse_gpx_track(gpx_path: Path, session_id: str) -> pd.DataFrame:
    with open(gpx_path, encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    rows = []
    for track in gpx.tracks:
        for segment in track.segments:
            for pt in segment.points:
                rows.append({
                    "session_id": session_id,
                    "source": "mobile",
                    "lat": pt.latitude,
                    "lon": pt.longitude,
                    "ele": pt.elevation,
                    "time": pt.time.replace(tzinfo=timezone.utc) if pt.time.tzinfo is None else pt.time,
                })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


# ──────────────────────────────────────────────────────────────
# SNR-GATED WIENER
# ──────────────────────────────────────────────────────────────

def _estimate_snr_chunk(audio, frame_len: int = 2048, hop: int = 256) -> float:
    """SNR estimado por energía de frames. Frames bottom-20% = ruido."""
    import numpy as np
    frames = [audio[i:i + frame_len] for i in range(0, len(audio) - frame_len, hop)]
    if not frames:
        return 0.0
    energies = np.array([np.mean(f ** 2) for f in frames])
    threshold = np.percentile(energies, 20)
    noise_e  = energies[energies <= threshold].mean()
    signal_e = energies[energies >  threshold].mean()
    if noise_e < 1e-12:
        return 60.0
    return float(10 * np.log10(signal_e / noise_e + 1e-10))


def apply_wiener_gated_wav(raw_wav: Path, out_wav: Path,
                            snr_gate_db: float = _SNR_GATE_DB,
                            wiener_params: dict = None):
    """Aplica Wiener Mobile chunk a chunk, solo si SNR_raw ≤ snr_gate_db.

    Para cada chunk de 10s del audio completo:
      SNR raw > snr_gate_db  →  chunk sin filtrar (raw)
      SNR raw ≤ snr_gate_db  →  chunk Wiener Mobile

    Escribe el WAV resultante en out_wav.
    """
    import numpy as np
    import soundfile as sf
    import librosa

    if wiener_params is None:
        wiener_params = _WIENER_MOBILE_PARAMS

    SR = 16_000
    CHUNK_SAMP = SR * 10

    audio, _ = librosa.load(str(raw_wav), sr=SR, mono=True)
    n_chunks = max(1, int(np.ceil(len(audio) / CHUNK_SAMP)))

    out_chunks = []
    n_wiener = 0
    n_raw_kept = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i in range(n_chunks):
            chunk = audio[i * CHUNK_SAMP:(i + 1) * CHUNK_SAMP]
            snr = _estimate_snr_chunk(chunk)
            if snr > snr_gate_db:
                out_chunks.append(chunk)
                n_raw_kept += 1
            else:
                in_f  = str(tmp_path / "_c_in.wav")
                out_f = str(tmp_path / "_c_out.wav")
                sf.write(in_f, chunk, SR, subtype="PCM_16")
                _wiener_clean(in_f, out_f, **wiener_params)
                cleaned, _ = librosa.load(out_f, sr=SR, mono=True)
                out_chunks.append(cleaned)
                n_wiener += 1

    result = np.concatenate(out_chunks)
    sf.write(str(out_wav), result, SR, subtype="PCM_16")
    print(f"  [gate={snr_gate_db}dB] {n_wiener}/{n_chunks} chunks → Wiener  |  {n_raw_kept}/{n_chunks} → Raw")


# ──────────────────────────────────────────────────────────────
# INFERENCE INLINE
# ──────────────────────────────────────────────────────────────

def infer_wav_direct(wav_path: Path, audio_start: datetime, mic_id: int,
                     class_filter=None) -> tuple[list, list]:
    """
    Ejecuta inferencia YOLO sobre un WAV. Retorna (rows_nms, rows_raw).
    class_filter: set de class_ids a conservar, o None para todas.
    """
    import numpy as np
    import librosa as lb
    import onnxruntime as ort
    from datetime import timedelta

    MODEL_PATH = ROOT / "models" / "YOLOv5n_original.onnx"
    SR = 16_000
    CHUNK_SEC = 10
    CHUNK_SAMP = SR * CHUNK_SEC
    CONF_THRESH = 0.1
    IOU_THRESH = 0.7
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

    def wave_to_mel(audio):
        X = np.abs(lb.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
        mel = lb.feature.melspectrogram(sr=SR, S=X, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                         power=1.0, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, htk=True, norm=None)
        mel = lb.amplitude_to_db(mel)
        mel = np.clip(mel, MEL_DB_MIN, MEL_DB_MAX)
        return (mel - NORM_MIN) / (NORM_MAX - NORM_MIN)

    def mel_to_input(mel):
        total_pad = YOLO_W - mel.shape[1]
        mel_padded = np.pad(mel, [(0, 0), (PAD_COLS, max(0, total_pad - PAD_COLS))], constant_values=PAD_VALUE)
        mel_padded = mel_padded[:, :YOLO_W]
        t = np.expand_dims(mel_padded, 0)
        return np.expand_dims(np.concatenate([t, t, t], axis=0), 0).astype(np.float32)

    def iou_1d(b1, b2):
        inter = max(0, min(b1[1], b2[1]) - max(b1[0], b2[0]))
        union = (b1[1] - b1[0]) + (b2[1] - b2[0]) - inter
        return inter / union if union > 0 else 0.0

    def nms(boxes):
        boxes = sorted(boxes, key=lambda b: b[3], reverse=True)
        result = []
        while boxes:
            best = boxes[0]
            result.append(best)
            boxes = [b for b in boxes[1:] if iou_1d(best, b) < IOU_THRESH]
        return result

    if not MODEL_PATH.exists():
        import urllib.request
        _MODEL_URL = "https://media.githubusercontent.com/media/ccastore/sed_dis/main/models/YOLOv5n_original.onnx"
        print(f"  [INFO] Modelo no encontrado. Descargando desde {_MODEL_URL} ...")
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, MODEL_PATH)
        print(f"  [INFO] Modelo guardado en {MODEL_PATH}")

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    audio, _ = lb.load(str(wav_path), sr=SR, mono=True)
    n_chunks = max(1, int(np.ceil(len(audio) / CHUNK_SAMP)))
    inner = YOLO_W - 2 * PAD_COLS

    pre_nms = []  # [t1_s, t2_s, cls_id, conf] — segundos absolutos desde audio_start
    raw_rows = []
    for i in range(n_chunks):
        chunk = audio[i * CHUNK_SAMP:(i + 1) * CHUNK_SAMP]
        if len(chunk) < CHUNK_SAMP:
            chunk = np.pad(chunk, (0, CHUNK_SAMP - len(chunk)))
        mel = wave_to_mel(chunk)
        inp = mel_to_input(mel)
        outs = session.run(None, {"images": inp})
        preds = outs[0].transpose()[:, :, 0]
        xc = preds[:, 0]
        w = preds[:, 2]
        cls = np.argmax(preds[:, 4:], axis=-1)
        probs = np.max(preds[:, 4:], axis=-1)
        x1 = np.clip((xc - w / 2 - PAD_COLS) * CHUNK_SEC / inner, 0, CHUNK_SEC)
        x2 = np.clip((xc + w / 2 - PAD_COLS) * CHUNK_SEC / inner, 0, CHUNK_SEC)
        boxes = [[float(x1[j]), float(x2[j]), int(cls[j]), float(probs[j])]
                 for j in range(len(probs)) if probs[j] >= CONF_THRESH and x2[j] > x1[j]]
        chunk_offset_s = i * CHUNK_SEC

        for bx1, bx2, cls_id, conf in boxes:
            if class_filter is not None and cls_id not in class_filter:
                continue
            t1_s = chunk_offset_s + bx1
            t2_s = chunk_offset_s + bx2
            pre_nms.append([t1_s, t2_s, cls_id, conf])
            raw_rows.append({
                "mic_id": mic_id,
                "timestamp_onset": (audio_start + timedelta(seconds=t1_s)).isoformat(),
                "timestamp_offset": (audio_start + timedelta(seconds=t2_s)).isoformat(),
                "class_id": float(cls_id),
                "confidence": conf,
                "source_file": wav_path.name,
            })

    rows = []
    for bx1, bx2, cls_id, conf in nms(pre_nms):
        rows.append({
            "mic_id": mic_id,
            "timestamp_onset": (audio_start + timedelta(seconds=bx1)).isoformat(),
            "timestamp_offset": (audio_start + timedelta(seconds=bx2)).isoformat(),
            "class_id": float(cls_id),
            "confidence": conf,
            "source_file": wav_path.name,
        })

    print(f"  [OK] {wav_path.name}: {len(rows)} detecciones ({len(raw_rows)} raw)")
    return rows, raw_rows


# ──────────────────────────────────────────────────────────────
# PROCESS ONE SESSION
# ──────────────────────────────────────────────────────────────

def process_session(session_dir: Path, snr_gate_db: float = _SNR_GATE_DB):
    print(f"\n[SESSION] {session_dir.name}")

    audio_path = find_audio(session_dir)
    gpx_path = find_gpx(session_dir)

    if not audio_path:
        print(f"  [SKIP] Sin archivo de audio en {session_dir}")
        return
    if not gpx_path:
        print(f"  [SKIP] Sin archivo GPX en {session_dir}")
        return

    meta = load_meta(session_dir, audio_path, gpx_path=gpx_path)
    audio_start = meta["audio_start_utc"]
    mic_id = meta.get("mic_id", 0)
    session_id = meta.get("session_id", session_dir.name)

    print(f"  Audio: {audio_path.name}")
    print(f"  GPX:   {gpx_path.name}")
    print(f"  Start: {audio_start.isoformat()}")
    print(f"  Session: {session_id}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav_name = audio_start_to_filename(audio_start, mic_id)
        wav_raw = tmp_path / wav_name
        print(f"  Convirtiendo audio → {wav_name}")
        convert_to_wav(audio_path, wav_raw)

        # Inferencia sobre audio crudo (sin Wiener) → comparación notebook
        print(f"  Inferencia raw (sin Wiener)...")
        det_raw_audio, _ = infer_wav_direct(wav_raw, audio_start, mic_id, class_filter=None)

        clean_gated = tmp_path / ("gated_" + wav_name)
        print(f"  Wiener Gated (gate={snr_gate_db}dB, nr=0.50, hp=100Hz, ×2) → todas las clases...")
        apply_wiener_gated_wav(wav_raw, clean_gated, snr_gate_db=snr_gate_db)
        det_wiener, raw_wiener = infer_wav_direct(
            clean_gated, audio_start, mic_id,
            class_filter=None
        )

        detections           = det_wiener      # producción: Wiener + NMS
        raw_detections       = raw_wiener      # Wiener pre-NMS (debug)
        raw_audio_detections = det_raw_audio   # audio crudo + NMS (comparación)

    print(f"  Detecciones: {len(detections)} NMS Wiener | {len(raw_audio_detections)} NMS Raw")

    _PRED_COLS = ["mic_id", "timestamp_onset", "timestamp_offset",
                  "class_id", "confidence", "session_id", "source"]

    def _build_df(det_list):
        if not det_list:
            return pd.DataFrame(columns=_PRED_COLS)
        df = pd.DataFrame(det_list)
        df["session_id"] = session_id
        df["source"] = "mobile"
        df["timestamp_onset"]  = pd.to_datetime(df["timestamp_onset"],  format="ISO8601", utc=True)
        df["timestamp_offset"] = pd.to_datetime(df["timestamp_offset"], format="ISO8601", utc=True)
        return df

    if not detections:
        print(f"  [WARN] Sin detecciones para {session_id}")
    df_preds           = _build_df(detections)
    df_raw             = _build_df(raw_detections)
    df_raw_audio       = _build_df(raw_audio_detections)

    df_track = parse_gpx_track(gpx_path, session_id)
    print(f"  GPX trackpoints: {len(df_track)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = PROCESSED_DIR / "predictions_mobile.parquet"
    if preds_path.exists():
        existing = pd.read_parquet(preds_path)
        existing = existing[existing["session_id"] != session_id]
        df_preds = pd.concat([existing, df_preds], ignore_index=True)
    df_preds.to_parquet(preds_path, index=False)
    print(f"  Predicciones móvil guardadas: {len(df_preds)} total en {preds_path.name}")

    raw_path = PROCESSED_DIR / "predictions_mobile_raw.parquet"
    if raw_path.exists():
        existing_raw = pd.read_parquet(raw_path)
        existing_raw = existing_raw[existing_raw["session_id"] != session_id]
        df_raw = pd.concat([existing_raw, df_raw], ignore_index=True)
    df_raw.to_parquet(raw_path, index=False)
    print(f"  Predicciones raw (pre-NMS Wiener): {len(df_raw)} total en {raw_path.name}")

    no_wiener_path = PROCESSED_DIR / "predictions_mobile_noWiener.parquet"
    if no_wiener_path.exists():
        existing_nw = pd.read_parquet(no_wiener_path)
        existing_nw = existing_nw[existing_nw["session_id"] != session_id]
        df_raw_audio = pd.concat([existing_nw, df_raw_audio], ignore_index=True)
    df_raw_audio.to_parquet(no_wiener_path, index=False)
    print(f"  Predicciones sin Wiener (NMS):     {len(df_raw_audio)} total en {no_wiener_path.name}")

    tracks_path = PROCESSED_DIR / "tracks_mobile.parquet"
    if tracks_path.exists():
        existing_t = pd.read_parquet(tracks_path)
        existing_t = existing_t[existing_t["session_id"] != session_id]
        df_track = pd.concat([existing_t, df_track], ignore_index=True)
    df_track.to_parquet(tracks_path, index=False)
    print(f"  Tracks móvil guardados: {len(df_track)} total en {tracks_path.name}")


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocesa sesiones móvil para inferencia YOLO")
    parser.add_argument("--session", type=Path, default=None,
                        help="Carpeta de una sesión concreta (por defecto: todas)")
    parser.add_argument("--snr-gate", type=float, default=_SNR_GATE_DB,
                        help=f"Umbral SNR-gate en dB (default: {_SNR_GATE_DB}). "
                             "Chunks con SNR raw > umbral no se filtran.")
    parser.add_argument("--reprocess-all", action="store_true",
                        help="Borra los parquets de salida antes de procesar (regenera todo desde cero).")
    args = parser.parse_args()

    if args.session:
        sessions = [args.session]
    else:
        sessions = [d for d in MOBILE_DIR.iterdir() if d.is_dir()]

    if args.reprocess_all:
        _to_delete = [
            PROCESSED_DIR / "predictions_mobile.parquet",
            PROCESSED_DIR / "predictions_mobile_raw.parquet",
            PROCESSED_DIR / "predictions_mobile_noWiener.parquet",
            PROCESSED_DIR / "tracks_mobile.parquet",
        ]
        for p in _to_delete:
            if p.exists():
                p.unlink()
                print(f"[--reprocess-all] Borrado: {p.name}")
        print("[--reprocess-all] Regenerando todo desde cero.")

    if not sessions:
        sys.exit(f"[ERROR] No se encontraron sesiones en {MOBILE_DIR}")

    print(f"[INFO] Sesiones a procesar: {len(sessions)}")
    print(f"[INFO] SNR gate: {args.snr_gate} dB")
    for s in sessions:
        process_session(s, snr_gate_db=args.snr_gate)

    print("\n[DONE] Procesamiento móvil completado.")
    print(f"  → {PROCESSED_DIR / 'predictions_mobile.parquet'}")
    print(f"  → {PROCESSED_DIR / 'tracks_mobile.parquet'}")


if __name__ == "__main__":
    main()
