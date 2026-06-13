import argparse
import warnings
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft
import os
from pathlib import Path
import shutil
import multiprocessing

# =====================================================================
# 1. PARCHE DE ENTORNO CRÍTICO ANTES DE LEVANTAR MULTIPROCESSING
# =====================================================================
os.environ["TORCHAUDIO_USE_BACKEND"] = "soundfile"
os.environ["ORT_ENABLE_MODULE_INITIALIZERS"] = "1"
# Limitar hilos internos de ONNX por proceso para evitar que colisionen entre sí
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch

def _select_torch_device(tag: str = ""):
    if torch.cuda.is_available():
        if tag:
            print(f"[{tag}] CUDA disponible — usando GPU NVIDIA")
        return torch.device("cuda"), "cuda"
    return torch.device("cpu"), "cpu"


def _declip_pass(audio, clip_thresh):
    mask = np.abs(audio) >= clip_thresh
    pct = mask.mean() * 100
    if pct > 0.05:
        idx = np.arange(len(audio))
        idx_clean = idx[~mask]
        if len(idx_clean) > 10:
            cs = CubicSpline(idx_clean, audio[idx_clean])
            audio = audio.copy()
            audio[mask] = np.clip(cs(idx[mask]), -1.0, 1.0)
    return audio, pct


def _wiener_pass(audio, nr_strength, hop=256, win=2048):
    frames = []
    mags = []
    for i in range(0, len(audio) - win, hop):
        spec = rfft(audio[i:i + win] * np.hanning(win))
        mags.append(np.abs(spec))
        frames.append((i, spec))
    if not mags:
        return audio
    noise = np.percentile(mags, 15, axis=0)
    out = np.zeros(len(audio))
    cnt = np.zeros(len(audio))
    w = np.hanning(win)
    for i, spec in frames:
        mag = np.abs(spec)
        gain = np.maximum(0, (mag - nr_strength * noise) / (mag + 1e-8))
        cleaned = irfft(gain * mag * np.exp(1j * np.angle(spec)))
        out[i:i + win] += cleaned * w
        cnt[i:i + win] += w
    return out / np.where(cnt < 1e-8, 1, cnt)


def remove_impulses(audio: np.ndarray, kernel_size: int = 11, threshold_sigma: float = 2.5, passes: int = 2, kernel_size_p2: int = 15, threshold_sigma_p2: float = 1.5) -> np.ndarray:
    from scipy.signal import medfilt as _medfilt
    a = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(a ** 2)))
    if rms < 1e-10:
        return a
    med  = _medfilt(a.astype(np.float64), kernel_size=kernel_size).astype(np.float32)
    mask = np.abs(a - med) > threshold_sigma * rms
    a    = a.copy()
    a[mask] = med[mask]
    for _ in range(passes - 1):
        rms2  = float(np.sqrt(np.mean(a ** 2)))
        med2  = _medfilt(a.astype(np.float64), kernel_size=kernel_size_p2).astype(np.float32)
        mask2 = np.abs(a - med2) > threshold_sigma_p2 * rms2
        a[mask2] = med2[mask2]
    return a


def _load_dfn3_model():
    from df.enhance import init_df
    model, df_state, _ = init_df()
    device, kind = _select_torch_device("DFN3")
    if kind != "cpu":
        model = model.to(device)
    return model, df_state


def clean_audio_dfn3(path_in: str, path_out: str, atten_lim_db: float = 75.0, model=None, df_state=None):
    import soundfile as sf
    import librosa
    from df.enhance import enhance, init_df, load_audio
    if model is None or df_state is None:
        model, df_state, _ = init_df()
    os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
    audio, _ = load_audio(path_in, sr=df_state.sr())
    device, kind = _select_torch_device()
    if kind != "cpu":
        audio = audio.to(device)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*sinc_interpolation.*", category=UserWarning)
        enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim_db)
    arr = enhanced.squeeze().cpu().numpy() if hasattr(enhanced, "numpy") else np.array(enhanced.cpu()).squeeze()
    if df_state.sr() != 16_000:
        arr = librosa.resample(arr, orig_sr=df_state.sr(), target_sr=16_000, res_type='kaiser_best')
    sf.write(path_out, arr, 16_000, subtype="PCM_16")


# =====================================================================
# 2. SELECCIÓN DE HARDWARE COMPATIBLE CON MULTIPROCESO
# =====================================================================
def _select_demucs_device():
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"], "cuda"
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"], "dml"
        if "DirectMLExecutionProvider" in available:
            return ["DirectMLExecutionProvider", "CPUExecutionProvider"], "dml"
    except ImportError:
        pass
    return ["CPUExecutionProvider"], "cpu"


def _load_demucs_model(model_name: str = "htdemucs"):
    model_filename = "htdemucs_ft.yaml" if model_name == "htdemucs_ft" else "htdemucs.yaml"
    providers, device_kind = _select_demucs_device()
    return model_filename, device_kind, providers


def clean_audio(path_in, path_out, clip_thresh=0.85, nr_strength=0.85, lp_cutoff=7500, hp_cutoff=None, passes=2, impulse_removal=False, impulse_kernel=11, impulse_threshold=2.5, impulse_passes=2, hpss_kernel=0):
    try:
        sr, data = wavfile.read(path_in)
    except Exception as e:
        raise ValueError(f"No se pudo leer el WAV '{path_in}': {e}")
    stereo = data.ndim == 2
    channels = [data[:, i] for i in range(data.shape[1])] if stereo else [data]
    results = []
    for ch in channels:
        audio = ch.astype(np.float32) / 32768.0
        pct_total = 0.0
        for p in range(passes):
            thresh = clip_thresh - p * 0.03
            audio, pct = _declip_pass(audio, thresh)
            if p == 0:
                pct_total = pct
            if impulse_removal and p == 0:
                audio = remove_impulses(audio, kernel_size=impulse_kernel, threshold_sigma=impulse_threshold, passes=impulse_passes)
            audio = _wiener_pass(audio, nr_strength)
        if hp_cutoff is not None:
            b, a = butter(4, hp_cutoff / (sr / 2), btype='high')
            audio = filtfilt(b, a, audio)
        lp_norm = min(lp_cutoff, sr / 2 - 1) / (sr / 2)
        b, a = butter(4, lp_norm, btype='low')
        audio = filtfilt(b, a, audio)
        if hpss_kernel > 0:
            import librosa as _lb
            _N_FFT, _HOP = 2048, 256
            D = _lb.stft(audio, n_fft=_N_FFT, hop_length=_HOP, win_length=_N_FFT)
            D_harm, _ = _lb.decompose.hpss(D, kernel_size=hpss_kernel)
            audio = _lb.istft(D_harm, hop_length=_HOP, win_length=_N_FFT, length=len(audio))
        results.append((np.clip(audio, -1, 1), pct_total))
    out_data = np.stack([r[0] for r in results], axis=1) if stereo else results[0][0]
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    wavfile.write(path_out, sr, (out_data * 32767).astype(np.int16))
    return results[0][1]


# =====================================================================
# GLOBAL CACHE AND WORKER INITIALIZATION
# =====================================================================
_GLOBAL_SEPARATOR = None
_GLOBAL_DFN3_MODEL = None
_GLOBAL_DFN3_STATE = None

def init_worker_context(demucs_device):
    """Función de arranque crítico que despierta la GPU de inmediato al instanciarse el proceso."""
    os.environ["ORT_ENABLE_MODULE_INITIALIZERS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    if demucs_device == "dml":
        try:
            import onnxruntime as ort
            # Obligar al subproceso a registrar la sesión DirectML nativa de C++ en Windows de inmediato
            _ = ort.InferenceSession(bytes(), ort.SessionOptions(), providers=["DirectMLExecutionProvider"])
        except Exception:
            pass


def process_file_worker(f, clean_dir, args, demucs_config, demucs_device, demucs_providers):
    global _GLOBAL_SEPARATOR, _GLOBAL_DFN3_MODEL, _GLOBAL_DFN3_STATE
    proc_id = multiprocessing.current_process().pid
    filename = f.name
    out_path = clean_dir / filename

    try:
        if args.method == "dfn3":
            if _GLOBAL_DFN3_MODEL is None:
                _GLOBAL_DFN3_MODEL, _GLOBAL_DFN3_STATE = _load_dfn3_model()
            clean_audio_dfn3(str(f), str(out_path), atten_lim_db=args.atten_lim_db, model=_GLOBAL_DFN3_MODEL, df_state=_GLOBAL_DFN3_STATE)
            print(f"[OK] {filename}")

        elif args.method == "demucs":
            if _GLOBAL_SEPARATOR is None:
                tmp_dir = clean_dir / f"tmp_onnx_{proc_id}"
                os.makedirs(tmp_dir, exist_ok=True)
                
                root_dir = Path(__file__).parent.parent
                model_dir = root_dir / "data" / "models"
                os.makedirs(model_dir, exist_ok=True)
                
                from audio_separator.separator import Separator

                # Configuración explícita de hilos por sesión ONNX para evitar bloqueos mutuos
                sep_kwargs = dict(
                    model_file_dir=str(model_dir),
                    output_dir=str(tmp_dir),
                    output_format="WAV",
                    normalization_threshold=1.0,
                    output_single_stem="vocals",
                    demucs_params={
                        "segment_size": "Default",
                        "shifts": 2,
                        "overlap": args.demucs_overlap,
                        "segments_enabled": True,
                    },
                )

                if demucs_device == "dml":
                    sep_kwargs["use_directml"] = True
                elif demucs_device == "cuda":
                    sep_kwargs["use_cuda"] = True

                separator_instance = Separator(**sep_kwargs)
                
                # Configurar opciones de optimización interna de ONNX para multiproceso masivo
                try:
                    separator_instance.model_runner.opts.intra_op_num_threads = 1
                    separator_instance.model_runner.opts.inter_op_num_threads = 1
                except Exception:
                    pass

                separator_instance.execution_providers = demucs_providers
                separator_instance.hardware_acceleration_enabled = (demucs_device != "cpu")
                separator_instance.load_model(model_filename=demucs_config)
                _GLOBAL_SEPARATOR = separator_instance

            separator = _GLOBAL_SEPARATOR
            tmp_dir = Path(separator.output_dir)

            output_files = separator.separate(str(f))
            if output_files and len(output_files) > 0:
                file_generado = tmp_dir / output_files[0]
                import soundfile as sf
                import librosa
                audio_vocals, _ = librosa.load(file_generado, sr=16_000, mono=True)
                sf.write(str(out_path), audio_vocals, 16_000, subtype="PCM_16")
                try:
                    file_generado.unlink()
                except Exception:
                    pass
                print(f"[OK] {filename} | GPU Activa [Proceso PID: {proc_id}]")
            else:
                raise RuntimeError(f"Fallo en Demucs: {filename}")

        else:  # wiener
            pct = clean_audio(str(f), str(out_path), passes=args.passes, impulse_removal=args.impulse_removal, impulse_kernel=args.impulse_kernel, impulse_threshold=args.impulse_threshold, impulse_passes=args.impulse_passes, hpss_kernel=args.hpss_kernel)
            print(f"[OK] {filename} | clipping: {pct:.2f}%")
        
        return True
    except Exception as e:
        print(f"[ERROR] {filename} -> {e}")
        return False


if __name__ == "__main__":
    import concurrent.futures
    # Forzar el método 'spawn' de Windows explícitamente antes de cualquier ejecución
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    _ROOT = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Limpia audios WAV")
    parser.add_argument("--method", choices=["wiener", "dfn3", "demucs"], default="wiener")
    parser.add_argument("--demucs-model", default="htdemucs", choices=["htdemucs", "htdemucs_ft"])
    parser.add_argument("--demucs-overlap", type=float, default=0.25)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--impulse-removal", action="store_true")
    parser.add_argument("--impulse-kernel", type=int, default=11)
    parser.add_argument("--impulse-threshold", type=float, default=2.5)
    parser.add_argument("--impulse-passes", type=int, default=2)
    parser.add_argument("--hpss-kernel", type=int, default=0)
    parser.add_argument("--atten-lim-db", type=float, default=75.0)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--reprocess-all", action="store_true")
    parser.add_argument("--audios-dir", default=str(_ROOT / "data" / "audios"))
    parser.add_argument("--clean-dir", default=None)
    parser.add_argument("--file-list", default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    audios_dir = Path(args.audios_dir)
    if args.clean_dir:
        clean_dir = Path(args.clean_dir)
    elif args.method == "dfn3":
        clean_dir = _ROOT / "data" / "clean_dfn"
    elif args.method == "demucs":
        clean_dir = _ROOT / "data" / "clean_demucs"
    else:
        clean_dir = _ROOT / "data" / "clean"

    os.makedirs(clean_dir, exist_ok=True)

    if args.file_list:
        fl = Path(args.file_list)
        if not fl.exists():
            raise SystemExit(f"[ERROR] --file-list no encontrado: {fl}")
        with open(fl, "r", encoding="utf-8") as f:
            names = {ln.strip() for ln in f if ln.strip()}
        wavs = sorted(p for p in (audios_dir / n for n in names) if p.exists())
    else:
        wavs = sorted(audios_dir.glob("*.wav"))

    if not wavs:
        print(f"No se encontraron archivos .wav")
    else:
        active_wavs = []
        skipped = 0
        fail = 0
        
        date_from = args.date_from
        date_to   = args.date_to

        for f in wavs:
            filename = f.name
            out_path = clean_dir / filename
            file_date = filename[:8]
            if file_date.isdigit():
                if date_from and file_date < date_from:
                    skipped += 1
                    continue
                if date_to and file_date > date_to:
                    skipped += 1
                    continue
            if not args.reprocess_all and out_path.exists():
                skipped += 1
                continue
            if f.stat().st_size == 0:
                fail += 1
                continue
            active_wavs.append(f)

        demucs_config = demucs_device = demucs_providers = None
        if args.method == "demucs":
            demucs_config, demucs_device, demucs_providers = _load_demucs_model(args.demucs_model)

        ok = 0
        if active_wavs:
            print(f"[INFO] Forzando paralelismo real: {args.workers} procesos independientes en GPU...")
            
            # El secreto mecánico: usamos initializer para despertar todas las sub-instancias simultáneamente
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=init_worker_context,
                initargs=(demucs_device,)
            ) as executor:
                
                future_to_file = {
                    executor.submit(
                        process_file_worker, f, clean_dir, args,
                        demucs_config, demucs_device, demucs_providers
                    ): f
                    for f in active_wavs
                }
                for future in concurrent.futures.as_completed(future_to_file):
                    if future.result():
                        ok += 1
                    else:
                        fail += 1
        
        for p in clean_dir.glob("tmp_onnx_*"):
            if p.is_dir():
                try: shutil.rmtree(p)
                except Exception: pass

        print("\n=========================")
        print(f"Procesados correctamente: {ok}")
        print(f"Omitidos:                 {skipped}")
        print(f"Fallidos:                 {fail}")