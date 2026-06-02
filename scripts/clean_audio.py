import argparse
import warnings
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from numpy.fft import rfft, irfft
import os
from pathlib import Path


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

        gain = np.maximum(
            0,
            (mag - nr_strength * noise) / (mag + 1e-8)
        )

        cleaned = irfft(
            gain * mag * np.exp(1j * np.angle(spec))
        )

        out[i:i + win] += cleaned * w
        cnt[i:i + win] += w

    return out / np.where(cnt < 1e-8, 1, cnt)


def remove_impulses(audio: np.ndarray,
                    kernel_size: int = 11,
                    threshold_sigma: float = 2.5,
                    passes: int = 2,
                    kernel_size_p2: int = 15,
                    threshold_sigma_p2: float = 1.5) -> np.ndarray:
    """Elimina impulsos (clicks/crispeos) por filtro mediana.

    P1 (global RMS): detecta picos grandes.
    P2+ (global RMS, parametros independientes): captura residuos mas finos.
    """
    from scipy.signal import medfilt as _medfilt

    a = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(a ** 2)))
    if rms < 1e-10:
        return a

    # Pasada 1
    med  = _medfilt(a.astype(np.float64), kernel_size=kernel_size).astype(np.float32)
    mask = np.abs(a - med) > threshold_sigma * rms
    a    = a.copy()
    a[mask] = med[mask]

    # Pasadas adicionales con parametros P2
    for _ in range(passes - 1):
        rms2  = float(np.sqrt(np.mean(a ** 2)))
        med2  = _medfilt(a.astype(np.float64), kernel_size=kernel_size_p2).astype(np.float32)
        mask2 = np.abs(a - med2) > threshold_sigma_p2 * rms2
        a[mask2] = med2[mask2]

    return a


def _load_dfn3_model():
    """Carga modelo DFN3 una sola vez. Usa CUDA si disponible."""
    import torch
    from df.enhance import init_df
    model, df_state, _ = init_df()
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("[DFN3] CUDA disponible — modelo en GPU")
    else:
        print("[DFN3] CUDA no disponible — usando CPU")
    return model, df_state


def clean_audio_dfn3(path_in: str, path_out: str, atten_lim_db: float = 75.0,
                     model=None, df_state=None):
    """Limpia audio con DeepFilterNet3. Salida: WAV mono 16kHz PCM_16.

    atten_lim_db: máximo dB de supresión (75 = default, 100 = sin límite).
    model/df_state: pasar para reusar modelo ya cargado (evita recarga por fichero).
    """
    import torch
    import soundfile as sf
    import librosa
    from df.enhance import enhance, init_df, load_audio

    if model is None or df_state is None:
        model, df_state, _ = init_df()

    os.makedirs(os.path.dirname(path_out) or ".", exist_ok=True)
    audio, _ = load_audio(path_in, sr=df_state.sr())
    if torch.cuda.is_available():
        audio = audio.to("cuda")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*sinc_interpolation.*", category=UserWarning)
        enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim_db)
    arr = enhanced.squeeze().cpu().numpy() if hasattr(enhanced, "numpy") else np.array(enhanced.cpu()).squeeze()
    if df_state.sr() != 16_000:
        arr = librosa.resample(arr, orig_sr=df_state.sr(), target_sr=16_000, res_type='kaiser_best')
    sf.write(path_out, arr, 16_000, subtype="PCM_16")


def clean_audio(path_in,
                path_out,
                clip_thresh=0.85,
                nr_strength=0.85,
                lp_cutoff=7500,
                hp_cutoff=None,
                passes=2,
                impulse_removal=False,
                impulse_kernel=11,
                impulse_threshold=2.5,
                impulse_passes=2):

    # =========================
    # CARGA SEGURA DEL WAV
    # =========================

    try:
        sr, data = wavfile.read(path_in)

    except Exception as e:
        raise ValueError(
            f"No se pudo leer el WAV '{path_in}': {e}"
        )

    # =========================
    # NORMALIZACION
    # =========================

    stereo = data.ndim == 2

    channels = (
        [data[:, i] for i in range(data.shape[1])]
        if stereo
        else [data]
    )

    results = []

    for ch in channels:

        audio = ch.astype(np.float32) / 32768.0

        pct_total = 0.0

        for p in range(passes):

            # =========================
            # FASE 1: DECLIP
            # =========================

            thresh = clip_thresh - p * 0.03

            audio, pct = _declip_pass(audio, thresh)

            if p == 0:
                pct_total = pct

            # =========================
            # FASE 2: ANTI-IMPULSO (opcional)
            # =========================

            if impulse_removal and p == 0:
                audio = remove_impulses(audio,
                                        kernel_size=impulse_kernel,
                                        threshold_sigma=impulse_threshold,
                                        passes=impulse_passes)

            # =========================
            # FASE 3: WIENER
            # =========================

            audio = _wiener_pass(audio, nr_strength)

        # =========================
        # FASE 3: HIGHPASS (opcional, elimina rumble <hp_cutoff Hz)
        # =========================

        if hp_cutoff is not None:
            b, a = butter(4, hp_cutoff / (sr / 2), btype='high')
            audio = filtfilt(b, a, audio)

        # =========================
        # FASE 4: LOWPASS FINAL
        # =========================

        lp_norm = min(lp_cutoff, sr / 2 - 1) / (sr / 2)
        b, a = butter(4, lp_norm, btype='low')

        audio = filtfilt(b, a, audio)

        results.append((
            np.clip(audio, -1, 1),
            pct_total
        ))

    # =========================
    # RECONSTRUIR OUTPUT
    # =========================

    out_data = (
        np.stack([r[0] for r in results], axis=1)
        if stereo
        else results[0][0]
    )

    # =========================
    # CREAR CARPETA DESTINO
    # =========================

    os.makedirs(
        os.path.dirname(path_out),
        exist_ok=True
    )

    # =========================
    # GUARDAR WAV
    # =========================

    wavfile.write(
        path_out,
        sr,
        (out_data * 32767).astype(np.int16)
    )

    return results[0][1]


if __name__ == "__main__":

    _ROOT = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(description="Limpia audios WAV con filtro Wiener + declipping o DeepFilterNet3")
    parser.add_argument("--method", choices=["wiener", "dfn3"], default="wiener",
                        help="Método de limpieza: 'wiener' (default) o 'dfn3' (DeepFilterNet3)")
    parser.add_argument("--passes", type=int, default=2, help="Pasadas Wiener (default: 2, ignorado en dfn3)")
    parser.add_argument("--impulse-removal", action="store_true",
                        help="Aplica remove_impulses antes de Wiener (elimina clicks/crispeos)")
    parser.add_argument("--impulse-kernel", type=int, default=11)
    parser.add_argument("--impulse-threshold", type=float, default=2.5)
    parser.add_argument("--impulse-passes", type=int, default=2)
    parser.add_argument("--atten-lim-db", type=float, default=75.0,
                        help="DFN3: máximo dB de supresión (default: 75.0)")
    parser.add_argument("--date-from", default=None,
                        help="Procesar solo archivos desde esta fecha (formato YYYYMMDD, inclusive)")
    parser.add_argument("--date-to", default=None,
                        help="Procesar solo archivos hasta esta fecha (formato YYYYMMDD, inclusive)")
    parser.add_argument("--reprocess-all", action="store_true",
                        help="Reprocesa todos los archivos aunque ya existan en clean-dir")
    parser.add_argument("--audios-dir", default=str(_ROOT / "data" / "audios"),
                        help="Carpeta con WAVs de entrada (default: data/audios)")
    parser.add_argument("--clean-dir", default=None,
                        help="Carpeta de salida (default: data/clean para wiener, data/clean_dfn para dfn3)")
    args = parser.parse_args()

    audios_dir = Path(args.audios_dir)
    if args.clean_dir:
        clean_dir = Path(args.clean_dir)
    elif args.method == "dfn3":
        clean_dir = _ROOT / "data" / "clean_dfn"
    else:
        clean_dir = _ROOT / "data" / "clean"

    os.makedirs(clean_dir, exist_ok=True)

    wavs = sorted(audios_dir.glob("*.wav"))

    if not wavs:
        print(f"No se encontraron archivos .wav en {audios_dir}/")

    else:

        print(f"[INFO] Método: {args.method.upper()} | Salida: {clean_dir}/")
        ok = 0
        fail = 0
        skipped = 0

        # Carga DFN3 una sola vez antes del loop
        dfn3_model = dfn3_state = None
        if args.method == "dfn3":
            dfn3_model, dfn3_state = _load_dfn3_model()

        date_from = args.date_from  # e.g. "20260311"
        date_to   = args.date_to    # e.g. "20260424"

        for f in wavs:

            filename = f.name
            out_path = clean_dir / filename

            # Filter by date embedded in filename (YYYYMMDD_...)
            file_date = filename[:8]
            if file_date.isdigit():
                if date_from and file_date < date_from:
                    skipped += 1
                    continue
                if date_to and file_date > date_to:
                    skipped += 1
                    continue

            if not args.reprocess_all and out_path.exists():
                print(f"[SKIP] {filename} (ya limpio)")
                skipped += 1
                continue

            if f.stat().st_size == 0:
                print(f"[VACIO] {filename}")
                fail += 1
                continue

            try:
                if args.method == "dfn3":
                    clean_audio_dfn3(str(f), str(out_path), atten_lim_db=args.atten_lim_db,
                                     model=dfn3_model, df_state=dfn3_state)
                    print(f"[OK] {filename} | atten_lim_db={args.atten_lim_db}")
                else:
                    pct = clean_audio(str(f), str(out_path),
                                      passes=args.passes,
                                      impulse_removal=args.impulse_removal,
                                      impulse_kernel=args.impulse_kernel,
                                      impulse_threshold=args.impulse_threshold,
                                      impulse_passes=args.impulse_passes)
                    print(f"[OK] {filename} | clipping detectado: {pct:.2f}%")
                ok += 1

            except Exception as e:

                print(f"[ERROR] {filename}")
                print(e)

                fail += 1

        print("\n=========================")
        print(f"Procesados correctamente: {ok}")
        print(f"Omitidos (ya existían):   {skipped}")
        print(f"Fallidos:                 {fail}")
        print(f"Salida en {clean_dir}/")