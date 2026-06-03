"""
vad_candidates.py
=================
Etapa A (prefiltro de candidatos Speech) basado en VAD (silero-vad).

Por qué VAD y no YOLO-Speech: el ruido musical del Wiener dispara Speech FP en
YOLO — usar ese detector como prefiltro selecciona justo los ficheros propensos
a FP (~62% del dataset). silero-vad está entrenado para detectar voz humana real,
no dispara con ruido de motor/tráfico → corta candidatos fuerte y apunta a voz real.

Flujo:
  1) SCORE (pesado, una vez): puntúa cada WAV de --in-dir con silero → CSV con
     max_prob y frac_speech por fichero (data/processed/speech_vad_scores.csv).
  2) SELECT (instantáneo, desde cache): tabla de barrido (nº candidatos por umbral)
     y volcado de speech_candidates.txt al --threshold elegido.

Uso:
  # Paso 1 — puntuar (sobre clean_cand = Wiener-solo-sin-HPSS)
  python scripts/vad_candidates.py score --in-dir data/clean_cand
  # Paso 2 — ver barrido y elegir umbral
  python scripts/vad_candidates.py select --threshold 0.5
"""

import argparse
import csv
from pathlib import Path

import numpy as np

_ROOT          = Path(__file__).parent.parent
CLEAN_CAND_DIR = _ROOT / "data" / "clean_cand"
SCORES_CSV     = _ROOT / "data" / "processed" / "speech_vad_scores.csv"
CANDIDATES_TXT = _ROOT / "data" / "processed" / "speech_candidates.txt"

SR          = 16_000
WINDOW      = 512               # silero requiere 512 muestras @ 16kHz
SWEEP       = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def _score(in_dir: Path, scores_csv: Path):
    import torch
    from silero_vad import load_silero_vad
    import librosa as lb

    model = load_silero_vad()
    model.eval()

    wavs = sorted(in_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"[ERROR] No hay .wav en {in_dir}")
    print(f"[SCORE] {len(wavs)} WAVs en {in_dir.name}/ | silero-vad @ {SR}Hz")

    scores_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_file", "max_prob", "frac_speech", "n_windows"])
        for n, wav in enumerate(wavs, 1):
            try:
                audio, _ = lb.load(str(wav), sr=SR, mono=True)
                model.reset_states()
                probs = []
                for i in range(0, len(audio) - WINDOW + 1, WINDOW):
                    chunk = torch.from_numpy(audio[i:i + WINDOW]).float()
                    probs.append(model(chunk, SR).item())
                if probs:
                    arr = np.asarray(probs)
                    max_p = float(arr.max())
                    frac  = float((arr >= 0.5).mean())
                    nw    = len(probs)
                else:
                    max_p = frac = 0.0
                    nw = 0
                w.writerow([wav.name, f"{max_p:.4f}", f"{frac:.4f}", nw])
            except Exception as e:
                print(f"  [ERROR] {wav.name} – {e}")
            if n % 500 == 0:
                print(f"  ... {n}/{len(wavs)}")
    print(f"[SCORE] Puntuaciones -> {scores_csv}")


def _load_scores(scores_csv: Path):
    if not scores_csv.exists():
        raise SystemExit(f"[ERROR] {scores_csv} no existe. Ejecutar 'score' primero.")
    rows = []
    with open(scores_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["source_file"], float(r["max_prob"])))
    return rows


def _select(scores_csv: Path, threshold: float, out_txt: Path):
    rows = _load_scores(scores_csv)
    total = len(rows)
    probs = np.asarray([p for _, p in rows])

    print(f"[SELECT] {total} ficheros puntuados. Barrido de conteo (por max_prob):")
    print(f"  {'umbral':>7} | {'candidatos':>10} | {'% dataset':>9}")
    for t in SWEEP:
        c = int((probs >= t).sum())
        print(f"  {t:>7.2f} | {c:>10} | {c/total*100:>8.1f}%")

    cands = sorted(fn for fn, p in rows if p >= threshold)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(cands) + ("\n" if cands else ""))
    print(f"\n[SELECT] umbral={threshold}: {len(cands)}/{total} "
          f"({len(cands)/total*100:.1f}%) -> {out_txt}")


def main():
    p = argparse.ArgumentParser(description="Prefiltro de candidatos Speech vía silero-vad")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("score", help="Puntúa WAVs con silero (pesado, una vez)")
    ps.add_argument("--in-dir", default=str(CLEAN_CAND_DIR),
                    help="Carpeta de WAVs a puntuar (default: data/clean_cand)")
    ps.add_argument("--scores-csv", default=str(SCORES_CSV))

    pl = sub.add_parser("select", help="Barrido de conteo + volcar candidatos (desde cache)")
    pl.add_argument("--threshold", type=float, default=0.5,
                    help="Umbral max_prob para marcar candidato (default: 0.5)")
    pl.add_argument("--scores-csv", default=str(SCORES_CSV))
    pl.add_argument("--out", default=str(CANDIDATES_TXT))

    args = p.parse_args()
    if args.cmd == "score":
        _score(Path(args.in_dir), Path(args.scores_csv))
    else:
        _select(Path(args.scores_csv), args.threshold, Path(args.out))


if __name__ == "__main__":
    main()
