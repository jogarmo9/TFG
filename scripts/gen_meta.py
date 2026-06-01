"""Genera meta.json para sesiones móvil que no lo tienen, usando primer trackpoint del GPX."""
import json
import gpxpy
from pathlib import Path
from datetime import timezone

mobile_dir = Path(__file__).parent.parent / "data" / "mobile"
created = []

for session_dir in sorted(mobile_dir.iterdir()):
    if not session_dir.is_dir():
        continue
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        print(f"[SKIP] {session_dir.name}: meta.json ya existe")
        continue

    gpx_files = list(session_dir.glob("*.gpx"))
    if not gpx_files:
        print(f"[SKIP] {session_dir.name}: sin GPX")
        continue

    with open(gpx_files[0], encoding="utf-8") as f:
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

    if not first_time:
        print(f"[SKIP] {session_dir.name}: sin timestamps en GPX")
        continue

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
    created.append(f"{session_dir.name}: {meta['audio_start_utc']}")

print(f"\nCreados {len(created)} meta.json:")
for c in created:
    print(f"  {c}")
