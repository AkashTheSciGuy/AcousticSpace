r"""Compare acoustic consistency using four separately provided reference WAVs.

Windows CMD from backend:
  .venv\Scripts\python.exe scripts\compare_reference_rir.py ^
    --dry-speech datasets\dry_speech.wav ^
    --wet-speech datasets\wet_speech_stem.wav ^
    --dry-ambience datasets\dry_ambience.wav ^
    --wet-ambience datasets\wet_ambience_stem.wav ^
    --output models\reference_comparison_v1.json

All input WAVs must be mono 16-kHz, finite, <=12 s. The script refuses
to overwrite an existing report or derive a real/fake verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.rir_features import SAMPLE_RATE, reference_consistency_features  # noqa: E402

MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_SECONDS = 12
ROLES = ("dry_speech", "wet_speech", "dry_ambience", "wet_ambience")


def load_reference(path: Path) -> tuple[np.ndarray, str]:
    if not path.is_file() or path.suffix.lower() != ".wav":
        raise ValueError(f"Missing WAV input: {path}")
    if not 0 < path.stat().st_size <= MAX_SOURCE_BYTES:
        raise ValueError(f"WAV exceeds 4 MiB or is empty: {path.name}")
    with sf.SoundFile(str(path)) as source:
        if source.format != "WAV" or source.channels != 1 or source.samplerate != SAMPLE_RATE:
            raise ValueError(f"Expected mono 16-kHz WAV: {path.name}")
        if not 256 <= source.frames <= MAX_SECONDS * SAMPLE_RATE:
            raise ValueError(f"WAV duration out of bounds: {path.name}")
        audio = source.read(dtype="float64", always_2d=False)
    if not np.isfinite(audio).all() or np.sqrt(np.mean(audio * audio)) < 1e-7:
        raise ValueError(f"Invalid, silent or nonfinite WAV: {path.name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return audio, digest


def compare(paths: dict[str, Path]) -> dict:
    if set(paths) != set(ROLES):
        raise ValueError("Exactly four labeled dry/wet signal paths are required.")
    recordings = {}
    digests = {}
    for role in ROLES:
        recordings[role], digests[role] = load_reference(paths[role])
    features = reference_consistency_features(*(recordings[role] for role in ROLES))
    return {
        "status": "analyzed",
        "input_type": "four_reference_aided_audio_signals",
        "features": features.to_dict(),
        "sample_rate_hz": SAMPLE_RATE,
        "input_sha256": digests,
        "prediction": None,
        "warning": (
            "Known isolated speech and ambience stems + dry sources required. "
            "This is not blind room impulse response estimation, source separation, "
            "real/fake detection or calibrated mismatch classification. "
            "Do not infer forensic accuracy from this report."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ROLES:
        parser.add_argument("--" + role.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    try:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        report = compare({role: getattr(args, role) for role in ROLES})
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    except (FileExistsError, OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"Reference-aided RIR comparison failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
