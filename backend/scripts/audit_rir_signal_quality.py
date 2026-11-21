r"""Inspect measured-RIR signal quality without modifying the original archive.

Windows CMD from backend:
    .venv\Scripts\python.exe scripts\audit_rir_signal_quality.py ^
      datasets\rir_sources\rirs_noises.zip ^
      --output datasets\rir_sources\rir_signal_quality_v1.json --channel 1

Uses one microphone channel (1-based), never averages microphone channels.
The output flags *possible* quality issues; it does NOT establish that a
recording is a physically independent room, nor select train/test splits.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_rir_archive import MAX_MEMBER_BYTES, audit  # noqa: E402

SAMPLE_RATE = 16_000
MIN_RIR_FRAMES = 256
MAX_RIR_FRAMES = SAMPLE_RATE * 2
SHORT_DURATION_SECONDS = 0.25
HIGH_TAIL_ENERGY_FRACTION = 0.10


def inspect_samples(samples: np.ndarray, sample_rate: int) -> dict:
    """Compute descriptive metrics; flags are heuristics, not a quality label."""
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("Selected channel must contain samples")
    if not np.isfinite(samples).all():
        raise ValueError("Selected channel contains non-finite values")
    absolute = np.abs(samples.astype(np.float64, copy=False))
    peak = float(np.max(absolute))
    rms = float(np.sqrt(np.mean(np.square(absolute))))
    peak_index = int(np.argmax(absolute))
    total_energy = float(np.sum(np.square(absolute)))
    tail_length = max(1, int(np.ceil(samples.size * 0.1)))
    tail_fraction = (
        float(np.sum(np.square(absolute[-tail_length:])) / total_energy)
        if total_energy > 0 else None
    )
    flags: list[str] = []
    if peak < 1e-6 or rms < 1e-7:
        flags.append("near_silent_channel")
    if samples.size / sample_rate < SHORT_DURATION_SECONDS:
        flags.append("shorter_than_250_ms")
    if not MIN_RIR_FRAMES <= samples.size <= MAX_RIR_FRAMES:
        flags.append("outside_existing_generator_duration_limits")
    if sample_rate != SAMPLE_RATE:
        flags.append("not_16khz")
    if peak_index >= int(samples.size * 0.9):
        flags.append("peak_in_final_10_percent")
    if tail_fraction is not None and tail_fraction >= HIGH_TAIL_ENERGY_FRACTION:
        flags.append("high_last_10_percent_energy")
    # A short run of saturated samples is stronger clipping evidence than one
    # naturally large PCM sample; this is still only a conservative heuristic.
    if int(np.count_nonzero(absolute >= 0.999)) >= 5:
        flags.append("possible_full_scale_clipping")
    return {
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "peak_sample_index": peak_index,
        "peak_seconds": round(peak_index / sample_rate, 6),
        "last_10_percent_energy_fraction": tail_fraction,
        "flags": flags,
        "note": (
            "Amplitude and tail-energy thresholds are heuristic. "
            "High tail energy may reflect sustained ambient noise, a truncated "
            "reverberant tail, or other factors; it does not prove corruption."
        ),
    }


def audit_signals(archive_path: Path, *, channel: int = 1) -> dict:
    """Read one channel per candidate WAV, bounded by the archive inventory."""
    if channel < 1:
        raise ValueError("--channel must be a 1-based positive integer")
    inventory = audit(archive_path)
    entries = {item["archive_path"]: item for item in inventory["files"]}
    findings = []
    errors = []
    flag_counts: Counter[str] = Counter()
    unflagged_by_family: Counter[str] = Counter()
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            original = entries.get(entry.filename)
            if original is None:
                continue
            if entry.file_size <= 0 or entry.file_size > MAX_MEMBER_BYTES:
                errors.append({"archive_path": entry.filename, "error": "member too large"})
                continue
            try:
                with archive.open(entry) as source:
                    data = source.read(MAX_MEMBER_BYTES + 1)
                if len(data) != entry.file_size:
                    raise ValueError("decompressed length mismatch")
                audio, rate = sf.read(
                    io.BytesIO(data), dtype="float32", always_2d=True,
                )
                if not 1 <= channel <= audio.shape[1]:
                    raise ValueError(
                        f"selected microphone channel {channel} exceeds "
                        f"available {audio.shape[1]}"
                    )
                mono = audio[:, channel - 1]
                metrics = inspect_samples(mono, rate)
            except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
                errors.append({"archive_path": entry.filename, "error": str(exc)})
                continue
            for flag in metrics["flags"]:
                flag_counts[flag] += 1
            if not metrics["flags"]:
                unflagged_by_family[original["family"]] += 1
            findings.append({
                "archive_path": entry.filename,
                "family": original["family"],
                "recording_series": original["recording_series"],
                "selected_channel_1_based": channel,
                "selected_audio_sha256": hashlib.sha256(mono.tobytes()).hexdigest(),
                "source_channels": int(audio.shape[1]),
                "sample_rate_hz": int(rate),
                "frames": int(mono.size),
                "duration_seconds": round(mono.size / rate, 6),
                **metrics,
            })
    return {
        "archive": str(archive_path.resolve()),
        "candidate_files": inventory["candidate_files"],
        "analyzed_files": len(findings),
        "decode_errors": len(errors),
        "selected_microphone_channel_1_based": channel,
        "unflagged_by_family": dict(sorted(unflagged_by_family.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "errors": errors,
        "files": findings,
        "warning": (
            "Heuristic flags are review prompts, NOT dataset exclusion labels. "
            "No verified independent room IDs or train/validation/test splits are "
            "inferred. No model is trained or scored."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--output", type=Path, help="New report file; refuses overwrite")
    args = parser.parse_args(argv)
    try:
        report = audit_signals(args.archive, channel=args.channel)
        if args.output is not None:
            destination = args.output.resolve()
            if destination.exists():
                raise FileExistsError(f"Refusing to overwrite report: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("x", encoding="utf-8") as writer:
                json.dump(report, writer, indent=2)
            print(f"Detailed report written: {destination}")
        print(json.dumps({key: value for key, value in report.items()
                          if key != "files"}, indent=2))
        return 0 if not report["errors"] else 3
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        print(f"RIR signal audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
