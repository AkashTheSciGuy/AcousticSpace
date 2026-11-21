r"""Inspect actual WAV headers for measured RIRs in a local SLR28 ZIP.

Run in Windows CMD from AcousticSpace/backend:
    .venv\Scripts\python.exe scripts\inspect_rir_audio.py ^
      datasets\rir_sources\rirs_noises.zip ^
      --output datasets\rir_sources\rir_audio_headers_v1.json

Only bounded audio members are decompressed into memory. The raw archive is
never modified, and no room labels or ML train/test splits are inferred.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_rir_archive import MAX_MEMBER_BYTES, audit  # noqa: E402


def inspect(path: Path) -> dict:
    inventory = audit(path)
    candidates = {item["archive_path"]: item for item in inventory["files"]}
    inspected = []
    problems = []
    sample_rates: Counter[str] = Counter()
    channels: Counter[str] = Counter()
    subtypes: Counter[str] = Counter()
    durations = []
    families: Counter[str] = Counter()
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            source = candidates.get(member.filename)
            if source is None:
                continue
            name = member.filename
            if member.is_dir() or member.file_size <= 0 or member.file_size > MAX_MEMBER_BYTES:
                problems.append({"archive_path": name, "error": "invalid member size"})
                continue
            try:
                # This reads no more than the advertised bounded size; zipfile
                # validates the archive member's CRC during decompression.
                with archive.open(member) as reader:
                    data = reader.read(MAX_MEMBER_BYTES + 1)
                if len(data) != member.file_size:
                    raise ValueError("decompressed member size mismatch")
                info = sf.info(io.BytesIO(data))
                if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0:
                    raise ValueError("invalid audio frame/rate/channel count")
                record = {
                    "archive_path": name,
                    "family": source["family"],
                    "recording_series": source["recording_series"],
                    "sample_rate_hz": info.samplerate,
                    "channels": info.channels,
                    "frames": info.frames,
                    "seconds": round(info.frames / info.samplerate, 6),
                    "format": info.format,
                    "subtype": info.subtype,
                }
            except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                problems.append({"archive_path": name, "error": str(exc)})
                continue
            inspected.append(record)
            families[record["family"]] += 1
            sample_rates[str(record["sample_rate_hz"])] += 1
            channels[str(record["channels"])] += 1
            subtypes[record["subtype"]] += 1
            durations.append(record["seconds"])
    return {
        "archive": str(path.resolve()),
        "candidate_files": inventory["candidate_files"],
        "valid_wav_headers": len(inspected),
        "invalid_wav_headers": len(problems),
        "valid_by_family": dict(sorted(families.items())),
        "sample_rates_hz": dict(sorted(sample_rates.items(), key=lambda kv: int(kv[0]))),
        "channel_counts": dict(sorted(channels.items(), key=lambda kv: int(kv[0]))),
        "subtypes": dict(sorted(subtypes.items())),
        "duration_seconds_min": min(durations) if durations else None,
        "duration_seconds_max": max(durations) if durations else None,
        "inspection_errors": problems,
        "files": inspected,
        "note": (
            "Header inspection only. Filename-derived recording series are not "
            "verified independent physical rooms. No model was trained or evaluated."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON report path (will not overwrite)")
    args = parser.parse_args(argv)
    try:
        report = inspect(args.archive)
        if args.output:
            target = args.output.resolve()
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite report: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as writer:
                json.dump(report, writer, indent=2)
            print(f"Detailed report written: {target}")
        print(json.dumps({key: value for key, value in report.items()
                          if key != "files"}, indent=2))
        return 0 if not report["inspection_errors"] else 3
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"RIR WAV inspection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
