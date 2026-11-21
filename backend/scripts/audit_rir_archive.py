"""Read-only inventory of measured-RIR candidates in the OpenSLR SLR28 ZIP.

From backend:
    .venv\Scripts\python.exe scripts\audit_rir_archive.py ^
      datasets\rir_sources\rirs_noises.zip ^
      --output datasets\rir_sources\rir_archive_audit.json

Recording series do NOT establish independent physical-room provenance.
No audio is extracted or modified.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = "RIRS_NOISES/real_rirs_isotropic_noises/"
FAMILIES = {
    "RVB2014_type1_rir_": "RVB2014",
    "RWCP_type1_rir_": "RWCP",
    "air_type1_air_binaural_": "AIR",
}
MAX_ENTRIES = 100_000
MAX_MEMBER_BYTES = 64 * 1024 * 1024


def recording_series(family: str, name: str) -> str:
    """Inventory label only; it is NOT a verified physical-room ID."""
    if family == "RVB2014":
        match = re.fullmatch(
            r"RVB2014_type1_rir_((?:small|medium|large)room\d+)_(?:near|far)_angl[ab]\.wav",
            name, re.IGNORECASE,
        )
        return match.group(1).lower() if match else "unrecognized"
    if family == "RWCP":
        match = re.fullmatch(
            r"RWCP_type1_rir_(circle|cirline)_([a-z0-9]+)_imp\d+\.wav",
            name, re.IGNORECASE,
        )
        return f"{match.group(1).lower()}_{match.group(2).lower()}" if match else "unrecognized"
    if family == "AIR":
        suffix = name.removeprefix("air_type1_air_binaural_").lower()
        for prefix in ("aula_carolina", "booth", "lecture", "meeting", "office", "stairway"):
            if suffix.startswith(prefix + "_"):
                return prefix
        return "other_or_phone"
    raise ValueError("Unsupported family")


def audit(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"ZIP not found: {path}")
    files = []
    groups = {name: Counter() for name in FAMILIES.values()}
    unknown = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) > MAX_ENTRIES:
            raise ValueError("Archive has too many entries")
        for entry in members:
            if entry.is_dir() or not entry.filename.startswith(ROOT):
                continue
            name = entry.filename[len(ROOT):]
            if "/" in name or "\\" in name or not name.lower().endswith(".wav"):
                continue
            family = next((label for prefix, label in FAMILIES.items()
                           if name.startswith(prefix)), None)
            if family is None:
                unknown += 1
                continue
            if entry.file_size > MAX_MEMBER_BYTES or entry.file_size <= 0:
                raise ValueError(f"Candidate file exceeds size bounds: {name}")
            if entry.flag_bits & 1:
                raise ValueError(f"Encrypted candidate WAV: {name}")
            series = recording_series(family, name)
            groups[family][series] += 1
            files.append({
                "archive_path": entry.filename,
                "family": family,
                "recording_series": series,
                "bytes_uncompressed": entry.file_size,
                "zip_crc32": f"{entry.CRC:08x}",
            })
    if not files:
        raise ValueError("No supported measured-RIR candidates found")
    return {
        "archive": str(path),
        "candidate_files": len(files),
        "family_counts": dict(Counter(item["family"] for item in files)),
        "recording_series_counts": {
            family: dict(sorted(counts.items())) for family, counts in groups.items()
        },
        "other_real_wavs_not_in_selected_families": unknown,
        "files": files,
        "warning": (
            "Filename-derived recording series are NOT verified physical-room identities. "
            "Do not infer independent rooms, split groups, or forensic accuracy from this audit."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, help="Write a new detailed JSON report")
    args = parser.parse_args(argv)
    try:
        report = audit(args.archive)
        if args.output is not None:
            target = args.output.resolve()
            if target.exists():
                raise FileExistsError(f"Refusing to overwrite report: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2)
            print(f"Detailed audit written: {target}")
        print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))
        return 0
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"RIR archive audit failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
