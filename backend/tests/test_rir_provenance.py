"""Strict structural provenance checks with self-declared room metadata."""
from __future__ import annotations

import csv
import hashlib

import numpy as np
import pytest
import soundfile as sf

from scripts.validate_rir_provenance import REQUIRED, main, validate


@pytest.fixture
def curated(tmp_path):
    root = tmp_path / "audio_root"
    root.mkdir()
    (root / "rirs").mkdir()
    rows = []
    count = 0
    for split in ("train", "val", "test"):
        for room_index in range(2):
            room_id = f"{split}_room_{room_index}"
            for position in range(2):
                count += 1
                path = root / "rirs" / f"rir_{count:02}.wav"
                samples = np.zeros(2000, dtype=float)
                samples[0] = 0.5
                samples[150 + count * 3] = 0.1
                samples[800 + count * 7] = .01
                sf.write(path, samples, 16000, format="WAV", subtype="PCM_16")
                rows.append({
                    "path": path.relative_to(root).as_posix(),
                    "rir_id": f"verified_candidate_{count:02}",
                    "site_id": "test_lab",
                    "physical_room_id": room_id,
                    "session_id": f"session_{room_id}",
                    "mic_device": "test_microphone",
                    "mic_position": f"position_{position}",
                    "source_position": f"source_{position}",
                    "provenance_reference": f"https://example.invalid/{room_id}",
                    "split": split,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                })
    manifest = tmp_path / "curated.csv"
    write_rows(manifest, rows)
    return root, manifest, rows


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUIRED)
        writer.writeheader()
        writer.writerows(rows)


def test_audit_accepts_nonleaking_curated_metadata_but_does_not_certify_room_identity(curated):
    root, manifest, _ = curated
    report = validate(root, manifest)
    assert report["status"] == "metadata_checks_passed"
    assert report["records_checked"] == 12
    assert report["rooms_by_split"] == {"train": 2, "val": 2, "test": 2}
    assert report["technical_audio_hashes_verified"] is True
    assert report["physical_provenance_independently_verified"] is False


def test_audit_detects_room_leakage(curated):
    root, manifest, rows = curated
    rows[4]["physical_room_id"] = rows[0]["physical_room_id"]
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="Physical room leakage"):
        validate(root, manifest)


def test_audit_detects_duplicate_hash(curated):
    root, manifest, rows = curated
    rows[1]["sha256"] = rows[0]["sha256"]
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="same WAV content"):
        validate(root, manifest)


def test_audit_detects_duplicate_room_position(curated):
    root, manifest, rows = curated
    rows[1]["mic_position"] = rows[0]["mic_position"]
    rows[1]["source_position"] = rows[0]["source_position"]
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="distinct mic/source positions"):
        validate(root, manifest)


def test_audit_detects_unknown_provenance_reference(curated):
    root, manifest, rows = curated
    rows[0]["provenance_reference"] = "TBD"
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="non-placeholder"):
        validate(root, manifest)


def test_audit_rejects_unsafe_path(curated):
    root, manifest, rows = curated
    rows[0]["path"] = "../other.wav"
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="invalid relative WAV filename"):
        validate(root, manifest)


def test_audit_detects_wrong_checksum(curated):
    root, manifest, rows = curated
    rows[0]["sha256"] = "0" * 64
    write_rows(manifest, rows)
    with pytest.raises(ValueError, match="does not match"):
        validate(root, manifest)


def test_audit_requires_all_three_splits(curated):
    root, manifest, rows = curated
    remaining = [row for row in rows if row["split"] != "test"]
    write_rows(manifest, remaining)
    with pytest.raises(ValueError, match="test: at least two"):
        validate(root, manifest)


def test_cli_refuses_to_overwrite_existing_report(curated, tmp_path):
    root, manifest, _ = curated
    target = tmp_path / "audit.json"
    args = [str(root), str(manifest), "--output", str(target)]
    assert main(args) == 0
    original = target.read_bytes()
    assert main(args) == 2
    assert target.read_bytes() == original
