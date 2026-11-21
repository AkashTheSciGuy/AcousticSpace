"""Pipeline integrity checks with generated fixtures (not forensic accuracy tests)."""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from scripts.audit_rir_archive import audit
from scripts.prepare_aru_rir_inputs import prepare as prepare_aru
from scripts.prepare_measured_rirs import prepare as prepare_rirs


ROOMS = ("smallroom1", "mediumroom1", "smallroom2",
         "largeroom1", "mediumroom2", "largeroom2")
POSITIONS = ("near_angla", "near_anglb", "far_angla", "far_anglb")


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def source_rirs(tmp_path):
    folder = tmp_path / "original"
    folder.mkdir()
    for room_i, room in enumerate(ROOMS):
        for position_i, position in enumerate(POSITIONS):
            signal = np.zeros((16000, 8), dtype=np.float32)
            signal[0] = 0.5
            signal[100 + 100 * room_i + 20 * position_i] = 0.2
            signal[1500 + 10 * room_i + 30 * position_i] = 0.1
            signal[:, 1] *= 0.7
            sf.write(folder / f"RVB2014_type1_rir_{room}_{position}.wav",
                     signal, 16000, subtype="PCM_16")
    return folder


def test_prepares_measured_rirs_without_downmixing_or_overwriting(source_rirs, tmp_path):
    output = tmp_path / "prepared"
    result = prepare_rirs(source_rirs, output, channel=2)
    assert result["prepared_files"] == 24
    assert result["rirs_per_split"] == {"train": 8, "val": 8, "test": 8}
    rows = _csv(output / "responses.csv")
    assert len(rows) == 24
    for row in rows:
        info = sf.info(output / row["path"])
        assert (info.samplerate, info.channels, info.frames) == (16000, 1, 16000)
    example = next(source_rirs.glob("*.wav"))
    original, _ = sf.read(example, always_2d=True)
    mono, _ = sf.read(output / "rirs" / (example.stem + "_ch02.wav"))
    np.testing.assert_allclose(mono, original[:, 1], atol=1e-7)
    with pytest.raises(FileExistsError):
        prepare_rirs(source_rirs, output, channel=2)


def test_aru_importer_splits_speakers_and_records_source_provenance(source_rirs, tmp_path):
    prepared_rirs = tmp_path / "prepared_rirs"
    prepare_rirs(source_rirs, prepared_rirs, channel=1)
    archive = tmp_path / "speech.zip"
    sample_rate = 32000
    t = np.arange(3 * sample_rate) / sample_rate
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for speaker_id in range(1, 7):
            for clip in (1, 2):
                speech = 0.2 * np.sin(2 * np.pi * (120 + speaker_id + clip) * t)
                speech[:sample_rate // 3] = 0
                file = tmp_path / f"ID{speaker_id:02d}_ARU_Standard speech sentence {clip}.wav"
                sf.write(file, speech, sample_rate, subtype="PCM_16")
                z.write(file, file.name)
                file.unlink()
    output = tmp_path / "prepared_aru"
    result = prepare_aru(archive, prepared_rirs, output, seconds=1)
    assert result["prepared_speech_files"] == 12
    assert result["prepared_rirs_per_split"] == {"train": 8, "val": 8, "test": 8}
    rows = _csv(output / "speech.csv")
    speaker_split = {}
    for row in rows:
        speaker_split.setdefault(row["speaker_id"], set()).add(row["split"])
        info = sf.info(output / row["path"])
        assert (info.samplerate, info.channels, info.frames) == (16000, 1, 16000)
    assert all(len(group) == 1 for group in speaker_split.values())
    provenance = json.loads((output / "preparation_metadata.json").read_text())["speech_provenance"]
    assert len(provenance) == 12
    assert all(len(item["source_member_sha256"]) == 64 for item in provenance)
    assert all(item["selected_start_frame"] >= 0 for item in provenance)
    with pytest.raises(FileExistsError):
        prepare_aru(archive, prepared_rirs, output)


def test_archive_inventory_does_not_infer_rooms(tmp_path):
    archive = tmp_path / "rirs.zip"
    prefix = "RIRS_NOISES/real_rirs_isotropic_noises/"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("RVB2014_type1_rir_smallroom1_near_angla.wav",
                     "RWCP_type1_rir_circle_ane_imp000.wav",
                     "air_type1_air_binaural_booth_0_1.wav"):
            z.writestr(prefix + name, b"example-audio-bytes")
    report = audit(archive)
    assert report["candidate_files"] == 3
    assert report["family_counts"] == {"RVB2014": 1, "RWCP": 1, "AIR": 1}
    assert report["recording_series_counts"]["RVB2014"] == {"smallroom1": 1}
    assert "NOT verified physical-room" in report["warning"]
