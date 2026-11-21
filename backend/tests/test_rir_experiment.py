"""Controlled RIR experiment smoke and anti-leakage checks.

Inputs are explicitly toy signals. Tests verify data integrity, not accuracy.
"""
import csv
from collections import defaultdict

import numpy as np
import pytest
import soundfile as sf

from app.rir_experiment import build_dataset, validate_source_manifests
from scripts.generate_rir_demo_inputs import create_demo_inputs


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "inputs"
    create_demo_inputs(root)
    return root


def test_demo_has_disjoint_speakers_and_rooms(inputs):
    speech, responses = validate_source_manifests(
        inputs, inputs / "speech.csv", inputs / "responses.csv")
    speakers = [set(x.speaker_id for x in speech[s]) for s in ("train", "val", "test")]
    rooms = [set(responses[s]) for s in ("train", "val", "test")]
    assert all(speakers[i].isdisjoint(speakers[j])
               and rooms[i].isdisjoint(rooms[j])
               for i, j in ((0, 1), (0, 2), (1, 2)))
    assert all(len(s) == 2 for s in rooms)


def test_dataset_pairs_and_audio_reconstruction(inputs, tmp_path):
    output = tmp_path / "dataset"
    report = build_dataset(inputs, inputs / "speech.csv", inputs / "responses.csv",
                           output, seconds=.5, snr_db=8.0)
    assert report["samples"] == 12
    assert report["per_split"] == {"train": 4, "val": 4, "test": 4}
    with (output / "manifest.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair_id"]].append(row)
        same_room = row["speech_room_id"] == row["ambience_room_id"]
        assert same_room == (row["label"] == "consistent")
        if same_room:
            assert row["speech_rir"] != row["ambience_rir"]
        mix, sr = sf.read(output / row["mixture"])
        speech, _ = sf.read(output / row["speech_stem"])
        noise, _ = sf.read(output / row["ambience_stem"])
        assert sr == 16000
        assert np.max(np.abs(mix - speech - noise)) <= 2 / 32768
        assert np.isfinite(mix).all()
        assert mix.size > 8000
        assert np.max(np.abs(mix)) <= .96
    assert len(by_pair) == 6
    for items in by_pair.values():
        assert {item["label"] for item in items} == {"consistent", "mismatched"}
        assert len({item["sha256_mix"] for item in items}) == 2
        assert len({item["dry_speech"] for item in items}) == 1
        assert len({item["dry_ambience"] for item in items}) == 1
        # Exactly identical speech stem across the balanced pair.
        a, _ = sf.read(output / items[0]["speech_stem"])
        b, _ = sf.read(output / items[1]["speech_stem"])
        np.testing.assert_array_equal(a, b)


def test_existing_output_is_not_overwritten(inputs, tmp_path):
    output = tmp_path / "dataset"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("untouched", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_dataset(inputs, inputs / "speech.csv", inputs / "responses.csv", output)
    assert sentinel.read_text(encoding="utf-8") == "untouched"


def _modify_csv(path, mutate):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    mutate(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_rejects_speaker_leakage(inputs):
    def change(rows):
        rows[-1]["speaker_id"] = rows[0]["speaker_id"]
    _modify_csv(inputs / "speech.csv", change)
    with pytest.raises(ValueError, match="Speaker leakage"):
        validate_source_manifests(inputs, inputs / "speech.csv", inputs / "responses.csv")


def test_rejects_room_leakage(inputs):
    def change(rows):
        rows[-1]["room_id"] = rows[0]["room_id"]
    _modify_csv(inputs / "responses.csv", change)
    with pytest.raises(ValueError, match="Room leakage"):
        validate_source_manifests(inputs, inputs / "speech.csv", inputs / "responses.csv")


def test_rejects_exact_duplicate_speech(inputs):
    def change(rows):
        rows[-1]["path"] = rows[0]["path"]
    _modify_csv(inputs / "speech.csv", change)
    with pytest.raises(ValueError, match="Duplicate asset"):
        validate_source_manifests(inputs, inputs / "speech.csv", inputs / "responses.csv")


def test_rejects_unsafe_path(inputs):
    def change(rows):
        rows[0]["path"] = "../../outside.wav"
    _modify_csv(inputs / "speech.csv", change)
    with pytest.raises(ValueError, match="Missing/unsafe"):
        validate_source_manifests(inputs, inputs / "speech.csv", inputs / "responses.csv")


def test_each_room_requires_two_independent_responses(inputs):
    def change(rows):
        del rows[1]
    _modify_csv(inputs / "responses.csv", change)
    with pytest.raises(ValueError, match="needs two distinct RIRs"):
        validate_source_manifests(inputs, inputs / "speech.csv", inputs / "responses.csv")


def test_rejects_settings_before_creating_files(inputs, tmp_path):
    output = tmp_path / "dataset"
    with pytest.raises(ValueError, match="generation settings"):
        build_dataset(inputs, inputs / "speech.csv", inputs / "responses.csv",
                      output, seconds=0)
    assert not output.exists()
