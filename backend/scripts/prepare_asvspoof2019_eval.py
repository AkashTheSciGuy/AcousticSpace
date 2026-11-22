r"""Make one deterministic, bounded ASVspoof 2019 LA official-eval subset.

The official evaluation partition MUST be available locally, including the
official protocol with labels. This script NEVER trains a model, evaluates
scores, invents labels, or modifies source audio.

Windows CMD, from backend:
  .venv\Scripts\python.exe scripts\prepare_asvspoof2019_eval.py ^
    "F:\Datasets\ASVspoof2019\LA" ^
    --per-class 100 --seed 42 ^
    --output "F:\Datasets\ASVspoof2019\eval_fixed_100_each_seed42.csv"

Then run scripts/evaluate_ast_external.py with a frozen checkpoint.
The resulting balanced, attack-stratified SUBSET is not the full official
ASVspoof 2019 benchmark and must not be described as cross-dataset testing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

OUTPUT_COLUMNS = ("path", "label", "sha256", "group", "source_reference")
PROTOCOL_NAME = "ASVspoof2019.LA.cm.eval.trl.txt"
AUDIO_DIRECTORY = "ASVspoof2019_LA_eval/flac"
MAX_PROTOCOL_BYTES = 16 * 1024 * 1024
MAX_AUDIO_BYTES = 20 * 1024 * 1024
MAX_PER_CLASS = 500


def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _protocol_file(root: Path) -> Path:
    candidates = (
        root / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_NAME,
        root / "ASVspoof2019_LA_protocols" / PROTOCOL_NAME,
    )
    for candidate in candidates:
        if candidate.is_file():
            if not 0 < candidate.stat().st_size <= MAX_PROTOCOL_BYTES:
                raise ValueError("Official evaluation protocol is empty or exceeds 16 MiB.")
            return candidate
    raise ValueError(f"Missing official evaluation protocol: {PROTOCOL_NAME}")


def _read_eval_protocol(root: Path) -> tuple[list[dict], Path]:
    protocol = _protocol_file(root)
    records: list[dict] = []
    seen_ids: set[str] = set()
    with protocol.open(encoding="utf-8-sig") as handle:
        for line, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            fields = raw.split()
            if len(fields) != 5:
                raise ValueError(f"Protocol line {line}: expected five official fields.")
            speaker, utterance, unused, attack, label = fields
            if not speaker.startswith("LA_") or not utterance.startswith("LA_E_"):
                raise ValueError(f"Protocol line {line}: unexpected evaluation speaker/utterance ID.")
            if unused != "-" or label not in ("bonafide", "spoof"):
                raise ValueError(f"Protocol line {line}: unsupported official label or field.")
            if (label == "bonafide" and attack != "-") or (label == "spoof" and attack == "-"):
                raise ValueError(f"Protocol line {line}: inconsistent attack marker and label.")
            if utterance in seen_ids:
                raise ValueError(f"Protocol line {line}: duplicate evaluation utterance ID.")
            seen_ids.add(utterance)
            records.append({
                "speaker": speaker,
                "utterance": utterance,
                "attack": attack,
                "label": "real" if label == "bonafide" else "fake",
                "line": line,
            })
    if not records:
        raise ValueError("Official evaluation protocol contains no recordings.")
    return records, protocol


def _stratified_sample(records: list[dict], count: int, *, seed: int) -> list[dict]:
    """Round-robin over real speakers / fake attack conditions, never predictions."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        key = record["speaker"] if record["label"] == "real" else record["attack"]
        buckets[key].append(record)
    if not buckets or sum(len(rows) for rows in buckets.values()) < count:
        raise ValueError(f"Fewer than {count} official records available for a class.")
    ordered_keys = sorted(buckets)
    random.Random(f"{seed}:keys").shuffle(ordered_keys)
    for key in ordered_keys:
        buckets[key].sort(key=lambda record: record["utterance"])
        random.Random(f"{seed}:{key}").shuffle(buckets[key])
    selected: list[dict] = []
    while len(selected) < count:
        made_progress = False
        for key in ordered_keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop())
                made_progress = True
        if not made_progress:
            raise ValueError("Unable to fulfill the predeclared sample count.")
    return selected


def create_manifest(root: Path, output: Path, *, per_class: int = 100, seed: int = 42) -> dict:
    """Write a new stable, label-stratified 2019 eval subset; never overwrite."""
    if type(per_class) is not int or not 1 <= per_class <= MAX_PER_CLASS:
        raise ValueError("per_class must be an integer between 1 and 500.")
    if type(seed) is not int or not 0 <= seed < 2 ** 63:
        raise ValueError("seed must be a nonnegative bounded integer.")
    root = Path(root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("ASVspoof 2019 LA dataset root is missing.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output}")

    official, protocol = _read_eval_protocol(root)
    real = [row for row in official if row["label"] == "real"]
    fake = [row for row in official if row["label"] == "fake"]
    chosen = _stratified_sample(real, per_class, seed=seed) + _stratified_sample(
        fake, per_class, seed=seed + 1
    )
    if len({row["speaker"] for row in chosen}) < 2:
        raise ValueError("Selected evaluation subset requires at least two distinct speakers.")
    selected_rows: list[dict] = []
    hashes: set[str] = set()
    attack_counts: Counter[str] = Counter()
    for item in chosen:
        path = root / AUDIO_DIRECTORY / (item["utterance"] + ".flac")
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_AUDIO_BYTES:
            raise ValueError(f"Selected official FLAC missing or >20 MiB: {item['utterance']}")
        digest = _sha256(path)
        if digest in hashes:
            raise ValueError("Duplicate audio bytes among selected official evaluation recordings.")
        hashes.add(digest)
        attack_counts[item["attack"]] += 1
        selected_rows.append({
            "path": path.relative_to(root).as_posix(),
            "label": item["label"],
            "sha256": digest,
            "group": item["speaker"],
            "source_reference": f"{protocol.name}:line={item['line']}:attack={item['attack']}",
        })
    # Sort by file name so even output byte ordering is deterministic.
    selected_rows.sort(key=lambda row: row["path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(selected_rows)
    return {
        "status": "prepared",
        "partition": "ASVspoof2019_LA_official_eval_subset",
        "subset_selection": "predeclared_equal_class_counts; round_robin_real_speaker_and_fake_attack",
        "official_protocol": str(protocol),
        "protocol_sha256": _sha256(protocol),
        "manifest_path": str(output),
        "manifest_sha256": _sha256(output),
        "seed": seed,
        "per_class": per_class,
        "counts": {"real": per_class, "fake": per_class},
        "speaker_groups": len({row["speaker"] for row in chosen}),
        "attack_counts_in_subset": dict(sorted(attack_counts.items())),
        "warning": (
            "Fixed balanced and attack-stratified subset of official 2019 evaluation "
            "audio; this is NOT official full-corpus performance and is NOT "
            "cross-dataset validation. Do not change seed/sample composition "
            "after seeing checkpoint predictions. Speaker/source metadata is "
            "provided by the official protocol but not independently verified."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("la_root", type=Path)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = create_manifest(args.la_root, args.output, per_class=args.per_class, seed=args.seed)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
