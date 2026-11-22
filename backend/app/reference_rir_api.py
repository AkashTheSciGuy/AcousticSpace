"""Reference-aided room-consistency HTTP endpoint for controlled recordings.

Unlike blind single-file deepfake analysis, this endpoint explicitly requires
dry source recordings and their *separately isolated* reverberant stems.
It exposes acoustic feature distances only, not a real/fake or mismatch verdict.
"""
from __future__ import annotations

import asyncio
import io
from typing import Annotated

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.rir_features import SAMPLE_RATE, reference_consistency_features

router = APIRouter(prefix="/api/rir", tags=["reference-aided-rir"])

# Bound each input independently; no files are persisted on disk.
MAX_REFERENCE_WAV_BYTES = 4 * 1024 * 1024
MAX_REFERENCE_SECONDS = 12
MIN_REFERENCE_FRAMES = 256


async def _decode_reference(file: UploadFile, role: str) -> np.ndarray:
    """Accept only bounded 16-kHz mono WAVs and close the upload on all paths."""
    try:
        name = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not name.lower().endswith(".wav"):
            raise HTTPException(415, detail=f"{role}: a WAV recording is required.")
        data = await file.read(MAX_REFERENCE_WAV_BYTES + 1)
        if len(data) > MAX_REFERENCE_WAV_BYTES:
            raise HTTPException(413, detail=f"{role}: WAV exceeds 4 MiB.")
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
            raise HTTPException(400, detail=f"{role}: invalid WAV header.")
        try:
            with sf.SoundFile(io.BytesIO(data)) as source:
                if source.format != "WAV" or source.samplerate != SAMPLE_RATE:
                    raise ValueError("requires a 16-kHz WAV recording")
                if source.channels != 1:
                    raise ValueError("requires a mono WAV recording")
                if not MIN_REFERENCE_FRAMES <= source.frames <= MAX_REFERENCE_SECONDS * SAMPLE_RATE:
                    raise ValueError("requires 256 samples to 12 seconds")
                samples = source.read(dtype="float64", always_2d=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(422, detail=f"{role}: {exc}") from exc
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise HTTPException(422, detail=f"{role}: non-finite or multichannel audio.")
        if float(np.sqrt(np.mean(np.square(samples)))) < 1e-7:
            raise HTTPException(422, detail=f"{role}: silent/near-silent recording.")
        return samples
    finally:
        await file.close()


@router.post("/reference-consistency")
async def reference_consistency(
    dry_speech: Annotated[UploadFile, File(description="Known dry voice signal, mono 16-kHz WAV")],
    wet_speech: Annotated[UploadFile, File(description="Isolated reverberant voice stem, mono 16-kHz WAV")],
    dry_ambience: Annotated[UploadFile, File(description="Known dry ambience signal, mono 16-kHz WAV")],
    wet_ambience: Annotated[UploadFile, File(description="Isolated reverberant ambience stem, mono 16-kHz WAV")],
) -> dict:
    """Estimate two *reference-aided* transfer responses and compare descriptors.

    No blind separation is performed, and the feature differences are not
    independently validated as detection probabilities.
    """
    # Upload decoding is bounded; run signal processing off the request event loop.
    decoded = await asyncio.gather(
        _decode_reference(dry_speech, "dry_speech"),
        _decode_reference(wet_speech, "wet_speech"),
        _decode_reference(dry_ambience, "dry_ambience"),
        _decode_reference(wet_ambience, "wet_ambience"),
    )
    try:
        result = await asyncio.to_thread(reference_consistency_features, *decoded)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        raise HTTPException(422, detail=f"Reference-aided transfer estimation failed: {exc}") from exc
    return {
        "status": "analyzed",
        "input_type": "four_reference_aided_audio_signals",
        "features": result.to_dict(),
        "feature_names": list(result.to_dict()),
        "sample_rate_hz": SAMPLE_RATE,
        "prediction": None,
        "limitations": [
            "Requires known dry speech, isolated reverberant speech stem, known dry ambience and isolated reverberant ambience stem.",
            "Does not infer those four signals from a single mixed recording.",
            "Feature distances are measurements, not calibrated mismatch probabilities or real/fake predictions.",
            "The existing controlled classifier failed to generalize on the reported held-out experiment.",
        ],
    }
