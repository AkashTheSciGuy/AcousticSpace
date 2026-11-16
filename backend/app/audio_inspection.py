"""Bounded audio inspection endpoint; returns measurements, not AI predictions."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio_validation import CHUNK_SIZE, MAX_FILE_BYTES, _looks_like_audio
from app.features import extract_features
from app.preprocessing import load_audio
from app.speech_acoustics import extract_speech_acoustics

router = APIRouter(prefix="/api/audio", tags=["audio"])


@router.post("/inspect")
async def inspect_audio(
    file: Annotated[UploadFile, File(description="WAV or MP3, at most 20 MiB")],
) -> dict:
    """Inspect an upload with basic features and exploratory speech descriptors."""
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    temporary_path: Path | None = None
    try:
        if extension not in {"wav", "mp3"}:
            raise HTTPException(415, "Only WAV and MP3 files are supported.")

        header = await file.read(12)
        if not header:
            raise HTTPException(400, "The uploaded file is empty.")
        if not _looks_like_audio(header, extension):
            raise HTTPException(400, "File content does not match its audio format.")

        total = len(header)
        with NamedTemporaryFile(prefix="acousticspace_", suffix=f".{extension}", delete=False) as temp:
            temporary_path = Path(temp.name)
            temp.write(header)
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_FILE_BYTES:
                    raise HTTPException(413, "Audio file exceeds the 20 MiB limit.")
                temp.write(chunk)

        try:
            audio = load_audio(temporary_path)
            features = extract_features(audio)
            speech_acoustics = extract_speech_acoustics(audio)
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(422, f"Audio could not be inspected: {exc}") from exc

        return {
            "status": "valid",
            "filename": filename,
            "format": extension,
            "size_bytes": total,
            "message": "Audio inspected successfully. No deepfake classification was performed.",
            "features": {
                "duration_seconds": round(features.duration_seconds, 4),
                "sample_rate": features.sample_rate,
                "rms_amplitude": round(features.rms_amplitude, 6),
                "zero_crossing_rate": round(features.zero_crossing_rate, 6),
                "spectral_centroid_hz": round(features.spectral_centroid_hz, 2),
            },
            "speech_acoustics": {
                "mean_spectral_flatness": round(speech_acoustics.mean_spectral_flatness, 6),
                "rms_coefficient_of_variation": round(speech_acoustics.rms_coefficient_of_variation, 6),
                "mean_positive_spectral_flux": round(speech_acoustics.mean_positive_spectral_flux, 6),
                "disclaimer": "Exploratory speech descriptors only; not a measured room impulse response or deepfake verdict.",
            },
        }
    finally:
        await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
