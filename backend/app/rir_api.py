"""Dedicated RIR upload endpoint. Ordinary speech is NOT a valid RIR input."""

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio_validation import CHUNK_SIZE, MAX_FILE_BYTES, _looks_like_audio
from app.preprocessing import load_audio
from app.rir_analysis import analyze_rir

router = APIRouter(prefix="/api/rir", tags=["rir"])


@router.post("/analyze")
async def analyze_uploaded_rir(
    file: Annotated[UploadFile, File(description="Measured or simulated room impulse response WAV (max 20 MiB)")],
) -> dict:
    """Return room-decay descriptors; never claim authenticity classification."""
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    path: Path | None = None
    try:
        if not filename.lower().endswith(".wav"):
            raise HTTPException(415, "Room impulse response must be a WAV file.")
        header = await file.read(12)
        if not _looks_like_audio(header, "wav"):
            raise HTTPException(400, "File content does not match WAV audio.")
        size = len(header)
        with NamedTemporaryFile(prefix="acousticspace_rir_", suffix=".wav", delete=False) as temp:
            path = Path(temp.name)
            temp.write(header)
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise HTTPException(413, "RIR file exceeds the 20 MiB limit.")
                temp.write(chunk)
        try:
            audio = load_audio(path)
            metrics = analyze_rir(audio.samples, audio.sample_rate)
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(422, f"RIR could not be analyzed: {exc}") from exc
        return {
            "status": "analyzed",
            "filename": filename,
            "size_bytes": size,
            "input_type": "room_impulse_response",
            "message": "RIR descriptors only; this endpoint does not determine whether speech is genuine or fake.",
            "metrics": metrics.to_dict(),
            "limitations": [
                "Upload an isolated measured or simulated impulse response, not ordinary speech.",
                "T20-derived RT60 requires a clean decay and is not always estimable.",
                "DRR uses a 2.5 ms direct-sound window and is an approximation.",
            ],
        }
    finally:
        await file.close()
        if path is not None:
            path.unlink(missing_ok=True)
