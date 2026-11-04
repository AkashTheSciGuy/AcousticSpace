"""Validate incoming audio uploads before handing them to the ML pipeline.

This endpoint does NOT classify audio or persist uploaded recordings.
"""

from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.preprocessing import MAX_DURATION_SECONDS, TARGET_SAMPLE_RATE

router = APIRouter(prefix="/api/audio", tags=["audio"])

MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MiB, aligned with the current frontend.
CHUNK_SIZE = 1024 * 1024
ALLOWED_EXTENSIONS = {"wav", "mp3"}


@router.get("/capabilities")
def get_audio_capabilities() -> dict[str, bool | int | float | list[str]]:
    """Describe supported audio operations and their current limits."""
    return {
        "supported_formats": sorted(ALLOWED_EXTENSIONS),
        "max_upload_bytes": MAX_FILE_BYTES,
        "max_inspection_duration_seconds": MAX_DURATION_SECONDS,
        "processing_sample_rate_hz": TARGET_SAMPLE_RATE,
        "upload_validation_available": True,
        "feature_inspection_available": True,
        "deepfake_classification_available": False,
    }


def _looks_like_audio(header: bytes, extension: str) -> bool:
    if extension == "wav":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"

    # MP3 commonly starts with an ID3 tag or an MPEG audio frame sync word.
    # These simple signatures are a preliminary check, not full decoding.
    if header[:3] == b"ID3":
        return True
    return len(header) >= 4 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0 and (header[1] & 0x18) != 0x08 and (header[1] & 0x06) != 0


@router.post("/validate")
async def validate_audio_upload(
    file: Annotated[UploadFile, File(description="WAV or MP3, maximum 20 MiB")],
) -> dict[str, str | int]:
    """Validate extension, signature, and size without saving the upload."""
    raw_name = file.filename or ""
    filename = raw_name.replace("\\", "/").rsplit("/", 1)[-1]
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Only WAV and MP3 files are supported.")

        header = await file.read(12)
        if not header:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if not _looks_like_audio(header, extension):
            raise HTTPException(status_code=400, detail="File content does not match its audio format.")

        total_size = len(header)
        while True:
            chunk = await file.read(CHUNK_SIZE)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail="Audio file exceeds the 20 MiB limit.")

        return {
            "status": "valid",
            "filename": filename,
            "format": extension,
            "size_bytes": total_size,
            "message": "Audio upload validated. Deepfake analysis is not available yet.",
        }
    finally:
        await file.close()
