"""Bounded speech diagnostic endpoint: no authenticity verdict."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio_validation import CHUNK_SIZE, MAX_FILE_BYTES, _looks_like_audio
from app.preprocessing import load_audio
from app.speech_diagnostics import analyze_speech_diagnostics

router = APIRouter(prefix="/api/audio", tags=["audio"])


@router.post("/diagnostics")
async def diagnostics(
    file: Annotated[UploadFile, File(description="WAV/MP3, 20 MiB, 60 seconds")],
) -> dict:
    """Small spectrogram and non-validated cadence/breath candidates."""
    path: Path | None = None
    try:
        filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in {"wav", "mp3"}:
            raise HTTPException(415, "Only WAV and MP3 files are supported.")
        header = await file.read(12)
        if not _looks_like_audio(header, ext):
            raise HTTPException(400, "Audio signature does not match file format.")
        total = len(header)
        with NamedTemporaryFile(prefix="acousticspace_diag_", suffix=f".{ext}",
                                delete=False) as temp:
            path = Path(temp.name)
            temp.write(header)
            while True:
                block = await file.read(CHUNK_SIZE)
                if not block:
                    break
                total += len(block)
                if total > MAX_FILE_BYTES:
                    raise HTTPException(413, "Audio exceeds 20 MiB.")
                temp.write(block)
        try:
            audio = load_audio(path)
            return {"status": "analyzed", "filename": filename,
                    **analyze_speech_diagnostics(audio)}
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(422, f"Audio diagnostics failed: {exc}") from exc
    finally:
        await file.close()
        if path is not None:
            path.unlink(missing_ok=True)
