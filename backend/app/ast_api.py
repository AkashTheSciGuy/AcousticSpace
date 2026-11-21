"""Optional AST prediction API; trained weights never come from HTTP requests."""
import asyncio
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.ast_inference import ASTUnavailableError, ast_status, predict_ast
from app.audio_validation import CHUNK_SIZE, MAX_FILE_BYTES, _looks_like_audio
from app.preprocessing import load_audio

router = APIRouter(prefix="/api/ast", tags=["optional-ast"])


@router.get("/status")
def status() -> dict:
    return ast_status()


@router.post("/predict")
async def predict(file: Annotated[UploadFile, File(description="WAV/MP3, 20 MiB max")]) -> dict:
    temp_path: Path | None = None
    try:
        readiness = await asyncio.to_thread(ast_status)
        if not readiness["available"]:
            raise HTTPException(503, detail=readiness["message"])
        filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in {"wav", "mp3"}:
            raise HTTPException(415, detail="Only WAV and MP3 files are supported.")
        header = await file.read(12)
        if not _looks_like_audio(header, ext):
            raise HTTPException(400, detail="Invalid audio signature.")
        size = len(header)
        with NamedTemporaryFile(prefix="acousticspace_ast_", suffix=f".{ext}", delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(header)
            while True:
                block = await file.read(CHUNK_SIZE)
                if not block:
                    break
                size += len(block)
                if size > MAX_FILE_BYTES:
                    raise HTTPException(413, detail="Audio exceeds 20 MiB.")
                temp.write(block)
        try:
            audio = await asyncio.to_thread(load_audio, temp_path)
            result = await asyncio.to_thread(predict_ast, audio)
        except ASTUnavailableError as exc:
            raise HTTPException(503, detail=str(exc)) from exc
        except (OSError, ValueError, RuntimeError) as exc:
            raise HTTPException(422, detail=f"AST prediction failed: {exc}") from exc
        return {"filename": filename, **result}
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
