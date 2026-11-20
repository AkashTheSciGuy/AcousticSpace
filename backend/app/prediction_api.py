"""Experimental model readiness and audio inference endpoints."""
from time import perf_counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.audio_validation import CHUNK_SIZE, MAX_FILE_BYTES, _looks_like_audio
from app.model_inference import ModelUnavailableError, model_status, predict_audio
from app.preprocessing import load_audio

router = APIRouter(prefix='/api/model', tags=['experimental-model'])
ModelVariant = Literal['baseline', 'acoustic']


@router.get('/status')
def get_model_status(variant: ModelVariant = 'baseline') -> dict:
    return model_status(variant)


@router.post('/predict')
async def predict_uploaded_audio(
    file: Annotated[UploadFile, File(description='WAV/MP3, at most 20 MiB and 60 s')],
    variant: ModelVariant = 'baseline',
) -> dict:
    """Use baseline by default; acoustic inference requires a separately trained artifact."""
    started = perf_counter()
    temp_path: Path | None = None
    try:
        state = await run_in_threadpool(model_status, variant)
        if not state['available']:
            raise HTTPException(status_code=503, detail=state['message'])
        filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
        ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
        if ext not in {'wav', 'mp3'}:
            raise HTTPException(status_code=415, detail='Only WAV and MP3 files are supported.')
        header = await file.read(12)
        if not _looks_like_audio(header, ext):
            raise HTTPException(status_code=400, detail='File content does not match its audio format.')
        size = len(header)
        with NamedTemporaryFile(prefix='acousticspace_predict_', suffix=f'.{ext}', delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(header)
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_BYTES:
                    raise HTTPException(status_code=413, detail='Audio exceeds the 20 MiB limit.')
                stream.write(chunk)
        try:
            audio = await run_in_threadpool(load_audio, temp_path)
            result = await run_in_threadpool(predict_audio, audio, variant)
        except ModelUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=f'Audio could not be processed: {exc}') from exc
        return {'filename': filename, **result,
                'processing_time_ms': round((perf_counter() - started) * 1000, 3)}
    finally:
        await file.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
