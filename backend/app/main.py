"""FastAPI entrypoint for the AcousticSpace backend."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.audio_validation import router as audio_router
from app.audio_inspection import router as inspection_router
from app.rir_api import router as rir_router
from app.prediction_api import router as prediction_router
from app.comparison_api import router as comparison_router

app = FastAPI(
    title='AcousticSpace API',
    description='Experimental speech-acoustic classification and separate isolated RIR measurements; no RIR-based speech verdicts',
    version='0.3.0',
)

app.include_router(audio_router)
app.include_router(inspection_router)
app.include_router(rir_router)
app.include_router(prediction_router)
app.include_router(comparison_router)


@app.get('/')
def root():
    """Return a welcome message for the API."""
    if os.getenv('ACOUSTICSPACE_FRONTEND_DIR'):
        from fastapi.responses import FileResponse
        return FileResponse(Path(os.environ['ACOUSTICSPACE_FRONTEND_DIR']) / 'index.html')
    return {'message': 'Welcome to AcousticSpace API', 'status': 'running'}


@app.get('/health')
def health_check() -> dict[str, str]:
    """Return a simple service health response."""
    return {'status': 'ok', 'service': 'AcousticSpace Backend'}


# Optional single-origin production UI. API routes above take precedence.
frontend_dir = os.getenv('ACOUSTICSPACE_FRONTEND_DIR')
if frontend_dir:
    app.mount('/', StaticFiles(directory=Path(frontend_dir), html=True), name='frontend')
