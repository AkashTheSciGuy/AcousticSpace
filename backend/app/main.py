"""FastAPI entrypoint for the AcousticSpace backend."""

from fastapi import FastAPI

from app.audio_validation import router as audio_router
from app.audio_inspection import router as inspection_router
from app.rir_api import router as rir_router
from app.prediction_api import router as prediction_router

app = FastAPI(
    title='AcousticSpace API',
    description='Backend API for AcousticSpace audio deepfake detection',
    version='0.3.0',
)

app.include_router(audio_router)
app.include_router(inspection_router)
app.include_router(rir_router)
app.include_router(prediction_router)


@app.get('/')
def root() -> dict[str, str]:
    """Return a welcome message for the API."""
    return {'message': 'Welcome to AcousticSpace API', 'status': 'running'}


@app.get('/health')
def health_check() -> dict[str, str]:
    """Return a simple service health response."""
    return {'status': 'ok', 'service': 'AcousticSpace Backend'}
