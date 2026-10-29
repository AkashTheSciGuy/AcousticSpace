"""FastAPI entrypoint for the AcousticSpace backend."""

from fastapi import FastAPI

app = FastAPI(
    title="AcousticSpace API",
    description="Backend API for AcousticSpace audio deepfake detection",
    version="0.1.0",
)


@app.get("/")
def root() -> dict[str, str]:
    """Return a welcome message for the API."""
    return {"message": "Welcome to AcousticSpace API", "status": "running"}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a simple service health response."""
    return {"status": "ok", "service": "AcousticSpace Backend"}
