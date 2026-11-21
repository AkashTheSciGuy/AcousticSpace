# AcousticSpace Docker demo

The local demo uses two containers: FastAPI on port 8000 and an Nginx-served React build on port 5173. Only loopback (127.0.0.1) ports are exposed. This is **not** a public-production security configuration.

## Requirements

- Docker Engine / Docker Desktop with the Compose v2 plugin.
- A prepared local models directory: `backend/models/`.
- Optionally, your self-trained `baseline_model.joblib`, `baseline_model.json`,
  `acoustic_candidate.joblib` and `acoustic_comparison.json`.
- Do not download untrusted joblib models: deserialization can execute code.

## Start (Windows CMD from repository root)

```cmd
cd /d F:\Projects\AcousticSpace
if not exist backend\models mkdir backend\models
docker compose up --build -d
docker compose ps
```

Open http://127.0.0.1:5173 and http://127.0.0.1:8000/docs.
Backend health: http://127.0.0.1:8000/health.

The `backend/models` folder is mounted read-only at `/app/models`. Model files are excluded from images and GitHub. When models are missing, the rest of the app works and the prediction status correctly shows unavailable.

View logs with `docker compose logs --tail 100 backend` and
`docker compose logs --tail 100 frontend`.

Stop the stack with `docker compose down` (does not delete your local model files).
If a development server already occupies ports 5173 or 8000, stop it before starting the containers.

## Research scope

- Audio models produce uncalibrated experimental scores.
- Window scores are not validated tamper localization.
- The RIR analyzer takes a **separate** isolated impulse response.
- The Docker build does not supply AST weights, a trained RIR-based detector or evidence of forensic accuracy.
