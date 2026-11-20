# API contract and deployment

The frontend uses same-origin `/api` calls. Vite proxies them to `127.0.0.1:8000` during development. The Docker image serves the built UI and API together on port 8000; `vite preview` alone is not a production API deployment.

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/` | API welcome JSON, or built HTML when `ACOUSTICSPACE_FRONTEND_DIR` is configured |
| GET | `/health` | Service liveness; does not claim model readiness |
| GET | `/docs`, `/redoc`, `/openapi.json` | Interactive/generated API documentation |
| POST | `/api/audio/validate` | Multipart `file`; WAV/MP3 extension, signature and 20 MiB size validation; does not decode/classify |
| POST | `/api/audio/inspect` | Multipart `file`; decoded signal features and exploratory descriptors, no authenticity verdict |
| POST | `/api/rir/analyze` | Multipart `file`; isolated RIR WAV only; nullable room-decay metrics and limitations |
| GET | `/api/model/status?variant=baseline` | `available`, variant/type/message, artifact hash when available, optional validated baseline evaluation |
| POST | `/api/model/predict?variant=baseline` | Multipart `file`; global real/fake label, two class scores, `model_version`, `processing_time_ms`, disclaimer |
| GET | `/api/model/comparison` | Validated local comparison or 404 if absent/invalid |

`variant` accepts `baseline` or `acoustic`; default is baseline. Prediction input is WAV/MP3, maximum 20 MiB and 60 s. Inspection/RIR decode to 16 kHz mono. Extension/signature validation is not a substitute for decoding.

Typical errors use `{"detail":"message"}`: 400 empty/signature failure; 413 too large; 415 unsupported extension; 422 undecodable/too long/invalid query or missing field; 503 selected model unavailable/incompatible. FastAPI validation errors can use an array in `detail`. Prediction checks readiness before file validation, so absent models return 503 first. Report 404 does not mean the API itself is offline.

`model_version` is `sha256:` plus the complete trusted joblib artifact digest. It identifies bytes, not training provenance, calibration or accuracy. Do not hot-swap model files during requests; restart after installing artifacts. `processing_time_ms` measures server request handling through prediction; it is not browser elapsed time.

## Local artifacts

By default install trusted locally generated artifacts in `backend/models/`:

- `baseline_model.joblib` and its optional `baseline_model.json` evaluation.
- `acoustic_candidate.joblib`.
- `acoustic_comparison.json`.

Operator environment overrides: `ACOUSTICSPACE_BASELINE_MODEL`, `ACOUSTICSPACE_ACOUSTIC_MODEL`, `ACOUSTICSPACE_COMPARISON_REPORT`. These are server settings, never user-uploaded paths. Only load artifacts you generated and trust: joblib/pickle can execute code. Use the original compatible training environment or retrain; do not treat a successfully unpickled artifact as validated research.

## Docker (from repository root)

```sh
docker compose up --build
```

Open `http://localhost:8000`. Compose binds localhost only and mounts `backend/models` read-only. The image contains no trained model or recordings; missing models are reported honestly as unavailable. Docker build context excludes datasets, models, recordings and CSV manifests. Do not expose this prototype publicly without separately designing upload quotas, authentication, TLS, rate limits and operational controls.

For non-Docker production-style local verification, build the frontend then set `ACOUSTICSPACE_FRONTEND_DIR` to its **absolute** `dist` path before launching Uvicorn. The UI and API then share port 8000.
