r"""Run a repeatable, local API acceptance test without launching a web server.

From AcousticSpace/backend in Windows CMD:
    .venv\Scripts\python.exe scripts\final_acceptance.py
    .venv\Scripts\python.exe scripts\final_acceptance.py --require-models
    .venv\Scripts\python.exe scripts\final_acceptance.py --check-ast

This uses generated non-speech WAV fixtures and FastAPI's in-process HTTP
client. It does NOT measure model accuracy or verify forensic claims.
Training datasets and checkpoints are never overwritten.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import wave
from pathlib import Path
from typing import Callable

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402


def wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode bounded mono PCM WAVs entirely in memory for an HTTP smoke test."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not 256 <= values.size <= 16000 * 3:
        raise ValueError("Fixture requires 256 to 48000 mono samples")
    if not np.isfinite(values).all() or np.max(np.abs(values)) > 1:
        raise ValueError("Fixture samples must be finite and normalized")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(np.rint(values * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def _speech_fixture() -> bytes:
    n = 16000
    times = np.arange(n, dtype=np.float64) / 16000
    # A harmonic test tone, not authentic speech or a labeled spoof sample.
    signal = 0.15 * np.sin(2 * np.pi * 220 * times)
    signal += 0.09 * np.sin(2 * np.pi * 440 * times)
    return wav_bytes(signal)


def _rir_fixture() -> bytes:
    times = np.arange(32000, dtype=np.float64) / 16000
    generator = np.random.default_rng(103)
    values = generator.normal(size=times.size)
    values *= np.exp(-np.log(1000.0) * times / 0.5)
    values[0] = 2.0
    values /= 3.0
    return wav_bytes(values)


def _expect(response, status: int = 200) -> dict:
    if response.status_code != status:
        raise ValueError(
            f"Expected HTTP {status}, got {response.status_code}: "
            f"{response.text[:180]}"
        )
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Response was not a JSON object")
    return payload


def check_endpoints(*, require_models: bool = False, check_ast: bool = False) -> dict:
    """Check actual HTTP routes; report optional model absence honestly."""
    checks: dict[str, dict] = {}

    def verify(name: str, action: Callable[[], dict]) -> None:
        try:
            detail = action()
            checks[name] = {"passed": True, **detail}
        except (AssertionError, KeyError, TypeError, ValueError, RuntimeError, OSError) as exc:
            checks[name] = {"passed": False, "error": str(exc)[:300]}

    with TestClient(app) as client:
        audio = _speech_fixture()
        rir = _rir_fixture()

        def get_health() -> dict:
            result = _expect(client.get("/health"))
            assert result.get("status") == "ok"
            return {"status": "ok"}

        verify("health", get_health)

        def schema() -> dict:
            result = _expect(client.get("/openapi.json"))
            expected_paths = {
                "/health", "/api/audio/capabilities", "/api/audio/validate",
                "/api/audio/inspect", "/api/audio/diagnostics",
                "/api/rir/analyze", "/api/rir/reference-consistency", "/api/model/status",
                "/api/model/predict", "/api/model/segments",
                "/api/ast/status", "/api/ast/predict",
            }
            observed = set(result["paths"])
            missing = sorted(expected_paths - observed)
            if missing:
                raise ValueError("Missing API routes: " + ", ".join(missing))
            return {"required_routes": len(expected_paths)}

        verify("openapi_contract", schema)

        def capabilities() -> dict:
            result = _expect(client.get("/api/audio/capabilities"))
            assert result["max_upload_bytes"] == 20 * 1024 * 1024
            assert result["max_inspection_duration_seconds"] == 60
            assert set(result["supported_formats"]) == {"wav", "mp3"}
            return {"upload_limit_mib": 20, "max_duration_seconds": 60}

        verify("audio_capabilities", capabilities)

        def validation() -> dict:
            result = _expect(client.post(
                "/api/audio/validate",
                files={"file": ("smoke.wav", audio, "audio/wav")},
            ))
            assert result["status"] == "valid"
            assert result["size_bytes"] == len(audio)
            return {"validated_bytes": len(audio)}

        verify("audio_validation", validation)

        def inspection() -> dict:
            result = _expect(client.post(
                "/api/audio/inspect",
                files={"file": ("smoke.wav", audio, "audio/wav")},
            ))
            assert result["status"] == "valid"
            assert result["features"]["sample_rate"] == 16000
            assert abs(result["features"]["duration_seconds"] - 1.0) < 0.01
            assert result["features"]["rms_amplitude"] > 0
            return {"sample_rate": 16000, "duration_seconds": 1.0}

        verify("audio_inspection", inspection)

        def diagnostics() -> dict:
            result = _expect(client.post(
                "/api/audio/diagnostics",
                files={"file": ("smoke.wav", audio, "audio/wav")},
            ))
            assert result["status"] == "analyzed"
            assert result["mel_bins"] == 48
            assert len(result["log_mel_db"]) == 48
            assert 1 <= result["time_bins"] <= 100
            return {"mel_bins": result["mel_bins"], "time_bins": result["time_bins"]}

        verify("spectrogram_diagnostics", diagnostics)

        def rir_analysis() -> dict:
            result = _expect(client.post(
                "/api/rir/analyze",
                files={"file": ("synthetic_impulse.wav", rir, "audio/wav")},
            ))
            assert result["status"] == "analyzed"
            assert result["input_type"] == "room_impulse_response"
            assert result["metrics"]["sample_rate_hz"] == 16000
            assert result["metrics"]["rt60_t20_seconds"] is not None
            return {"rt60_t20_seconds": result["metrics"]["rt60_t20_seconds"]}

        verify("isolated_rir_analysis", rir_analysis)

        def reference_aided_rir() -> dict:
            from scipy.signal import fftconvolve

            rng = np.random.default_rng(321)
            speech = rng.normal(size=8000) * 0.05
            ambience = rng.normal(size=8000) * 0.05
            response = np.zeros(2000, dtype=np.float64)
            response[0] = 0.9
            response[500] = 0.1
            payload = {
                "dry_speech": wav_bytes(speech),
                "wet_speech": wav_bytes(fftconvolve(speech, response)),
                "dry_ambience": wav_bytes(ambience),
                "wet_ambience": wav_bytes(fftconvolve(ambience, response)),
            }
            result = _expect(client.post(
                "/api/rir/reference-consistency",
                files={name: (name + ".wav", data, "audio/wav")
                       for name, data in payload.items()},
            ))
            assert result["status"] == "analyzed"
            assert result["prediction"] is None
            assert len(result["features"]) == 7
            assert all(np.isfinite(list(result["features"].values())))
            return {
                "features": len(result["features"]),
                "reference_signals": 4,
                "note": "Known isolated stems, not blind separation or deepfake detection",
            }

        verify("reference_aided_rir_comparison", reference_aided_rir)

        def rejection() -> dict:
            response = client.post(
                "/api/audio/inspect",
                files={"file": ("broken.wav", b"not-a-wav", "audio/wav")},
            )
            _expect(response, 400)
            response = client.post(
                "/api/rir/analyze",
                files={"file": ("not-a-rir.mp3", b"ID3abc", "audio/mpeg")},
            )
            _expect(response, 415)
            return {"invalid_audio_http": 400, "unsupported_rir_http": 415}

        verify("invalid_upload_rejection", rejection)

        available_models: dict[str, bool] = {}
        for variant in ("baseline", "acoustic"):
            def model_check(variant: str = variant) -> dict:
                status = _expect(client.get(f"/api/model/status?variant={variant}"))
                assert isinstance(status["available"], bool)
                available_models[variant] = status["available"]
                if require_models and not status["available"]:
                    raise ValueError(f"{variant} model missing: {status['message']}")
                if not status["available"]:
                    return {"available": False, "note": status["message"]}
                result = _expect(client.post(
                    f"/api/model/predict?variant={variant}",
                    files={"file": ("smoke.wav", audio, "audio/wav")},
                ))
                assert result["status"] == "predicted"
                assert result["label"] in ("real", "fake")
                assert result["variant"] == variant
                assert 0 <= result["fake_probability"] <= 1
                assert "Not RIR-based" in result["disclaimer"]
                return {
                    "available": True,
                    "prediction_endpoint": "ok",
                    "note": "Fixture checks inference plumbing, not classification accuracy",
                }
            verify(f"{variant}_model", model_check)

        def comparison() -> dict:
            response = client.get("/api/model/comparison")
            if response.status_code == 404 and not require_models:
                return {"available": False, "note": "Optional local comparison report absent"}
            result = _expect(response)
            assert result["experiment"] == "speech_acoustics_feature_ablation_v1"
            assert "models" in result
            return {"available": True, "models": sorted(result["models"])}

        verify("feature_comparison", comparison)

        def segment_score() -> dict:
            if not available_models.get("baseline", False):
                if require_models:
                    raise ValueError("Baseline model needed to score windows")
                return {"available": False, "note": "Skipped (baseline model not installed)"}
            result = _expect(client.post(
                "/api/model/segments?variant=baseline",
                files={"file": ("smoke.wav", audio, "audio/wav")},
            ))
            assert len(result["segments"]) == 1
            assert result["segments"][0]["start_seconds"] == 0
            assert "NOT validated" in result["warning"]
            return {"available": True, "windows": len(result["segments"])}

        verify("experimental_window_scoring", segment_score)

        if check_ast:
            def ast() -> dict:
                result = _expect(client.get("/api/ast/status"))
                assert isinstance(result["available"], bool)
                return {
                    "available": result["available"],
                    "note": "AST model load status only; not independent accuracy",
                }
            verify("optional_ast_status", ast)

    return {
        "status": "passed" if all(row["passed"] for row in checks.values()) else "failed",
        "mode": "models_required" if require_models else "core_with_optional_models",
        "checks": checks,
        "checked_routes": len(checks),
        "scientific_limits": (
            "Generated non-speech fixtures and local artifact checks verify the "
            "application plumbing only. No deepfake accuracy, source separation, "
            "blind-RIR inference, breath detection, or tamper localization is validated."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-models", action="store_true",
                        help="Fail if baseline, six-feature candidate or comparison is unavailable")
    parser.add_argument("--check-ast", action="store_true",
                        help="Check optional local AST checkpoint readiness (may load a large model)")
    parser.add_argument("--output", type=Path,
                        help="Write a NEW report file; never overwrite previous results")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {args.output}")
        report = check_endpoints(
            require_models=args.require_models,
            check_ast=args.check_ast,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2)
    except (FileExistsError, OSError, ValueError, RuntimeError) as exc:
        print(f"Demo acceptance failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
