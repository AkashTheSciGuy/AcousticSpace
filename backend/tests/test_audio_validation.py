"""Tests for the validation-only audio upload endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def valid_wav() -> bytes:
    return b"RIFF\x04\x00\x00\x00WAVE"


def upload(filename: str, content: bytes, content_type: str = "application/octet-stream"):
    return client.post(
        "/api/audio/validate",
        files={"file": (filename, content, content_type)},
    )


def test_accepts_wav_with_valid_header() -> None:
    result = upload("recording.wav", valid_wav(), "audio/wav")
    assert result.status_code == 200
    assert result.json()["status"] == "valid"
    assert result.json()["format"] == "wav"
    assert result.json()["size_bytes"] == len(valid_wav())


def test_accepts_mp3_with_id3_header() -> None:
    result = upload("voice.MP3", b"ID3\x04\x00\x00\x00\x00\x00\x00", "audio/mpeg")
    assert result.status_code == 200
    assert result.json()["format"] == "mp3"


def test_accepts_mp3_with_frame_sync() -> None:
    result = upload("sample.mp3", b"\xff\xfb\x90\x64" + b"\x00" * 8)
    assert result.status_code == 200


def test_rejects_unsupported_extension() -> None:
    result = upload("data.txt", b"plain text")
    assert result.status_code == 415


def test_rejects_empty_file() -> None:
    result = upload("empty.wav", b"")
    assert result.status_code == 400
    assert "empty" in result.json()["detail"].lower()


def test_rejects_wrong_wav_signature() -> None:
    result = upload("fake.wav", b"not actually a WAV file")
    assert result.status_code == 400


def test_rejects_wrong_mp3_signature() -> None:
    result = upload("fake.mp3", b"not actually MP3 data")
    assert result.status_code == 400


def test_rejects_missing_file_field() -> None:
    result = client.post("/api/audio/validate")
    assert result.status_code == 422


def test_rejects_file_over_20_mib() -> None:
    content = valid_wav() + b"\x00" * (20 * 1024 * 1024)
    result = upload("oversized.wav", content)
    assert result.status_code == 413


def test_sanitizes_uploaded_name() -> None:
    result = upload(r"C:\recordings\take.wav", valid_wav())
    assert result.status_code == 200
    assert result.json()["filename"] == "take.wav"


def test_detects_header_without_trusting_mime_type() -> None:
    result = upload("clip.wav", valid_wav(), "application/octet-stream")
    assert result.status_code == 200


def test_capabilities_expose_supported_formats_and_upload_limit() -> None:
    response = client.get("/api/audio/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert data["supported_formats"] == ["mp3", "wav"]
    assert data["max_upload_bytes"] == 20 * 1024 * 1024


def test_capabilities_match_preprocessing_settings() -> None:
    from app.preprocessing import MAX_DURATION_SECONDS, TARGET_SAMPLE_RATE

    data = client.get("/api/audio/capabilities").json()
    assert data["max_inspection_duration_seconds"] == MAX_DURATION_SECONDS
    assert data["processing_sample_rate_hz"] == TARGET_SAMPLE_RATE


def test_capabilities_do_not_claim_a_deepfake_model() -> None:
    data = client.get("/api/audio/capabilities").json()
    assert data["upload_validation_available"] is True
    assert data["feature_inspection_available"] is True
    assert data["deepfake_classification_available"] is False


def test_capabilities_endpoint_is_documented() -> None:
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "get" in response.json()["paths"]["/api/audio/capabilities"]
