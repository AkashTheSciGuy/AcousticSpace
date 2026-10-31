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
