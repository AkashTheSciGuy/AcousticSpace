"""FLAC training ingestion must not expand public API format support."""
from pathlib import Path

import numpy as np
import soundfile as sf

from app.preprocessing import load_audio


def test_decode_official_style_flac(tmp_path: Path):
    rate = 16000
    t = np.arange(rate, dtype=np.float32) / rate
    file = tmp_path / 'LA_T_1000137.flac'
    sf.write(file, np.sin(2 * np.pi * 440 * t) * 0.1, rate)
    decoded = load_audio(file)
    assert decoded.sample_rate == 16000
    assert decoded.original_sample_rate == 16000
    assert decoded.duration_seconds == 1.0
    assert decoded.rms_amplitude > 0
