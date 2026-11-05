"""Room impulse-response descriptors for a measured or simulated RIR.

Input MUST be an impulse-response recording, not arbitrary speech.
RT60 uses the T20 regression region of the Schroeder energy decay curve;
results are estimates and depend on signal-to-noise ratio and decay quality.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class RirMetrics:
    onset_seconds: float
    rt60_t20_seconds: float | None
    clarity_c50_db: float | None
    direct_to_reverberant_ratio_db: float | None
    decay_fit_r_squared: float | None
    sample_rate_hz: int
    analysis_duration_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _energy_ratio_db(numerator: float, denominator: float) -> float | None:
    if denominator <= 1e-15 or numerator <= 1e-15:
        return None
    return float(10 * np.log10(numerator / denominator))


def analyze_rir(samples: np.ndarray, sample_rate: int) -> RirMetrics:
    """Estimate T20-derived RT60, C50, and approximate DRR from a mono RIR.

    The onset is the maximum absolute sample. The T20 fit requires an EDC
    segment spanning -5 to -25 dB and a sufficiently linear regression.
    Missing or unreliable results are returned as null, never invented.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or values.size < 256:
        raise ValueError("RIR must contain at least 256 mono samples.")
    if not np.isfinite(values).all():
        raise ValueError("RIR samples must be finite.")
    if sample_rate < 1000:
        raise ValueError("Sample rate must be at least 1000 Hz.")
    if np.max(np.abs(values)) < 1e-8:
        raise ValueError("RIR is silent or too quiet for analysis.")

    onset = int(np.argmax(np.abs(values)))
    tail = values[onset:]
    if tail.size < max(256, round(sample_rate * 0.05)):
        raise ValueError("RIR has insufficient audio following the impulse onset.")
    energy = np.square(tail)
    total = float(np.sum(energy))
    if total <= 1e-15:
        raise ValueError("RIR energy is too small for analysis.")

    fifty_ms = min(len(energy), max(1, round(0.05 * sample_rate)))
    direct_ms = min(len(energy), max(1, round(0.0025 * sample_rate)))
    c50 = _energy_ratio_db(float(energy[:fifty_ms].sum()), float(energy[fifty_ms:].sum()))
    drr = _energy_ratio_db(float(energy[:direct_ms].sum()), float(energy[direct_ms:].sum()))

    # Integrated decay calculated backwards (Schroeder integration).
    edc = np.cumsum(energy[::-1], dtype=np.float64)[::-1]
    db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-15))
    fit_indices = np.flatnonzero((db <= -5.0) & (db >= -25.0))
    rt60: float | None = None
    r_squared: float | None = None
    if fit_indices.size >= 20:
        times = fit_indices / sample_rate
        slope, intercept = np.polyfit(times, db[fit_indices], 1)
        predictions = slope * times + intercept
        residual = float(np.square(db[fit_indices] - predictions).sum())
        spread = float(np.square(db[fit_indices] - db[fit_indices].mean()).sum())
        quality = 1.0 - residual / spread if spread > 1e-12 else 0.0
        # Reject poorly behaved decay curves instead of returning false precision.
        if slope < -1e-6 and quality >= 0.90:
            candidate = -60.0 / slope
            if 0.01 <= candidate <= 30.0:
                rt60 = float(candidate)
                r_squared = float(quality)

    return RirMetrics(
        onset_seconds=round(onset / sample_rate, 6),
        rt60_t20_seconds=round(rt60, 4) if rt60 is not None else None,
        clarity_c50_db=round(c50, 3) if c50 is not None else None,
        direct_to_reverberant_ratio_db=round(drr, 3) if drr is not None else None,
        decay_fit_r_squared=round(r_squared, 4) if r_squared is not None else None,
        sample_rate_hz=sample_rate,
        analysis_duration_seconds=round(len(tail) / sample_rate, 4),
    )
