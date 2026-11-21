"""Reference-aided acoustic-consistency features for controlled RIR experiments.

Requires dry source audio and separately observed wet speech/noise stems.
These are ORACLE references available only in the controlled experiment.
No ground-truth RIR files, room IDs, labels, or filenames enter the features.
This is NOT blind RIR recovery from a single mixed speech recording.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

from app.rir_analysis import analyze_rir

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class ConsistencyFeatures:
    rt60_difference_seconds: float
    rt60_missing: float
    c50_difference_db: float
    drr_difference_db: float
    late_energy_difference: float
    decay_profile_distance_db: float
    spectral_shape_distance_db: float

    def as_array(self) -> np.ndarray:
        return np.array(list(asdict(self).values()), dtype=np.float64)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


FEATURE_NAMES = tuple(ConsistencyFeatures.__dataclass_fields__)


def _signal(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 256 <= values.size <= 12 * SAMPLE_RATE:
        raise ValueError(f"{name} must be finite mono audio lasting at most 12 seconds.")
    if not np.isfinite(values).all() or np.sqrt(np.mean(values * values)) < 1e-7:
        raise ValueError(f"{name} must be finite and non-silent.")
    return values


def estimate_reference_transfer(dry: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Regularized frequency-domain transfer estimate from a known dry source.

    Uses a weak Wiener/Tikhonov floor and normalizes global gain. Results
    depend on source spectral excitation and are not physical ground truth.
    """
    source = _signal(dry, "dry reference")
    observation = _signal(wet, "wet stem")
    taps = observation.size - source.size + 1
    if not 256 <= taps <= 2 * SAMPLE_RATE:
        raise ValueError("Stem/reference lengths imply an invalid RIR duration.")
    fft_size = next_fast_len(observation.size)
    source_fft = rfft(source, fft_size)
    output_fft = rfft(observation, fft_size)
    source_power = np.square(np.abs(source_fft))
    regularization = 1e-6 * float(source_power.max())
    transfer_fft = (output_fft * np.conj(source_fft) /
                    (source_power + regularization))
    response = irfft(transfer_fft, fft_size)[:taps]
    if not np.isfinite(response).all():
        raise ValueError("Transfer estimate is not finite.")
    norm = float(np.linalg.norm(response))
    if norm < 1e-8:
        raise ValueError("Transfer estimate contains insufficient energy.")
    return response / norm


def _response_descriptors(ir: np.ndarray) -> tuple:
    """Extract bounded, amplitude-normalized envelope and spectral descriptors."""
    # The largest impulse is used as a working onset, not verified direct sound.
    onset = int(np.argmax(np.abs(ir)))
    response = ir[onset:]
    if response.size < 256:
        raise ValueError("Estimated transfer has no measurable decay after onset.")
    energy = np.square(response)
    total = float(energy.sum())
    if total < 1e-15:
        raise ValueError("Estimated transfer has no measurable energy.")

    direct = min(energy.size, int(.0025 * SAMPLE_RATE))
    early = min(energy.size, int(.05 * SAMPLE_RATE))

    def ratio_db(left: float, right: float) -> float:
        return float(10 * np.log10((left + 1e-12) / (right + 1e-12)))

    drr = ratio_db(float(energy[:direct].sum()), float(energy[direct:].sum()))
    c50 = ratio_db(float(energy[:early].sum()), float(energy[early:].sum()))
    late = float(energy[early:].sum() / total)

    # A bounded Schroeder curve, sampled at regular 20-ms intervals.
    edc = np.cumsum(energy[::-1], dtype=np.float64)[::-1]
    offsets = np.arange(0, int(.6 * SAMPLE_RATE), 320, dtype=int)
    sampled = edc[np.minimum(offsets, edc.size - 1)] / total
    decay_db = np.clip(10 * np.log10(np.maximum(sampled, 1e-8)), -60, 0)

    # Coarse log-transfer-spectrum shape, relative to its average.
    spectrum = np.abs(rfft(ir, n=next_fast_len(max(ir.size, 16384))))
    freqs = np.fft.rfftfreq(next_fast_len(max(ir.size, 16384)),
                           d=1 / SAMPLE_RATE)
    bounds = ((80, 300), (300, 750), (750, 1500), (1500, 3000),
              (3000, 5000), (5000, 7500))
    band_db = np.array([
        20 * np.log10(max(float(np.sqrt(np.mean(
            np.square(spectrum[(freqs >= low) & (freqs < high)])))), 1e-8))
        for low, high in bounds
    ])
    band_db -= band_db.mean()

    # RT60 is a supporting descriptor only; unavailable estimates are marked.
    estimate = analyze_rir(ir, SAMPLE_RATE).rt60_t20_seconds
    return estimate, c50, drr, late, decay_db, band_db


def reference_consistency_features(
    dry_speech: np.ndarray, wet_speech: np.ndarray,
    dry_ambience: np.ndarray, wet_ambience: np.ndarray,
) -> ConsistencyFeatures:
    """Compare estimated speech/ambience transfers without oracle room labels."""
    speech_ir = estimate_reference_transfer(dry_speech, wet_speech)
    ambience_ir = estimate_reference_transfer(dry_ambience, wet_ambience)
    s_rt, s_c50, s_drr, s_late, s_decay, s_bands = _response_descriptors(speech_ir)
    a_rt, a_c50, a_drr, a_late, a_decay, a_bands = _response_descriptors(ambience_ir)
    rt_missing = float(s_rt is None or a_rt is None)
    values = ConsistencyFeatures(
        rt60_difference_seconds=abs(s_rt - a_rt) if not rt_missing else 0.0,
        rt60_missing=rt_missing,
        c50_difference_db=abs(s_c50 - a_c50),
        drr_difference_db=abs(s_drr - a_drr),
        late_energy_difference=abs(s_late - a_late),
        decay_profile_distance_db=float(np.mean(np.abs(s_decay - a_decay))),
        spectral_shape_distance_db=float(np.mean(np.abs(s_bands - a_bands))),
    )
    if not np.isfinite(values.as_array()).all():
        raise ValueError("Consistency features contain invalid values.")
    return values
