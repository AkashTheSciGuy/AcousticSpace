"""Generate a clearly synthetic RIR for API demonstration only.

Run from backend/: python scripts/generate_demo_rir.py
DO NOT use generated files to claim real-world deepfake model accuracy.
"""

from pathlib import Path
import wave

import numpy as np


def main() -> None:
    rate = 16_000
    seconds = 2.0
    rt60 = 0.6
    times = np.arange(int(seconds * rate)) / rate
    rng = np.random.default_rng(42)
    samples = rng.normal(size=times.size) * np.exp(-np.log(1000) * times / rt60)
    samples[0] = 10
    samples /= 11
    path = Path('synthetic_rir_demo.wav')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(np.rint(np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes())
    print(f'Wrote {path.resolve()} (synthetic, target RT60={rt60}s)')


if __name__ == '__main__':
    main()
