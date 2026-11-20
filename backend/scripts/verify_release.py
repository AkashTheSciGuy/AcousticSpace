"""Read-only release preflight for locally trained AcousticSpace artifacts.

Run from backend/:
    .venv\\Scripts\\python.exe scripts\\verify_release.py

This validates local model interfaces and saved reports, but is not a
substitute for integration tests or an independent evaluation dataset.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.comparison_api import read_comparison  # noqa: E402
from app.model_inference import model_status, predict_audio  # noqa: E402


def check_release(audio_path: Path | None = None) -> dict:
    """Inspect the two configured model variants and local comparison report."""
    models = {variant: model_status(variant) for variant in ('baseline', 'acoustic')}
    comparison = read_comparison()
    checks = {
        'baseline_model_loaded': models['baseline']['available'],
        'acoustic_candidate_loaded': models['acoustic']['available'],
        'comparison_report_valid': comparison is not None,
        'baseline_evaluation_available': models['baseline'].get('evaluation') is not None,
    }
    inference = {}
    if audio_path is not None:
        from app.preprocessing import load_audio
        audio = load_audio(audio_path)
        for variant in ('baseline', 'acoustic'):
            started = perf_counter()
            try:
                prediction = predict_audio(audio, variant)
                checks[f'{variant}_inference_smoke'] = True
                inference[variant] = {**prediction, 'processing_time_ms': round((perf_counter() - started) * 1000, 3)}
            except RuntimeError as exc:
                checks[f'{variant}_inference_smoke'] = False
                inference[variant] = {'error': str(exc)}
    return {
        'inference_smoke': inference,
        'status': 'ready_for_local_demo' if all(checks.values()) else 'incomplete',
        'checks': checks,
        'model_messages': {variant: models[variant]['message'] for variant in models},
        'warning': ('Checks only local artifacts, saved self-reported metrics and model interfaces. '
                    'No independent RIR-based detection or forensic accuracy verified.'),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, help='Optional local WAV/MP3 for actual inference smoke checks; not an accuracy evaluation')
    args = parser.parse_args(argv)
    try:
        result = check_release(args.audio)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Release preflight failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ready_for_local_demo' else 2


if __name__ == '__main__':
    raise SystemExit(main())
