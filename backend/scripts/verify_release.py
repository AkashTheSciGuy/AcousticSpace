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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.comparison_api import read_comparison  # noqa: E402
from app.model_inference import model_status  # noqa: E402


def check_release() -> dict:
    """Inspect the two configured model variants and local comparison report."""
    models = {variant: model_status(variant) for variant in ('baseline', 'acoustic')}
    comparison = read_comparison()
    checks = {
        'baseline_model_loaded': models['baseline']['available'],
        'acoustic_candidate_loaded': models['acoustic']['available'],
        'comparison_report_valid': comparison is not None,
        'baseline_evaluation_available': models['baseline'].get('evaluation') is not None,
    }
    return {
        'status': 'ready_for_local_demo' if all(checks.values()) else 'incomplete',
        'checks': checks,
        'model_messages': {variant: models[variant]['message'] for variant in models},
        'warning': ('Checks only local artifacts, saved self-reported metrics and model interfaces. '
                    'No independent RIR-based detection or forensic accuracy verified.'),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        result = check_release()
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'Release preflight failed: {exc}', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'ready_for_local_demo' else 2


if __name__ == '__main__':
    raise SystemExit(main())
