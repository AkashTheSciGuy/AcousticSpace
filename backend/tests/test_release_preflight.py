"""Unit tests for read-only final demo preflight."""
import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'verify_release.py'
    spec = importlib.util.spec_from_file_location('verify_release', path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_incomplete_when_candidate_missing(monkeypatch, capsys):
    mod = module()
    def status(variant):
        return {'available': variant == 'baseline', 'message': 'ok' if variant == 'baseline' else 'missing',
                **({'evaluation': {}} if variant == 'baseline' else {})}
    monkeypatch.setattr(mod, 'model_status', status)
    monkeypatch.setattr(mod, 'read_comparison', lambda: {'experiment': 'test'})
    assert mod.main([]) == 2
    assert '"status": "incomplete"' in capsys.readouterr().out


def test_ready_only_with_both_models_and_reports(monkeypatch, capsys):
    mod = module()
    monkeypatch.setattr(mod, 'model_status', lambda variant: {'available': True, 'message': 'ok',
                                                               **({'evaluation': {}} if variant == 'baseline' else {})})
    monkeypatch.setattr(mod, 'read_comparison', lambda: {'experiment': 'test'})
    assert mod.main([]) == 0
    output = capsys.readouterr().out
    assert 'ready_for_local_demo' in output
    assert 'comparison_report_valid' in output


def test_missing_baseline_evaluation_is_incomplete(monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, 'model_status', lambda variant: {'available': True, 'message': 'ok'})
    monkeypatch.setattr(mod, 'read_comparison', lambda: {'experiment': 'test'})
    assert mod.check_release()['checks']['baseline_evaluation_available'] is False


def test_missing_comparison_is_incomplete(monkeypatch):
    mod = module()
    monkeypatch.setattr(mod, 'model_status', lambda variant: {'available': True, 'message': 'ok',
                                                               **({'evaluation': {}} if variant == 'baseline' else {})})
    monkeypatch.setattr(mod, 'read_comparison', lambda: None)
    assert mod.check_release()['checks']['comparison_report_valid'] is False
