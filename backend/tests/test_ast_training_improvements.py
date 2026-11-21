"""AST training guardrails: no silent underweighting of the last minibatch."""
import pytest

from scripts.train_ast import accumulation_divisor


@pytest.mark.parametrize("step,expected", [(1, 4), (2, 4), (3, 4),
                                           (4, 4), (5, 1)])
def test_partial_accumulation_is_scaled_correctly(step, expected):
    assert accumulation_divisor(step, total=5, accumulate=4) == expected


def test_full_groups_match_accumulation_factor():
    assert [accumulation_divisor(i, 8, 4) for i in range(1, 9)] == [4] * 8


@pytest.mark.parametrize("step,total,accumulate", [
    (0, 5, 4), (6, 5, 4), (1, 0, 4), (1, 5, 0)
])
def test_invalid_accumulation_parameters_fail(step, total, accumulate):
    with pytest.raises(ValueError):
        accumulation_divisor(step, total, accumulate)
