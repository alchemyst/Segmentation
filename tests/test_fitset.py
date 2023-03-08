import segment
import pytest

a = segment.LineThroughEndPoints(segment.DataContainer([1, 2], [1, 2]))
b = segment.LineThroughEndPoints(segment.DataContainer([2, 3], [2, 1]))

def test_init():
    s = segment.FitSet()
    assert len(s) == 0

def test_append():
    s = segment.FitSet([a])
    assert len(s) == 1, "Construction failed to add one element"

    s.append(b)
    assert len(s) == 2, "Append failed to add one element"

def test_eval_single():
    s = segment.FitSet([a])
    assert s.eval(1.5) == pytest.approx(1.5, 2), "Evaluation at single point failed"

def test_eval_many():
    s = segment.FitSet([a, b])

    for x, expected_y in zip([1.5, 2.5], [1.5, 1.5]):
        assert s.eval(x) == pytest.approx(expected_y, 2), "Evaluation with multiple fits failed"
