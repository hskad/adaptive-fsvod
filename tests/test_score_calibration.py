import copy

import pytest
import torch

from adaptive_fsvod.evaluate import decode


def example():
    return {"logits": torch.tensor([[4., 0.], [3., 0.], [0., 5.]]),
            "boxes": torch.tensor([[.5, .5, .4, .4], [.5, .5, .4, .4], [.1, .1, .1, .1]])}


def test_zero_exponent_preserves_baseline_without_quality_input():
    output = example()
    native = decode(output, ["a"], 100, 100)
    assert native == decode(output, ["a"], 100, 100, objectness_beta=0)
    assert native == decode(output, ["a"], 100, 100, objectness=torch.tensor([float("nan")]), objectness_beta=0)


def test_quality_changes_ranking_before_nms_and_preserves_background_decision():
    output = example()
    saved = copy.deepcopy(output)
    quality = torch.tensor([.01, .9, 1.])
    native = decode(output, ["a"], 100, 100)
    calibrated = decode(output, ["a"], 100, 100, objectness=quality, objectness_beta=.5)
    assert len(native) == len(calibrated) == 1
    expected = float(output["logits"].softmax(-1)[1, 0] * quality[1].sqrt())
    assert calibrated[0]["score"] == pytest.approx(expected)
    assert calibrated[0]["bbox"] == native[0]["bbox"]
    for name in output:
        assert torch.equal(output[name], saved[name])


def test_zero_quality_is_rejected_by_score_floor():
    assert decode(example(), ["a"], 100, 100, objectness=torch.zeros(3), objectness_beta=1) == []


@pytest.mark.parametrize("quality", [None, torch.ones(2), torch.tensor([1., 1., -1.]),
                                    torch.tensor([1., 2., 1.]), torch.tensor([1., float("nan"), 1.])])
def test_invalid_quality_is_rejected(quality):
    with pytest.raises(ValueError):
        decode(example(), ["a"], 100, 100, objectness=quality, objectness_beta=.5)


@pytest.mark.parametrize("beta", [-1., float("inf"), float("nan")])
def test_invalid_exponent_is_rejected(beta):
    with pytest.raises(ValueError):
        decode(example(), ["a"], 100, 100, objectness=torch.ones(3), objectness_beta=beta)
