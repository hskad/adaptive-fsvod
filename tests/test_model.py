import torch
import pytest

from adaptive_fsvod.cache import CachedEpisodes
from adaptive_fsvod.data import VideoManifest, read_json
from adaptive_fsvod.loss import DetectionLoss
from adaptive_fsvod.model import ModelConfig, TemporalDetector, confidence_gate, select_memory


def test_memory_excludes_background_and_limits_each_class():
    probabilities = torch.tensor([[.9, .05, .05], [.8, .1, .1], [.05, .9, .05], [.1, .1, .8]])
    config = ModelConfig(memory_mode="fixed", fixed_threshold=.6, max_memory_per_class=1)
    selected, _ = select_memory(probabilities, config)
    assert selected.tolist() == [0, 2]


def test_adaptive_bounds_and_empty_memory():
    probabilities = torch.tensor([[.01, .01, .98], [.1, .1, .8]])
    config = ModelConfig(memory_mode="adaptive", threshold_min=.5, threshold_max=.8)
    selected, thresholds = select_memory(probabilities, config)
    assert selected.numel() == 0
    assert torch.all((thresholds >= .5) & (thresholds <= .8))


def test_empty_memory_bypass_and_reset(corpus):
    torch.manual_seed(5)
    cache = CachedEpisodes(VideoManifest(corpus / "train.json"), corpus / "cache/train")
    episode = read_json(corpus / "train_episodes.json")["episodes"][0]
    frames, support, _, _ = cache.load(episode)
    model = TemporalDetector(ModelConfig(input_dim=16, hidden_dim=16, fixed_threshold=1, dropout=0)).eval()
    output1, output2 = model(frames, support), model(frames, support)
    for first, second in zip(output1, output2):
        assert torch.equal(first["logits"], second["logits"])
        assert first["memory"].shape == (0, 16)
    # With all temporal memory rejected, changing preceding frames cannot affect a frame.
    alone = model([frames[-1]], support)[0]
    assert torch.equal(alone["logits"], output1[-1]["logits"])


def test_nonempty_memory_can_train(corpus):
    cache = CachedEpisodes(VideoManifest(corpus / "train.json"), corpus / "cache/train")
    episode = read_json(corpus / "train_episodes.json")["episodes"][0]
    frames, support, targets, _ = cache.load(episode)
    model = TemporalDetector(ModelConfig(input_dim=16, hidden_dim=16, fixed_threshold=0, dropout=0))
    outputs = model(frames, support)
    assert outputs[0]["memory"].shape[0] > 0
    criterion = DetectionLoss()
    loss = sum(criterion(out, target)["total"] for out, target in zip(outputs, targets))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.attention.in_proj_weight.grad is not None
    assert torch.isfinite(model.attention.in_proj_weight.grad).all()


def test_matching_trains_boxes_even_when_iou_is_zero():
    predicted = torch.tensor([[.1, .1, .05, .05]], requires_grad=True)
    logits = torch.tensor([[1., 0.]], requires_grad=True)
    target = {"boxes": torch.tensor([[.8, .8, .1, .1]]), "labels": torch.tensor([0])}
    loss = DetectionLoss()({"boxes": predicted, "logits": logits}, target)["total"]
    loss.backward()
    assert predicted.grad.abs().sum() > 0


def test_empty_frame_supervises_background():
    logits = torch.tensor([[4., 0.]], requires_grad=True)
    boxes = torch.tensor([[.5, .5, .2, .2]], requires_grad=True)
    target = {"boxes": torch.empty(0, 4), "labels": torch.empty(0, dtype=torch.long)}
    loss = DetectionLoss()({"logits": logits, "boxes": boxes}, target)["total"]
    loss.backward()
    assert logits.grad[0, 0] > 0 and logits.grad[0, 1] < 0


def test_confidence_gate_margin_and_background():
    probabilities = torch.tensor([[.8, .1, .1], [.6, .3, .1], [.2, .1, .7], [1., 0., 0.]])
    gate = confidence_gate(probabilities, .7)
    torch.testing.assert_close(gate, torch.tensor([1 / 3, 0., 0., 1.]))


def test_zero_gate_preserves_single_frame_predictions():
    from dataclasses import replace
    cfg = ModelConfig(input_dim=16, hidden_dim=16, dropout=0, fusion_gate="confidence")
    model = TemporalDetector(cfg).eval()
    # Make the temporal residual nonzero so this check cannot pass just because
    # its final layer was initialized to zero.
    torch.nn.init.normal_(model.ffn[-1].weight, std=.2)
    model.background_head.bias.data.fill_(100)
    features = torch.randn(8, 16)
    boxes = torch.full((8, 4), .3)
    prototypes = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    with torch.no_grad():
        gated = model.forward_frame(features, boxes, prototypes, torch.randn(5, 16))
        plain = model.forward_frame(features, boxes, prototypes, None)
    assert gated["fusion_gate"].count_nonzero() == 0
    assert torch.equal(gated["logits"], plain["logits"])
    assert torch.equal(gated["boxes"], plain["boxes"])
    legacy = TemporalDetector(replace(cfg, fusion_gate="none"))
    assert set(legacy.state_dict()) == set(model.state_dict())
    assert sum(p.numel() for p in legacy.parameters()) == sum(p.numel() for p in model.parameters())


def test_gated_temporal_branch_gets_gradients(corpus):
    cache = CachedEpisodes(VideoManifest(corpus / "train.json"), corpus / "cache/train")
    episode = read_json(corpus / "train_episodes.json")["episodes"][0]
    frames, support, targets, _ = cache.load(episode)
    model = TemporalDetector(ModelConfig(input_dim=16, hidden_dim=16, fixed_threshold=0,
                                        dropout=0, fusion_gate="confidence", gate_threshold=0))
    outputs = model(frames, support)
    assert outputs[0]["fusion_gate"].count_nonzero() == 0
    assert outputs[1]["fusion_gate"].max() > 0
    loss = sum(DetectionLoss()(out, target)["total"] for out, target in zip(outputs, targets))
    loss.backward()
    assert model.attention.in_proj_weight.grad is not None
    assert model.attention.in_proj_weight.grad.abs().sum() > 0


def test_objectness_prior_has_expected_attention_ratio():
    model = TemporalDetector(ModelConfig(input_dim=4, hidden_dim=4, attention_heads=1,
                                        dropout=0, memory_weighting="objectness")).eval()
    with torch.no_grad():
        model.attention.in_proj_weight.zero_()
        model.attention.in_proj_weight[8:12].copy_(torch.eye(4))
        model.attention.in_proj_bias.zero_()
        model.attention.out_proj.weight.copy_(torch.eye(4))
        model.attention.out_proj.bias.zero_()
    contexts = []
    handle = model.attention.register_forward_hook(lambda module, inputs, output: contexts.append(output[0].detach()))
    with torch.no_grad():
        model.forward_frame(torch.tensor([[0., 0., 1., 0.]]), torch.full((1, 4), .3),
                            torch.tensor([[0., 0., 1., 0.]]), torch.eye(4)[:2],
                            objectness=torch.ones(1), memory_quality=torch.tensor([.9, .1]))
    handle.remove()
    torch.testing.assert_close(contexts[0][0, 0], torch.tensor([.9, .1, 0., 0.]))


def test_uniform_memory_quality_matches_unweighted_and_missing_scores_fail():
    from dataclasses import replace
    cfg = ModelConfig(input_dim=16, hidden_dim=16, dropout=0)
    baseline = TemporalDetector(cfg).eval()
    weighted = TemporalDetector(replace(cfg, memory_weighting="objectness")).eval()
    weighted.load_state_dict(baseline.state_dict())
    features = torch.randn(8, 16)
    boxes = torch.full((8, 4), .3)
    prototypes = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
    memory = torch.randn(5, 16)
    with torch.no_grad():
        plain = baseline.forward_frame(features, boxes, prototypes, memory)
        equal = weighted.forward_frame(features, boxes, prototypes, memory,
                                       objectness=torch.ones(8), memory_quality=torch.ones(5))
    torch.testing.assert_close(plain["logits"], equal["logits"])
    torch.testing.assert_close(plain["boxes"], equal["boxes"])
    with pytest.raises(ValueError, match="objectness score"):
        weighted.forward_frame(features, boxes, prototypes, memory)
    with torch.no_grad():
        zero = weighted.forward_frame(features, boxes, prototypes, memory,
                                      objectness=torch.ones(8), memory_quality=torch.zeros(5))
        alone = weighted.forward_frame(features, boxes, prototypes, None, objectness=torch.ones(8))
    assert torch.equal(zero["logits"], alone["logits"])
    assert torch.equal(zero["boxes"], alone["boxes"])


def test_weighted_memory_retains_alignment_and_can_train(corpus):
    cached = CachedEpisodes(VideoManifest(corpus / "train.json"), corpus / "cache/train")
    episode = read_json(corpus / "train_episodes.json")["episodes"][0]
    frames, support, targets, _ = cached.load(episode)
    for frame in frames:
        frame["objectness"] = torch.linspace(.1, .9, len(frame["features"]))
    model = TemporalDetector(ModelConfig(input_dim=16, hidden_dim=16, dropout=0,
                                        fixed_threshold=0, memory_weighting="objectness"))
    outputs = model(frames, support)
    for frame, output in zip(frames, outputs):
        torch.testing.assert_close(output["memory_quality"], frame["objectness"][output["memory_indices"]])
    loss = sum(DetectionLoss()(out, target)["total"] for out, target in zip(outputs, targets))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.attention.in_proj_weight.grad.abs().sum() > 0
