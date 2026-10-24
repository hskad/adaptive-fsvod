import torch

from adaptive_fsvod.cache import CachedEpisodes
from adaptive_fsvod.data import VideoManifest, read_json
from adaptive_fsvod.loss import DetectionLoss
from adaptive_fsvod.model import ModelConfig, TemporalDetector, select_memory


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
