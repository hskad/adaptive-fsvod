from dataclasses import asdict

import pytest
import torch

from adaptive_fsvod.cache import CachedEpisodes, fit_feature_center
from adaptive_fsvod.data import VideoManifest, write_json
from adaptive_fsvod.model import ModelConfig, TemporalDetector
from adaptive_fsvod.train import parser, train


def test_center_is_class_balanced_training_only_and_adds_no_parameters(corpus):
    cached = CachedEpisodes(VideoManifest(corpus / "train.json"), corpus / "cache/train")
    center = fit_feature_center(cached)
    # Synthetic support samples in each class all share its known unit center.
    expected = torch.stack([torch.load(cached.root / "support" / f"{examples[0]['key']}.pt", weights_only=True)
                            for examples in cached.pool.values()]).mean(0)
    torch.testing.assert_close(center, expected)
    validation = CachedEpisodes(VideoManifest(corpus / "val.json"), corpus / "cache/val")
    with pytest.raises(ValueError, match="training support features only"):
        fit_feature_center(validation)
    baseline = TemporalDetector(ModelConfig(input_dim=16, hidden_dim=16))
    centered = TemporalDetector(ModelConfig(input_dim=16, hidden_dim=16, feature_centering=True))
    assert not centered.feature_center.requires_grad
    assert "feature_center" in centered.state_dict()
    assert "feature_center" not in baseline.state_dict()
    assert sum(p.numel() for p in baseline.parameters()) == sum(p.numel() for p in centered.parameters())


def test_center_saved_in_checkpoint_and_resume_rejects_changed_support(corpus, tmp_path):
    config = corpus / "centered.json"
    write_json(config, asdict(ModelConfig(input_dim=16, hidden_dim=16, dropout=0, feature_centering=True)))
    values = {"train-manifest": corpus / "train.json", "val-manifest": corpus / "val.json",
              "train-cache": corpus / "cache/train", "val-cache": corpus / "cache/val",
              "train-episodes": corpus / "train_episodes.json", "val-episodes": corpus / "val_episodes.json",
              "config": config, "output": tmp_path / "run", "steps": 2, "eval-every": 1, "device": "cpu"}
    arguments = [part for key, value in values.items() for part in ("--" + key, str(value))]
    checkpoint_path = train(parser().parse_args(arguments))
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    cached = CachedEpisodes(VideoManifest(corpus / "train.json"), corpus / "cache/train")
    torch.testing.assert_close(checkpoint["model"]["feature_center"], fit_feature_center(cached))
    assert checkpoint["feature_center"]["source"] == "class-balanced training support cache"
    example = next(iter(cached.pool.values()))[0]
    path = cached.root / "support" / f"{example['key']}.pt"
    torch.save(torch.ones(16), path)
    with pytest.raises(ValueError, match="feature center differs"):
        train(parser().parse_args(arguments + ["--resume", str(tmp_path / "run/last.pt")]))
