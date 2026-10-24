import importlib

import pytest
import torch

from adaptive_fsvod.data import read_json, write_json


def test_resume_matches_uninterrupted_training(corpus, tmp_path, monkeypatch):
    module = importlib.import_module("adaptive_fsvod.train")
    config = read_json(corpus / "fixed.json")
    config["dropout"] = 0.2
    write_json(corpus / "fixed.json", config)

    def arguments(output, resume=None):
        flags = {"train-manifest": corpus / "train.json", "val-manifest": corpus / "val.json",
                 "train-cache": corpus / "cache/train", "val-cache": corpus / "cache/val",
                 "train-episodes": corpus / "train_episodes.json", "val-episodes": corpus / "val_episodes.json",
                 "config": corpus / "fixed.json", "output": output,
                 "steps": 8, "eval-every": 4, "device": "cpu"}
        if resume:
            flags["resume"] = resume
        command = [item for key, value in flags.items() for item in ["--" + key, str(value)]]
        return module.parser().parse_args(command)

    complete = tmp_path / "complete"
    interrupted = tmp_path / "interrupted"
    module.train(arguments(complete))
    save = module.save_tensor

    def disconnect(path, checkpoint):
        save(path, checkpoint)
        if path.name == "last.pt" and checkpoint["step"] == 4:
            raise RuntimeError("simulated disconnect")

    monkeypatch.setattr(module, "save_tensor", disconnect)
    with pytest.raises(RuntimeError, match="simulated disconnect"):
        module.train(arguments(interrupted))
    assert (interrupted / "best.pt").exists()
    monkeypatch.setattr(module, "save_tensor", save)
    module.train(arguments(interrupted, interrupted / "last.pt"))
    full = torch.load(complete / "last.pt", weights_only=True)
    resumed = torch.load(interrupted / "last.pt", weights_only=True)
    assert full["step"] == resumed["step"] == 8
    for name in full["model"]:
        assert torch.equal(full["model"][name], resumed["model"][name]), name
