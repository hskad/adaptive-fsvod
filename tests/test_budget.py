import importlib

import torch


def test_actual_update_cap_preserves_schedule_prefix(corpus, tmp_path, monkeypatch):
    module = importlib.import_module("adaptive_fsvod.train")
    snapshots = []
    original = module.save_tensor

    def capture(path, checkpoint):
        if path.name == "last.pt" and checkpoint["step"] == 2:
            snapshots.append({key: value.detach().clone() for key, value in checkpoint["model"].items()})
        original(path, checkpoint)

    monkeypatch.setattr(module, "save_tensor", capture)

    def arguments(name, cap=None):
        flags = {"train-manifest": corpus / "train.json", "val-manifest": corpus / "val.json",
                 "train-cache": corpus / "cache/train", "val-cache": corpus / "cache/val",
                 "train-episodes": corpus / "train_episodes.json", "val-episodes": corpus / "val_episodes.json",
                 "config": corpus / "fixed.json", "output": tmp_path / name,
                 "steps": 4, "eval-every": 2, "device": "cpu"}
        if cap is not None:
            flags["stop-after"] = cap
        return module.parser().parse_args([x for key, value in flags.items() for x in ("--" + key, str(value))])

    module.train(arguments("full"))
    module.train(arguments("limited", 2))
    assert len(snapshots) == 2
    for key in snapshots[0]:
        assert torch.equal(snapshots[0][key], snapshots[1][key]), key
    limited = torch.load(tmp_path / "limited/last.pt", weights_only=True)
    assert limited["step"] == 2
    assert limited["arguments"]["steps"] == 4
    assert limited["arguments"]["stop_after"] == 2
    assert len((tmp_path / "limited/training.jsonl").read_text().splitlines()) == 2
