"""Equal-budget 50-update feature-centering study, selected on validation."""

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from adaptive_fsvod.data import checksum, read_json, write_json
from adaptive_fsvod.model import ModelConfig
from run_revision import confirmation, immutable_json, training_run
from prepare_confirmation import prepare, cache


def search(root):
    candidates = []
    for method in ("fixed", "centered"):
        for threshold in (.6, .7, .8):
            name = f"{method}_tau{round(threshold * 100):03d}"
            path = root / "revision_v5/configs" / f"{name}.json"
            immutable_json(path, asdict(ModelConfig(fixed_threshold=threshold, feature_centering=method == "centered")))
            row = training_run(root, path, name, 42, "revision_v5", stop_after=50)
            candidates.append({**row, "method": method, "memory_threshold": threshold})
    selected = {method: max((r for r in candidates if r["method"] == method),
                             key=lambda r: (r["validation_AP"], r["memory_threshold"]))
                for method in ("fixed", "centered")}
    immutable_json(root / "revision_v5/selection.json", {
        "actual_optimizer_updates": 50, "schedule_horizon": 300, "warmup_updates": 15,
        "search_seed": 42, "seeds_to_replicate": [43, 44], "candidates": candidates, "selected": selected,
        "train_manifest_sha256": checksum(root / "pilot/train.json"),
        "val_manifest_sha256": checksum(root / "pilot/val.json"),
        "train_episodes_sha256": checksum(root / "pilot/train_episodes.json"),
        "val_episodes_sha256": checksum(root / "pilot/val_episodes.json")})
    print("FROZEN FAST SELECTION", json.dumps(selected), flush=True)


def replicate(root):
    selection = read_json(root / "revision_v5/selection.json")
    rows = []
    for method, selected in selection["selected"].items():
        rows.append(dict(selected, method=method))
        path = Path(selected["config"])
        if checksum(path) != selected["config_sha256"]:
            raise ValueError("Selected configuration changed")
        for seed in (43, 44):
            rows.append(dict(training_run(root, path, selected["name"], seed, "revision_v5", stop_after=50), method=method))
    write_json(root / "revision_v5/replications.json", rows)
    for method in ("fixed", "centered"):
        values = [r["validation_AP"] for r in rows if r["method"] == method]
        print("VALIDATION", method, values, "mean", statistics.mean(values), "sample SD", statistics.stdev(values), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("search", "replicate", "confirm", "all"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cache-device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.experiment_dir.resolve()
    if args.stage == "all":
        search(root)
        replicate(root)
        prepare(root, "revision_v5")
        cache(root, "revision_v5", args.cache_device)
        immutable_json(root / "revision_v5/confirmation_freeze.json", {
            "selection_sha256": checksum(root / "revision_v5/selection.json"),
            "replications_sha256": checksum(root / "revision_v5/replications.json"),
            "manifest_sha256": checksum(root / "revision_v5/confirmation/confirmation.json"),
            "episodes_sha256": checksum(root / "revision_v5/confirmation/episodes.json")})
        confirmation(root, "revision_v5")
    elif args.stage == "confirm":
        confirmation(root, "revision_v5")
    else:
        {"search": search, "replicate": replicate}[args.stage](root)
