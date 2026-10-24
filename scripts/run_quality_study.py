"""Bounded, predeclared objectness-weighted memory study; see QUALITY_PLAN.md."""

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from adaptive_fsvod.data import checksum, read_json, write_json
from adaptive_fsvod.model import ModelConfig
from run_revision import confirmation, immutable_json, training_run


def search(root):
    previous = read_json(root / "revision_v2/selection.json")
    candidates = []
    for threshold in (.6, .7, .8):
        name = f"weighted_tau{round(threshold * 100):03d}"
        path = root / "revision_v3/configs" / f"{name}.json"
        immutable_json(path, asdict(ModelConfig(fixed_threshold=threshold, memory_weighting="objectness")))
        row = training_run(root, path, name, 42, "revision_v3")
        candidates.append({**row, "method": "weighted", "memory_threshold": threshold})
    chosen = max(candidates, key=lambda r: (r["validation_AP"], r["memory_threshold"]))
    # Reuse the already-completed, same-budget CPU baseline search and controls.
    # Their default weighting is 'none'; weights and forward behavior are unchanged.
    selected = {"fixed": previous["selected"]["fixed"], "weighted": chosen, "none": previous["selected"]["none"]}
    protocol = {"stage": "validation_selection", "search_seed": 42, "seeds_to_replicate": [43, 44],
                "selection_metric": "best validation AP", "candidates": candidates, "selected": selected,
                "reused_baseline_search": [r for r in previous["candidates"] if r["method"] == "fixed"],
                "reference_selection_sha256": checksum(root / "revision_v2/selection.json"),
                **{key: previous[key] for key in ("train_manifest_sha256", "val_manifest_sha256",
                                                  "train_episodes_sha256", "val_episodes_sha256")}}
    immutable_json(root / "revision_v3/selection.json", protocol)
    print("FROZEN QUALITY SELECTION", json.dumps(selected), flush=True)


def replicate(root):
    selection = read_json(root / "revision_v3/selection.json")
    old = read_json(root / "revision_v2/replications.json")
    rows = [r for r in old if r["method"] in ("fixed", "none")]
    chosen = selection["selected"]["weighted"]
    rows.append(dict(chosen, method="weighted"))
    config = Path(chosen["config"])
    if checksum(config) != chosen["config_sha256"]:
        raise ValueError("Selected configuration changed")
    for seed in (43, 44):
        rows.append(dict(training_run(root, config, chosen["name"], seed, "revision_v3"), method="weighted"))
    write_json(root / "revision_v3/replications.json", rows)
    for method in ("fixed", "weighted", "none"):
        values = [r["validation_AP"] for r in rows if r["method"] == method]
        print("VALIDATION", method, values, "mean", statistics.mean(values), "sample SD", statistics.stdev(values), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("search", "replicate", "confirm"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.experiment_dir.resolve()
    if args.stage == "confirm":
        confirmation(root, "revision_v3")
    else:
        {"search": search, "replicate": replicate}[args.stage](root)
