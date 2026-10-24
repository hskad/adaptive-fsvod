"""Run the predeclared validation search, replications, or fresh confirmation."""

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

import torch

from adaptive_fsvod.cache import CachedEpisodes
from adaptive_fsvod.data import VideoManifest, checksum, read_json, validate_splits, write_json
from adaptive_fsvod.evaluate import cache_signature, evaluate, load_detector
from adaptive_fsvod.model import ModelConfig
from adaptive_fsvod.train import parser as train_parser, train
from prepare_confirmation import prior_confirmation_path


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing to change existing protocol/configuration: {path}")
    else:
        write_json(path, value)


def measure(checkpoint_path, manifest_path, cache_path, episodes_path, output):
    cached = CachedEpisodes(VideoManifest(manifest_path), cache_path)
    model, checkpoint = load_detector(checkpoint_path, "cpu")
    if checkpoint["cache_signature"] != cache_signature(cached.metadata):
        raise ValueError("Encoder settings differ")
    episodes = read_json(episodes_path)
    metrics, truth, predictions = evaluate(model, cached, episodes, "cpu")
    metrics.update({"checkpoint_sha256": checksum(checkpoint_path), "training_step": checkpoint["step"],
                    "training_seed": checkpoint["arguments"]["seed"],
                    "training_budget": checkpoint["arguments"].get("stop_after") or checkpoint["arguments"]["steps"],
                    "training_schedule_budget": checkpoint["arguments"]["steps"],
                    "training_git": checkpoint["environment"]["git"],
                    "model_config": checkpoint["model_config"],
                    "cache_signature": checkpoint["cache_signature"],
                    "manifest_sha256": checksum(manifest_path), "episodes_sha256": checksum(episodes_path)})
    for name, value in [("metrics", metrics), ("ground_truth", truth), ("predictions", predictions)]:
        write_json(output / f"{name}.json", value)
    print("MEASURE", output.name, json.dumps(metrics), flush=True)
    return metrics


def training_run(root, config_path, name, seed, study="revision_v2", stop_after=None):
    output = root / study / "runs" / f"{name}_seed{seed}"
    flags = {
        "train-manifest": root / "pilot/train.json", "val-manifest": root / "pilot/val.json",
        "train-cache": root / "cache/train", "val-cache": root / "cache/val",
        "train-episodes": root / "pilot/train_episodes.json", "val-episodes": root / "pilot/val_episodes.json",
        "config": config_path, "output": output, "steps": 300, "eval-every": 50,
        "seed": seed, "device": "cpu"}
    if stop_after is not None:
        flags["stop-after"] = stop_after
    if (output / "last.pt").exists():
        flags["resume"] = output / "last.pt"
    command = [part for key, value in flags.items() for part in ("--" + key, str(value))]
    print("TRAIN", name, seed, flush=True)
    best_path = train(train_parser().parse_args(command))
    metrics = measure(best_path, root / "pilot/val.json", root / "cache/val",
                      root / "pilot/val_episodes.json", output / "validation")
    return {"name": name, "seed": seed, "config": str(config_path), "config_sha256": checksum(config_path),
            "checkpoint": str(best_path), "checkpoint_sha256": checksum(best_path), "validation_AP": metrics["AP"]}


def search(root):
    candidates = []
    config_dir = root / "revision_v2/configs"
    for method in ("fixed", "gated"):
        for threshold in (.6, .7, .8):
            name = f"{method}_tau{round(threshold * 100):03d}"
            cfg = ModelConfig(fixed_threshold=threshold, fusion_gate="confidence" if method == "gated" else "none",
                              gate_threshold=.7)
            path = config_dir / f"{name}.json"
            immutable_json(path, asdict(cfg))
            row = training_run(root, path, name, 42)
            candidates.append({**row, "method": method, "memory_threshold": threshold})
    no_memory = config_dir / "none.json"
    immutable_json(no_memory, asdict(ModelConfig(memory_mode="none")))
    control = training_run(root, no_memory, "none", 42)
    selected = {}
    for method in ("fixed", "gated"):
        selected[method] = max((r for r in candidates if r["method"] == method),
                               key=lambda r: (r["validation_AP"], r["memory_threshold"]))
    selected["none"] = control
    protocol = {"stage": "validation_selection", "search_seed": 42, "seeds_to_replicate": [43, 44],
                "selection_metric": "best validation AP", "gate_threshold_fixed": .7,
                "candidates": candidates, "selected": selected,
                "train_manifest_sha256": checksum(root / "pilot/train.json"),
                "val_manifest_sha256": checksum(root / "pilot/val.json"),
                "train_episodes_sha256": checksum(root / "pilot/train_episodes.json"),
                "val_episodes_sha256": checksum(root / "pilot/val_episodes.json")}
    immutable_json(root / "revision_v2/selection.json", protocol)
    print("FROZEN VALIDATION SELECTION", json.dumps(selected), flush=True)


def replicate(root):
    selection = read_json(root / "revision_v2/selection.json")
    rows = [dict(row, method=method) for method, row in selection["selected"].items()]
    for seed in (43, 44):
        for method, selected in selection["selected"].items():
            config = Path(selected["config"])
            if checksum(config) != selected["config_sha256"]:
                raise ValueError("Selected configuration changed")
            rows.append(dict(training_run(root, config, selected["name"], seed), method=method))
    write_json(root / "revision_v2/replications.json", rows)
    for method in ("fixed", "gated", "none"):
        values = [r["validation_AP"] for r in rows if r["method"] == method]
        print("VALIDATION", method, values, "mean", statistics.mean(values), "sample SD", statistics.stdev(values), flush=True)


def confirmation(root, study="revision_v2"):
    confirm = root / study / "confirmation"
    manifest_path = confirm / "confirmation.json"
    manifest = VideoManifest(manifest_path)
    originals = [VideoManifest(root / f"pilot/{split}.json") for split in ("train", "val", "test")]
    if study != "revision_v2":
        originals.append(VideoManifest(prior_confirmation_path(root)))
    validate_splits(originals + [manifest])
    rows = read_json(root / study / "replications.json")
    output = []
    for row in rows:
        checkpoint = Path(row["checkpoint"])
        if checksum(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint changed")
        metrics = measure(checkpoint, manifest_path, confirm / "cache", confirm / "episodes.json",
                          root / study / "evaluations" / f"{row['method']}_seed{row['seed']}" / "confirmation")
        output.append({"method": row["method"], "seed": row["seed"], **metrics})
    summary = {}
    for method in dict.fromkeys(row["method"] for row in rows):
        values = [r for r in output if r["method"] == method]
        summary[method] = {key: {"mean": statistics.mean(r[key] for r in values),
                                "sample_std": statistics.stdev(r[key] for r in values)}
                           for key in ("AP", "AP50", "AP75")}
    write_json(root / study / "confirmation_results.json", {"rows": output, "summary": summary})
    print("FRESH CONFIRMATION SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("search", "replicate", "confirm"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    {"search": search, "replicate": replicate, "confirm": confirmation}[args.stage](args.experiment_dir.resolve())
