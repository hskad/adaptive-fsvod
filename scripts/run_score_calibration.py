"""Run the bounded, validation-first objectness score study without training."""

import argparse
import copy
import csv
import shutil
import statistics
from pathlib import Path

import torch

from adaptive_fsvod.cache import CachedEpisodes
from adaptive_fsvod.data import VideoManifest, checksum, read_json, write_json
from adaptive_fsvod.evaluate import cache_signature, coco_metrics, evaluate, load_detector
from adaptive_fsvod.train import environment
from prepare_confirmation import immutable


STUDY = "score_calibration"


def freeze(root, target):
    directory = root / STUDY
    inputs = {"manifest": root / "pilot/val.json", "episodes": root / "pilot/val_episodes.json",
              "cache_metadata": root / "cache/val/metadata.json", "support_pool": root / "cache/val/support_pool.json"}
    models = []
    for row in read_json(root / "revision_v2/replications.json"):
        if row["method"] != "fixed":
            continue
        if checksum(row["checkpoint"]) != row["checkpoint_sha256"] or checksum(row["config"]) != row["config_sha256"]:
            raise ValueError("Reference checkpoint or model configuration changed")
        saved = torch.load(row["checkpoint"], map_location="cpu", weights_only=True)
        if saved["arguments"]["steps"] != 300 or saved["model_config"].get("feature_centering", False):
            raise ValueError("Expected the uncentered 300-update fixed reference")
        models.append(copy.deepcopy(row))
    if sorted(r["seed"] for r in models) != [42, 43, 44]:
        raise ValueError("Expected all three reference seeds")
    tensors = {p.relative_to(root / "cache/val").as_posix(): checksum(p)
               for p in sorted((root / "cache/val").rglob("*.pt"))}
    protocol = {"study": STUDY, "search_seed": 42, "beta_candidates": [0., .5, 1.],
                "selection": "maximum seed-42 validation AP; ties prefer smaller beta",
                "replication_seeds": [43, 44], "minimum_mean_gain_AP": 1.,
                "require_positive_gain_each_seed": True, "score_floor": .001, "nms_iou": .5,
                "max_detections": 100, "models": models,
                "inputs": {name: {"path": str(path), "sha256": checksum(path)} for name, path in inputs.items()},
                "validation_cache_tensors_sha256": tensors,
                "no_training": True,
                "heldout_policy": "No inspected test data may select beta. New classes are used only after the validation gate passes."}
    immutable(directory / "protocol.json", protocol)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directory / "protocol.json", target / "protocol.json")
    print("FROZEN", len(models), "models and", len(tensors), "validation feature tensors", flush=True)


def verify(root):
    protocol = read_json(root / STUDY / "protocol.json")
    for record in protocol["inputs"].values():
        if checksum(record["path"]) != record["sha256"]:
            raise ValueError("Frozen validation input changed")
    for name, digest in protocol["validation_cache_tensors_sha256"].items():
        if checksum(root / "cache/val" / name) != digest:
            raise ValueError("Frozen validation feature tensor changed")
    for row in protocol["models"]:
        if checksum(row["checkpoint"]) != row["checkpoint_sha256"] or checksum(row["config"]) != row["config_sha256"]:
            raise ValueError("Frozen reference changed")
    return protocol


def measure(root, row, beta, execution):
    output = root / STUDY / "validation" / f"beta{round(beta * 100):03d}_seed{row['seed']}"
    inputs = root / "pilot"
    cached = CachedEpisodes(VideoManifest(inputs / "val.json"), root / "cache/val")
    model, checkpoint = load_detector(row["checkpoint"])
    if checkpoint["cache_signature"] != cache_signature(cached.metadata):
        raise ValueError("Encoder signatures differ")
    episodes_path = inputs / "val_episodes.json"
    metrics, truth, predictions = evaluate(model, cached, read_json(episodes_path), "cpu", objectness_beta=beta)
    if beta == 0 and abs(metrics["AP"] - row["validation_AP"]) > 1e-9:
        raise ValueError("Zero-beta evaluation differs from the archived baseline")
    metrics.update({"training_seed": row["seed"], "checkpoint_sha256": row["checkpoint_sha256"],
                    "training_budget": 300, "checkpoint_step": checkpoint["step"],
                    "model_config": checkpoint["model_config"], "cache_signature": checkpoint["cache_signature"],
                    "manifest_sha256": checksum(inputs / "val.json"), "episodes_sha256": checksum(episodes_path),
                    "study_protocol_sha256": checksum(root / STUDY / "protocol.json"),
                    "evaluation_environment": execution})
    for name, value in (("metrics", metrics), ("ground_truth", truth), ("predictions", predictions)):
        immutable(output / f"{name}.json", value)
    print("VALIDATION", "seed", row["seed"], "beta", beta,
          {key: round(metrics[key], 4) for key in ("AP", "AP50", "AP75")}, flush=True)
    return {"seed": row["seed"], "beta": beta, "output": str(output), **metrics}


def validate(root):
    protocol = verify(root)
    execution = {"environment": environment(), "threads": torch.get_num_threads()}
    records = []
    reference = next(r for r in protocol["models"] if r["seed"] == 42)
    for beta in protocol["beta_candidates"]:
        records.append(measure(root, reference, beta, execution))
    selected = max(records, key=lambda r: (r["AP"], -r["beta"]))
    chosen = selected["beta"]
    immutable(root / STUDY / "selection.json", {"selected_beta": chosen, "search_seed": 42,
              "selected_validation_AP": selected["AP"],
              "search_records": [{"beta": r["beta"], "AP": r["AP"]} for r in records],
              "checkpoint_sha256": reference["checkpoint_sha256"]})
    pairs = []
    if chosen:
        for seed in (43, 44):
            row = next(r for r in protocol["models"] if r["seed"] == seed)
            for beta in (0., chosen):
                records.append(measure(root, row, beta, execution))
        for seed in (42, 43, 44):
            base = next(r["AP"] for r in records if r["seed"] == seed and r["beta"] == 0)
            modified = next(r["AP"] for r in records if r["seed"] == seed and r["beta"] == chosen)
            pairs.append({"seed": seed, "baseline_AP": base, "calibrated_AP": modified,
                          "difference": modified - base})
    mean_gain = statistics.mean(p["difference"] for p in pairs) if pairs else 0.
    passed = bool(chosen and mean_gain >= protocol["minimum_mean_gain_AP"] and all(p["difference"] > 0 for p in pairs))
    result = {"selected_beta": chosen, "rows": records, "paired": pairs,
              "mean_validation_AP_gain": mean_gain, "validation_gate_passed": passed,
              "decision": "Freeze and prepare an unused-class confirmation" if passed else "Stop this family; do not extract or score new test data",
              "note": "Validation-selected evidence only; no new test improvement is claimed."}
    immutable(root / STUDY / "validation_results.json", result)
    print("VALIDATION DECISION", {k: v for k, v in result.items() if k != "rows"}, flush=True)


def export(root, target):
    verify(root)
    directory = root / STUDY
    result = read_json(directory / "validation_results.json")
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for row in result["rows"]:
        source = Path(row["output"])
        recomputed = coco_metrics(read_json(source / "ground_truth.json"), read_json(source / "predictions.json"))
        for key in ("AP", "AP50", "AP75"):
            if abs(recomputed[key] - row[key]) > 1e-9:
                raise ValueError("Recorded validation metric differs from saved predictions")
        destination = target / "validation" / source.name
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "ground_truth.json", "predictions.json"):
            shutil.copyfile(source / name, destination / name)
            hashes[(destination / name).relative_to(target).as_posix()] = checksum(source / name)
    for name in ("protocol.json", "selection.json", "validation_results.json"):
        shutil.copyfile(directory / name, target / name)
    write_json(target / "source_artifact_sha256.json", hashes)
    with (target / "validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["seed", "beta", "AP", "AP50", "AP75"])
        writer.writeheader()
        writer.writerows({k: row[k] for k in writer.fieldnames} for row in result["rows"])
    print("Verified and exported", len(result["rows"]), "validation evaluations", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("freeze", "validate", "export"), required=True)
    parser.add_argument("--output", type=Path, default=Path("results/score_calibration"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.experiment_dir.resolve()
    if args.stage == "freeze":
        freeze(root, args.output)
    elif args.stage == "validate":
        validate(root)
    else:
        export(root, args.output)
