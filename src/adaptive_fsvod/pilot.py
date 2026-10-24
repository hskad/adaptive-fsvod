"""Collect a pilot, or resume its fixed training matrix before collection.

This runner does not change hyperparameters or select methods with test AP.
Input manifests, feature caches, and episode files must already exist.
"""

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
import zipfile
from pathlib import Path

from .compare import compare
from .data import read_json, write_json


METHODS = ("fixed", "adaptive")
METRICS = ("AP", "AP50", "AP75")


def execute(root, phase, module, flags):
    write_json(root / "runs/automation_status.json", {"phase": phase, "status": "running"})
    print(phase, flush=True)
    command = [sys.executable, "-u", "-m", "adaptive_fsvod." + module]
    for key, value in flags.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    subprocess.run(command, check=True)


def run_matrix(args):
    root = args.experiment_dir
    data, cache = root / "pilot", root / "cache"
    for seed in args.seeds:
        for method in METHODS:
            directory = root / "runs" / f"{method}_seed{seed}"
            flags = dict(train_manifest=data / "train.json", val_manifest=data / "val.json",
                         train_cache=cache / "train", val_cache=cache / "val",
                         train_episodes=data / "train_episodes.json",
                         val_episodes=data / "val_episodes.json",
                         config=args.config_dir / f"{method}.json", output=directory,
                         steps=args.steps, eval_every=args.eval_every, seed=seed, device=args.device)
            if (directory / "last.pt").exists():
                # train.py verifies resume provenance and the original total budget.
                # Completed runs have no remaining optimizer steps.
                flags["resume"] = directory / "last.pt"
            execute(root, f"Train/resume {method}, seed {seed}", "train", flags)
            execute(root, f"Evaluate {method}, seed {seed}", "evaluate", dict(
                manifest=data / "test.json", cache=cache / "test",
                episodes=data / "test_episodes.json", checkpoint=directory / "best.pt",
                output=directory / "test", device=args.device))


def summarize(root, seeds):
    rows, reference, configs = [], None, {}
    shared_keys = ("manifest_sha256", "episodes_sha256", "training_budget", "cache_signature",
                   "score_floor", "nms_iou", "synthetic")
    train_keys = ("train_manifest_sha256", "val_manifest_sha256",
                  "train_episodes_sha256", "val_episodes_sha256")
    for seed in seeds:
        comparison = root / "runs" / f"comparison_seed{seed}"
        compare(root / "runs" / f"fixed_seed{seed}/test/metrics.json",
                root / "runs" / f"adaptive_seed{seed}/test/metrics.json", comparison)
        for method in METHODS:
            directory = root / "runs" / f"{method}_seed{seed}"
            result = read_json(directory / "test/metrics.json")
            provenance = read_json(directory / "config.json")
            if result["training_seed"] != seed or provenance["arguments"]["seed"] != seed:
                raise ValueError(f"Training seed mismatch: {directory}")
            protocol = {**{key: result[key] for key in shared_keys},
                        **{key: provenance[key] for key in train_keys}}
            if reference is not None and reference != protocol:
                raise ValueError("Cannot pool runs with different data or evaluation protocols")
            reference = protocol
            if method in configs and configs[method] != result["model_config"]:
                raise ValueError(f"Model settings changed across seeds: {method}")
            configs[method] = result["model_config"]
            records = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines()]
            chosen = next(r for r in records if r["step"] == result["training_step"] and "validation" in r)
            validation = chosen["validation"]
            row = {"method": method, "seed": seed, "selected_step": result["training_step"],
                   **{key: result[key] for key in METRICS},
                   "validation_AP": validation["AP"],
                   "validation_mean_threshold": validation["mean_threshold"],
                   "validation_mean_memory_tokens": validation["mean_memory_tokens"],
                   "test_mean_memory_tokens": result["mean_memory_tokens"],
                   "last_session_elapsed_seconds": records[-1]["elapsed_seconds"],
                   "synthetic": result["synthetic"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    aggregates = {}
    for method in METHODS:
        values = [r for r in rows if r["method"] == method]
        aggregates[method] = {
            metric: {"mean": statistics.mean(r[metric] for r in values),
                     "sample_std": statistics.stdev(r[metric] for r in values) if len(values) > 1 else None}
            for metric in METRICS}
    differences = {
        str(seed): next(r["AP"] for r in rows if r["seed"] == seed and r["method"] == "adaptive")
        - next(r["AP"] for r in rows if r["seed"] == seed and r["method"] == "fixed")
        for seed in seeds}
    exports = root / "exports"
    exports.mkdir(exist_ok=True)
    summary = {"seeds": seeds, "rows": rows, "aggregates": aggregates,
               "paired_AP_differences": differences, "protocol": reference,
               "note": "AP differences are points. Sample SD is over training seeds, not new test sets."}
    write_json(exports / "pilot_summary.json", summary)
    with (exports / "pilot_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("SUMMARY", json.dumps(aggregates), flush=True)
    print("PAIRED AP DIFFERENCES", json.dumps(differences), flush=True)
    return summary


def bundle(root, seeds):
    exports = root / "exports"
    archive = exports / ("pilot_seeds_" + "_".join(map(str, seeds)) + ".zip")
    folders = [root / "runs/overfit_check"]
    for seed in seeds:
        folders.extend(root / "runs" / f"{method}_seed{seed}" for method in METHODS)
        folders.append(root / "runs" / f"comparison_seed{seed}")
    paths = {p for folder in folders for p in folder.rglob("*") if p.is_file()}
    paths.update((root / "pilot").glob("*.json"))
    paths.update(exports / name for name in ("pilot_summary.json", "pilot_summary.csv"))
    for split in ("train", "val", "test"):
        paths.update(root / "cache" / split / name for name in ("metadata.json", "support_pool.json"))
    # Keep source images, encoder caches, and existing archives out of the export.
    paths = {p for p in paths if p.suffix in {".json", ".jsonl", ".pt", ".csv", ".png"}}
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for path in sorted(paths):
            if not path.is_file():
                raise FileNotFoundError(path)
            output.write(path, path.relative_to(root))
    temporary.replace(archive)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    manifest = {"path": str(archive), "bytes": archive.stat().st_size,
                "sha256": digest.hexdigest(), "files": len(paths)}
    write_json(archive.with_suffix(".manifest.json"), manifest)
    print("ARCHIVE", json.dumps(manifest), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--action", choices=("collect", "run"), default="collect",
                        help="collect reads completed runs; run trains/resumes and evaluates first")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config-dir", type=Path, default=Path(__file__).resolve().parents[2] / "configs")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds must be unique")
    args.experiment_dir = args.experiment_dir.resolve()
    status = args.experiment_dir / "runs/automation_status.json"
    try:
        if args.action == "run":
            run_matrix(args)
        write_json(status, {"phase": "collect", "status": "running"})
        summarize(args.experiment_dir, args.seeds)
        write_json(status, {"phase": "archive", "status": "running"})
        bundle(args.experiment_dir, args.seeds)
        write_json(status, {"status": "complete", "action": args.action, "seeds": args.seeds})
        print("COMPLETE", flush=True)
    except Exception as error:
        previous = read_json(status) if status.exists() else {}
        write_json(status, {**previous, "status": "failed", "error": str(error)})
        raise


if __name__ == "__main__":
    main()
