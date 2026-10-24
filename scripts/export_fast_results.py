"""Verify and export the completed limited-budget study and inference weights."""

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from adaptive_fsvod.data import checksum, read_json, write_json
from adaptive_fsvod.evaluate import coco_metrics, load_detector


def export(root, target):
    study = root / "revision_v5"
    results = read_json(study / "confirmation_results.json")
    replicas = read_json(study / "replications.json")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(study / "confirmation_results.json", target / "confirmation_results.json")
    imported = {}
    verified = []
    weights = root / "exports/fast_adaptation_inference"
    weights.mkdir(parents=True, exist_ok=True)
    reference = None
    for row in results["rows"]:
        method, seed = row["method"], row["seed"]
        source = study / "evaluations" / f"{method}_seed{seed}" / "confirmation"
        directory = target / "confirmation" / f"{method}_seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "ground_truth.json", "predictions.json"):
            shutil.copyfile(source / name, directory / name)
            imported[(directory / name).relative_to(target).as_posix()] = checksum(source / name)
        computed = coco_metrics(read_json(source / "ground_truth.json"), read_json(source / "predictions.json"))
        for key in ("AP", "AP50", "AP75"):
            assert abs(row[key] - computed[key]) < 1e-9
        protocol = {key: row[key] for key in ("manifest_sha256", "episodes_sha256", "cache_signature",
                                             "training_budget", "training_schedule_budget", "score_floor", "nms_iou")}
        assert protocol["training_budget"] == 50 and protocol["training_schedule_budget"] == 300
        assert reference is None or protocol == reference
        reference = protocol
        origin = next(r for r in replicas if r["method"] == method and r["seed"] == seed)
        path = Path(origin["checkpoint"])
        assert checksum(path) == row["checkpoint_sha256"] == origin["checkpoint_sha256"]
        model, checkpoint = load_detector(path)
        assert checkpoint["step"] == 50
        assert checkpoint["arguments"]["stop_after"] == 50
        assert checkpoint["model_config"]["feature_centering"] == (method == "centered")
        keep = {key: value for key, value in checkpoint.items()
                if key not in ("optimizer", "scheduler", "torch_rng", "python_rng", "cuda_rng")}
        keep["artifact_type"] = "inference checkpoint; train from scratch for optimizer resume"
        output = weights / f"{method}_seed{seed}.pt"
        torch.save(keep, output)
        restored, _ = load_detector(output)
        for key, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key]), key
        verified.append({"method": method, "seed": seed, "file": output.name,
                         "bytes": output.stat().st_size, "sha256": checksum(output),
                         "source_checkpoint_sha256": row["checkpoint_sha256"],
                         "step": 50, "parameters": sum(p.numel() for p in model.parameters()),
                         "AP": row["AP"], "AP50": row["AP50"], "AP75": row["AP75"]})
        del model, restored, checkpoint, keep
    paired = [{"seed": seed,
               "fixed_AP": next(r["AP"] for r in results["rows"] if r["seed"] == seed and r["method"] == "fixed"),
               "centered_AP": next(r["AP"] for r in results["rows"] if r["seed"] == seed and r["method"] == "centered")}
              for seed in (42, 43, 44)]
    for row in paired:
        row["difference"] = row["centered_AP"] - row["fixed_AP"]
    summary = {"confirmation": results["summary"], "paired": paired,
               "mean_AP_gain": sum(r["difference"] for r in paired) / 3,
               "wins": sum(r["difference"] > 0 for r in paired), "protocol": reference,
               "checkpoints": verified,
               "note": "Small fresh set, 18 distinct frames and 6 videos; training-seed SD is not dataset uncertainty. No fully converged accuracy improvement is claimed."}
    write_json(target / "verified_summary.json", summary)
    write_json(target / "confirmation_artifact_sha256.json", imported)
    with (target / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "AP_mean", "AP_sample_SD", "AP50_mean", "AP75_mean", "actual_updates"])
        writer.writeheader()
        for method, values in results["summary"].items():
            writer.writerow({"method": method, "AP_mean": values["AP"]["mean"], "AP_sample_SD": values["AP"]["sample_std"],
                             "AP50_mean": values["AP50"]["mean"], "AP75_mean": values["AP75"]["mean"], "actual_updates": 50})
    write_json(weights / "manifest.json", summary)
    shutil.copyfile(target / "comparison.csv", weights / "comparison.csv")
    archive = weights.parent / "fast_adaptation_inference.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(weights.iterdir()):
            if path.is_file():
                bundle.write(path, path.name)
    write_json(archive.with_suffix(".manifest.json"), {"bytes": archive.stat().st_size, "sha256": checksum(archive)})
    print("Verified all six COCO results and inference checkpoint tensor round-trips", flush=True)
    print("Mean AP gain:", summary["mean_AP_gain"], "wins:", summary["wins"], "/3", flush=True)
    print("Inference archive:", archive, "bytes:", archive.stat().st_size, flush=True)
    plot(root, target, summary)


def plot(root, target, summary):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    colors = {"fixed": "#2563eb", "centered": "#059669"}
    for method, color in colors.items():
        source = "revision_v2" if method == "fixed" else "revision_v4"
        path = root / source / "runs" / f"{method}_tau080_seed42/training.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows = [r for r in rows if "validation" in r]
        axes[0, 0].plot([r["step"] for r in rows], [r["validation"]["AP"] for r in rows], marker="o", color=color, label=method.capitalize())
    axes[0, 0].axvline(50, color="#64748b", linestyle="--", alpha=.6)
    axes[0, 0].set(title="Validation learning curves (seed 42, threshold 0.8)", xlabel="Optimizer updates", ylabel="Box AP (%)")
    axes[0, 0].legend(frameon=False)
    replicas = read_json(root / "revision_v5/replications.json")
    import statistics
    for ax, split in [(axes[0, 1], "validation"), (axes[1, 0], "confirmation")]:
        means, errors = [], []
        for method in colors:
            values = [r["validation_AP"] for r in replicas if r["method"] == method]
            means.append(statistics.mean(values) if split == "validation" else summary["confirmation"][method]["AP"]["mean"])
            errors.append(statistics.stdev(values) if split == "validation" else summary["confirmation"][method]["AP"]["sample_std"])
        bars = ax.bar(["Fixed", "Centered"], means, yerr=errors, capsize=5, color=list(colors.values()), width=.5)
        ax.bar_label(bars, fmt="%.2f", padding=8)
        ax.set(title=f"{split.capitalize()} AP at equal 50-update budgets", ylabel="Box AP (%)", ylim=(0, 70))
    pairs = summary["paired"]
    axes[1, 1].axhline(0, color="#64748b", linewidth=1)
    axes[1, 1].scatter([str(r["seed"]) for r in pairs], [r["difference"] for r in pairs], color="#059669", s=65)
    for r in pairs:
        axes[1, 1].annotate(f"{r['difference']:+.2f}", (str(r["seed"]), r["difference"]), xytext=(0, 9), textcoords="offset points", ha="center")
    axes[1, 1].set(title="Fresh confirmation: paired gains", xlabel="Training seed", ylabel="Centered minus fixed (AP points)", ylim=(-.7, 2.1), xlim=(-.5, 2.5))
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    figure.savefig(target / "fast_adaptation.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/revision_v5"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    export(args.experiment_dir.resolve(), args.output)
