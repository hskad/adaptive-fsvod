"""Verify larger-set outputs and quantify paired uncertainty across videos."""

import argparse
import csv
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_fsvod.bootstrap import ClusterCocoAP, stratified_video_draws
from adaptive_fsvod.data import VideoManifest, checksum, read_json, write_json
from adaptive_fsvod.evaluate import coco_metrics
from run_larger_confirmation import STUDY, cache_directory, verify


def analyse(root, target):
    verify(root)
    source = root / STUDY
    target.mkdir(parents=True, exist_ok=True)
    results = read_json(source / "results.json")
    manifest = VideoManifest(source / "confirmation.json")
    episodes = read_json(source / "episodes.json")
    by_episode = {e["id"]: e["video_id"] for e in episodes["episodes"]}
    video_classes = {vid: next(iter({str(o["category_id"]) for f in v["frames"] for o in f["objects"]}))
                     for vid, v in manifest.videos.items()}
    clusters, draws = stratified_video_draws(video_classes, count=1000, seed=20261030)
    hashes, bootstrap, video_metrics = {}, {}, {}
    common_truth, reference = None, None
    started = time.perf_counter()
    for row in results["rows"]:
        key = f"{row['method']}_seed{row['seed']}"
        directory = source / "evaluations" / key
        truth = read_json(directory / "ground_truth.json")
        predictions = read_json(directory / "predictions.json")
        if common_truth is not None and truth != common_truth:
            raise ValueError("Models were not evaluated against identical ground truth")
        common_truth = truth
        protocol = {k: row[k] for k in ("manifest_sha256", "episodes_sha256", "cache_signature",
                                       "score_floor", "nms_iou")}
        if reference is not None and protocol != reference:
            raise ValueError("Evaluation settings differ between models")
        reference = protocol
        full = coco_metrics(truth, predictions)
        for name in full:
            if abs(full[name] - row[name]) > 1e-9:
                raise ValueError("Stored metric does not match saved detections")
        mapping = {image["id"]: by_episode[image["episode_id"]] for image in truth["images"]}
        accumulator = ClusterCocoAP(truth, predictions, mapping, clusters)
        fast = accumulator.metrics(np.ones(len(clusters), dtype=int))
        for name in full:
            if abs(full[name] - fast[name]) > 1e-9:
                raise ValueError("Cached COCO accumulation differs from standard COCO")
        bootstrap[key] = []
        for i, weights in enumerate(draws):
            bootstrap[key].append(accumulator.metrics(weights)["AP"])
            if (i + 1) % 250 == 0:
                print("BOOTSTRAP", key, i + 1, "/", len(draws), flush=True)
        video_metrics[key] = {}
        for i, vid in enumerate(clusters):
            weights = np.zeros(len(clusters), dtype=int)
            weights[i] = 1
            video_metrics[key][vid] = accumulator.metrics(weights)
        output = target / "evaluations" / key
        output.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "ground_truth.json", "predictions.json"):
            shutil.copyfile(directory / name, output / name)
            hashes[(output / name).relative_to(target).as_posix()] = checksum(directory / name)
        print("VERIFIED", key, full, "elapsed seconds", round(time.perf_counter() - started, 1), flush=True)
    paired, effects = [], []
    for seed in (42, 43, 44):
        values = {r["method"]: r["AP"] for r in results["rows"] if r["seed"] == seed}
        paired.append({"seed": seed, "fixed_50_AP": values["fixed_50"],
                       "centered_50_AP": values["centered_50"], "fixed_300_AP": values["fixed_300"],
                       "AP_difference": values["centered_50"] - values["fixed_50"]})
        effects.append(np.array(bootstrap[f"centered_50_seed{seed}"]) - np.array(bootstrap[f"fixed_50_seed{seed}"]))
    mean_effects = np.stack(effects).mean(0)
    interval = np.quantile(mean_effects, [.025, .975]).tolist()
    mean_effect = float(np.mean([r["AP_difference"] for r in paired]))
    per_video = []
    for vid in clusters:
        values = {method: float(np.mean([video_metrics[f"{method}_seed{seed}"][vid]["AP"]
                                        for seed in (42, 43, 44)]))
                  for method in ("fixed_50", "centered_50", "fixed_300")}
        per_video.append({"video_id": vid, "class_id": video_classes[vid],
                          "class_name": manifest.classes[video_classes[vid]], **values,
                          "difference": values["centered_50"] - values["fixed_50"]})
    summary = {"summary": results["summary"], "paired": paired,
               "primary_mean_AP_difference": mean_effect,
               "paired_seed_wins": sum(r["AP_difference"] > 0 for r in paired),
               "video_bootstrap": {"draws": 1000, "seed": 20261030, "percentile_95_interval": interval,
                                   "unit": "query video; stratified within the six fixed classes",
                                   "includes_zero": interval[0] <= 0 <= interval[1],
                                   "conditional_on": "fixed class identities, fixed support episodes, and the three trained seed pairs",
                                   "does_not_cover": "unseen-class, new-support, training-development or model-selection uncertainty"},
               "classes": manifest.classes, "distinct_videos": len(clusters),
               "distinct_frames": sum(len(v["frames"]) for v in manifest.videos.values()),
               "episodes": len(episodes["episodes"]), "episode_frame_instances": results["rows"][0]["frames"],
               "per_video_AP": per_video,
               "per_video_note": "Diagnostic single-query-video AP; its mean is not aggregate COCO AP.",
               "protocol": reference}
    write_json(target / "verified_summary.json", summary)
    write_json(target / "bootstrap_draws.json", {"clusters": clusters, "multiplicities": draws.tolist(),
                                                "AP_by_model": bootstrap, "paired_mean_effect": mean_effects.tolist()})
    write_json(target / "source_artifact_sha256.json", hashes)
    shutil.copyfile(source / "results.json", target / "results.json")
    for name, rows in [("comparison.csv", [dict(method=method, AP=values["AP"]["mean"],
                                               AP_sample_SD=values["AP"]["sample_std"],
                                               AP50=values["AP50"]["mean"], AP75=values["AP75"]["mean"])
                                            for method, values in results["summary"].items()]),
                       ("paired.csv", paired), ("per_video.csv", per_video)]:
        with (target / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name in ("metadata.json", "support_pool.json"):
        output = target / "cache_metadata" / name
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_directory(root) / name, output)
    shutil.copyfile(source / "extraction_settings.json", target / "extraction_settings.json")
    shutil.copyfile(source / "evaluation_environment.json", target / "evaluation_environment.json")
    shutil.copyfile(source / "images/download_manifest.json", target / "download_manifest.json")
    plot(target, summary, mean_effects)
    print("FINAL LARGER SUMMARY", summary, flush=True)


def plot(target, summary, effects):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    methods = ["fixed_50", "centered_50", "fixed_300"]
    means = [summary["summary"][m]["AP"]["mean"] for m in methods]
    errors = [summary["summary"][m]["AP"]["sample_std"] for m in methods]
    bars = axes[0].bar(["Fixed\n50 updates", "Centered\n50 updates", "Fixed\n300 budget"], means,
                       yerr=errors, capsize=4, color=["#2563eb", "#059669", "#64748b"], width=.65)
    axes[0].bar_label(bars, fmt="%.2f", padding=7)
    axes[0].set(title="Larger confirmation: aggregate box AP", ylabel="AP (%)", ylim=(0, max(means) + 13))
    axes[0].text(.02, .98, "Error bars: training-seed sample SD", transform=axes[0].transAxes, va="top", fontsize=8)
    axes[1].hist(effects, bins=30, color="#059669", alpha=.8)
    axes[1].axvline(0, color="#334155", linestyle="--")
    for value in summary["video_bootstrap"]["percentile_95_interval"]:
        axes[1].axvline(value, color="#b45309", linestyle=":")
    axes[1].set(title="Paired video bootstrap (1,000 draws)", xlabel="Centered - fixed at 50 updates (AP points)", ylabel="Draw count")
    rows = summary["per_video_AP"]
    for category, name in summary["classes"].items():
        values = [r["difference"] for r in rows if r["class_id"] == category]
        axes[2].scatter(values, [name] * len(values), s=35, alpha=.75)
    axes[2].axvline(0, color="#334155", linestyle="--")
    axes[2].set(title="Individual query-video AP differences", xlabel="Centered - fixed at 50 updates (AP points)")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=.15)
        axis.set_axisbelow(True)
    figure.savefig(target / "larger_confirmation.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/larger_confirmation"))
    args = parser.parse_args()
    analyse(args.experiment_dir.resolve(), args.output)
