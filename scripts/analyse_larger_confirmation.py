"""Verify larger-set outputs and quantify paired uncertainty across videos."""

import argparse
import copy
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
    hashes, bootstrap, video_metrics, category_metrics = {}, {}, {}, {}
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
        # pycocotools adds fields to its input dictionaries; keep raw records intact.
        full = coco_metrics(copy.deepcopy(truth), copy.deepcopy(predictions))
        for name in full:
            if abs(full[name] - row[name]) > 1e-9:
                raise ValueError("Stored metric does not match saved detections")
        mapping = {image["id"]: by_episode[image["episode_id"]] for image in truth["images"]}
        accumulator = ClusterCocoAP(truth, predictions, mapping, clusters)
        fast = accumulator.metrics(np.ones(len(clusters), dtype=int))
        category_metrics[key] = accumulator.metrics(np.ones(len(clusters), dtype=int), by_category=True)
        if abs(np.mean([v["AP"] for v in category_metrics[key].values()]) - full["AP"]) > 1e-9:
            raise ValueError("Category AP does not average to overall AP")
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
    per_class = []
    for index, category in enumerate(sorted(manifest.classes), 1):
        values = {method: float(np.mean([category_metrics[f"{method}_seed{seed}"][index]["AP"]
                                        for seed in (42, 43, 44)]))
                  for method in ("fixed_50", "centered_50", "fixed_300")}
        per_class.append({"class_id": category, "class_name": manifest.classes[category], **values,
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
               "per_class_AP": per_class,
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
                       ("paired.csv", paired), ("per_video.csv", per_video), ("per_class.csv", per_class)]:
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
    write_readme(target, summary)
    print("FINAL LARGER SUMMARY", summary, flush=True)


def write_readme(target, summary):
    interval = summary["video_bootstrap"]["percentile_95_interval"]
    gain = summary["primary_mean_AP_difference"]
    if interval[0] > 0:
        interpretation = "The interval is positive on this protocol. This supports a matched-budget gain on these fixed classes and support episodes."
    elif interval[1] < 0:
        interpretation = "The interval is negative on this protocol. Centering reduced AP on this larger confirmation set at the matched budget."
    else:
        interpretation = "The interval includes zero. This larger check does not establish an improvement at the matched budget."
    lines = ["# Larger confirmation of frozen feature centering", "",
             f"The primary paired effect is **{gain:+.3f} AP points** at equal 50-update budgets.",
             f"The paired, class-stratified query-video bootstrap 95% interval is **[{interval[0]:+.3f}, {interval[1]:+.3f}]**.",
             interpretation, "", "## Aggregate results", "",
             "| Method | Training budget | AP mean ± seed SD | AP50 mean | AP75 mean |",
             "|---|---:|---:|---:|---:|"]
    for method, label, budget in [("fixed_50", "Fixed", 50), ("centered_50", "Centered", 50), ("fixed_300", "Fixed reference", 300)]:
        values = summary["summary"][method]
        lines.append(f"| {label} | {budget} | {values['AP']['mean']:.3f} ± {values['AP']['sample_std']:.3f} | {values['AP50']['mean']:.3f} | {values['AP75']['mean']:.3f} |")
    lines += ["", "Three training seeds: 42, 43, 44. Seed SD is not uncertainty over new data.",
              "The reference has a 300-update training budget and uses validation-selected checkpoints. The primary pair has exactly 50 updates and a shared 300-update learning-rate schedule.",
              "", "| Seed | Fixed 50 AP | Centered 50 AP | Difference | Fixed 300 reference AP |", "|---|---:|---:|---:|---:|"]
    for row in summary["paired"]:
        lines.append(f"| {row['seed']} | {row['fixed_50_AP']:.3f} | {row['centered_50_AP']:.3f} | {row['AP_difference']:+.3f} | {row['fixed_300_AP']:.3f} |")
    lines += ["", "## Class-level AP", "",
              "| Class | Fixed 50 | Centered 50 | Difference |", "|---|---:|---:|---:|"]
    for row in summary["per_class_AP"]:
        lines.append(f"| {row['class_name']} | {row['fixed_50']:.3f} | {row['centered_50']:.3f} | {row['difference']:+.3f} |")
    lines += ["", "## Fixed protocol and limits", "",
              f"Six fresh classes: {', '.join(summary['classes'].values())}. There are 24 distinct videos, 192 distinct frames, 120 balanced two-way one-shot episodes, and {summary['episode_frame_instances']} episode-frame instances. Repeated frames are not independent samples.",
              "", "Every class and video is disjoint from the original splits and both earlier fresh confirmations. All nine checkpoints, thresholds, and the training-only feature center were fixed before scoring. Every query video is tested against every other class. Its support crops come from other videos. A video can supply support to another query episode.",
              "", "The bootstrap uses 1,000 shared draws with seed 20261030. Each draw resamples four query videos within each of the six fixed classes, retaining all related frames and episodes. The interval is conditional on the fixed class identities, support episodes, and trained seed pairs. It excludes uncertainty from new classes, new support examples, and model development. It is not an unconditional significance claim.",
              "", "All methods use one shared frozen OWLv2 SDPA feature cache. The optional kernel preserves the attention equation and differs slightly in floating-point rounding from native eager attention. The pre-score kernel check and complete cache hashes are in `protocol/`. The historical eager cache and all earlier results remain intact.",
              "", "## Records and reproduction", "",
              "- `protocol/`: pre-score selection, manifest, episodes, support pool, model hashes, cache hashes, and extraction checks.",
              "- `evaluations/`: nine metric files with exact predictions and ground truth. All AP/AP50/AP75 values were recomputed from these records.",
              "- `verified_summary.json`, `comparison.csv`, and `paired.csv`: aggregate and paired results.",
              "- `bootstrap_draws.json`: actual query-video multiplicities, per-model AP draws, and paired effects.",
              "- `per_video.csv`: diagnostic single-query-video AP; its average is not aggregate COCO AP.",
              "- `per_class.csv`: native COCO AP by class, averaged across training seeds.",
              "- `evaluation_environment.json`, `cache_metadata/`, and `download_manifest.json`: software, encoder, image CRCs, dimensions, and hashes.",
              "- `source_artifact_sha256.json`: hashes of the copied raw metric and detection records.",
              "", "![Larger confirmation](larger_confirmation.png)", "",
              "See the [pre-score plan](../../docs/LARGER_CONFIRMATION_PLAN.md) and [executable runbook](../../docs/RUNBOOK.md#larger-confirmation-without-further-training). The source head training records are in [revision_v5](../revision_v5/README.md) and [revision_v2](../revision_v2/README.md). Weights are retained locally; a public weight download is still required for final submission. No source RGB images or model tensors are stored in Git.",
              "", "Keep the earlier +0.612 AP small-set result and the failed full-budget centering and memory-rule experiments. This larger check does not replace those records. No model was changed after reading these scores.", ""]
    (target / "README.md").write_text("\n".join(lines), encoding="utf-8")


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
    axes[1].axvline(0, color="#334155", linestyle="--", label="Zero difference")
    for index, value in enumerate(summary["video_bootstrap"]["percentile_95_interval"]):
        axes[1].axvline(value, color="#b45309", linestyle=":", label="95% interval" if index == 0 else "_nolegend_")
    axes[1].legend(frameon=False, fontsize=8)
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
