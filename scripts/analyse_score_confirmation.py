"""Verify objectness-score results and the predeclared query-video uncertainty."""

import argparse
import copy
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from adaptive_fsvod.bootstrap import ClusterCocoAP, stratified_video_draws
from adaptive_fsvod.data import VideoManifest, checksum, read_json, write_json
from adaptive_fsvod.evaluate import coco_metrics
from prepare_score_confirmation import verify_fresh
from run_score_calibration import STUDY


def analyse(root, target):
    protocol = verify_fresh(root)
    source = root / STUDY
    confirmation = source / "confirmation"
    result = read_json(source / "confirmation_results.json")
    manifest = VideoManifest(confirmation / "confirmation.json")
    episodes = read_json(confirmation / "episodes.json")
    episode_video = {e["id"]: e["video_id"] for e in episodes["episodes"]}
    video_classes = {vid: next(iter({str(o["category_id"]) for f in video["frames"] for o in f["objects"]}))
                     for vid, video in manifest.videos.items()}
    clusters, draws = stratified_video_draws(video_classes, count=1000, seed=20261102)
    target.mkdir(parents=True, exist_ok=True)
    hashes, bootstrap, categories, per_video = {}, {}, {}, {}
    common_truth = None
    for row in result["rows"]:
        key = f"{row['method']}_seed{row['seed']}"
        directory = source / "evaluations" / key
        truth = read_json(directory / "ground_truth.json")
        predictions = read_json(directory / "predictions.json")
        if common_truth is not None and truth != common_truth:
            raise ValueError("Models used different ground truth")
        common_truth = truth
        computed = coco_metrics(copy.deepcopy(truth), copy.deepcopy(predictions))
        mapping = {image["id"]: episode_video[image["episode_id"]] for image in truth["images"]}
        accumulator = ClusterCocoAP(truth, predictions, mapping, clusters)
        accumulated = accumulator.metrics(np.ones(len(clusters), dtype=int))
        for name in computed:
            if abs(computed[name] - row[name]) > 1e-9 or abs(computed[name] - accumulated[name]) > 1e-9:
                raise ValueError("Stored or cached AP differs from standard COCO evaluation")
        categories[key] = accumulator.metrics(np.ones(len(clusters), dtype=int), by_category=True)
        bootstrap[key] = []
        for index, weights in enumerate(draws):
            bootstrap[key].append(accumulator.metrics(weights)["AP"])
            if (index + 1) % 250 == 0:
                print("BOOTSTRAP", key, index + 1, "/1000", flush=True)
        per_video[key] = {}
        for index, video in enumerate(clusters):
            weights = np.zeros(len(clusters), dtype=int)
            weights[index] = 1
            per_video[key][video] = accumulator.metrics(weights)["AP"]
        destination = target / "evaluations" / key
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("metrics.json", "ground_truth.json", "predictions.json"):
            shutil.copyfile(directory / name, destination / name)
            hashes[(destination / name).relative_to(target).as_posix()] = checksum(directory / name)
        print("VERIFIED", key, computed, flush=True)
    paired = []
    differences = []
    for seed in (42, 43, 44):
        base = next(r for r in result["rows"] if r["seed"] == seed and r["method"] == "baseline")
        modified = next(r for r in result["rows"] if r["seed"] == seed and r["method"] == "objectness")
        paired.append({"seed": seed, "baseline_AP": base["AP"], "objectness_AP": modified["AP"],
                       "difference": modified["AP"] - base["AP"]})
        differences.append(np.array(bootstrap[f"objectness_seed{seed}"]) - np.array(bootstrap[f"baseline_seed{seed}"]))
    effects = np.stack(differences).mean(0)
    interval = np.quantile(effects, [.025, .975]).tolist()
    class_rows, video_rows = [], []
    for index, category in enumerate(sorted(manifest.classes), 1):
        values = {method: float(np.mean([categories[f"{method}_seed{s}"][index]["AP"] for s in (42, 43, 44)]))
                  for method in ("baseline", "objectness")}
        class_rows.append({"class_id": category, "class_name": manifest.classes[category], **values,
                           "difference": values["objectness"] - values["baseline"]})
    for video in clusters:
        values = {method: float(np.mean([per_video[f"{method}_seed{s}"][video] for s in (42, 43, 44)]))
                  for method in ("baseline", "objectness")}
        video_rows.append({"video_id": video, "class_name": manifest.classes[video_classes[video]], **values,
                           "difference": values["objectness"] - values["baseline"]})
    summary = {"summary": result["summary"], "paired": paired,
               "mean_AP_gain": float(np.mean([r["difference"] for r in paired])),
               "paired_seed_wins": sum(r["difference"] > 0 for r in paired),
               "selected_beta": protocol["selected_beta"], "training_budget": 300, "additional_training_updates": 0,
               "distinct_videos": len(clusters), "distinct_frames": protocol["frames"],
               "episodes": len(episodes["episodes"]), "episode_frame_instances": result["rows"][0]["frames"],
               "classes": manifest.classes, "per_class_AP": class_rows, "per_video_AP": video_rows,
               "video_bootstrap": {"draws": 1000, "seed": 20261102, "percentile_95_interval": interval,
                                   "includes_zero": interval[0] <= 0 <= interval[1],
                                   "conditional_on": "fixed six classes, fixed support episodes, and the three trained seed pairs",
                                   "does_not_cover": "new class identities, changed support examples, or model-development uncertainty"},
               "scope": "Reduced FSYTV adaptation; final-score extension using the same frozen heads. No original-paper benchmark reproduction or global novelty claim."}
    write_json(target / "verified_summary.json", summary)
    write_json(target / "bootstrap_draws.json", {"clusters": clusters, "multiplicities": draws.tolist(),
                                                "AP_by_model": bootstrap, "paired_mean_effect": effects.tolist()})
    for name, rows in (("comparison.csv", [{"method": method, **{k: values[k]["mean"] for k in ("AP", "AP50", "AP75")},
                                           "AP_sample_SD": values["AP"]["sample_std"]}
                                          for method, values in result["summary"].items()]),
                       ("paired.csv", paired), ("per_class.csv", class_rows), ("per_video.csv", video_rows)):
        with (target / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for name in ("confirmation_results.json", "confirmation_environment.json"):
        shutil.copyfile(source / name, target / name)
    write_json(target / "confirmation_artifact_sha256.json", hashes)
    write_readme(target, summary)
    plot(target, summary, effects)
    print("FINAL PAIRED EFFECT", summary["mean_AP_gain"], "95% interval", interval,
          "seed wins", summary["paired_seed_wins"], "/3", flush=True)


def write_readme(target, summary):
    low, high = summary["video_bootstrap"]["percentile_95_interval"]
    outcome = "The conditional interval is positive on this protocol." if low > 0 else (
        "The conditional interval is negative on this protocol." if high < 0 else "The interval includes zero; this test does not establish a gain.")
    lines = ["# Objectness in the final detection score", "",
             f"The fresh-set paired change is **{summary['mean_AP_gain']:+.3f} AP points**.",
             f"The paired, class-stratified query-video bootstrap 95% interval is **[{low:+.3f}, {high:+.3f}]**.", outcome, "",
             "## Method", "",
             "Use `score = class_score * objectness ** beta` before the existing class-wise NMS. Validation selected beta=1 from {0, 0.5, 1}. Keep the original foreground/background decision, boxes, model weights, and temporal memory computation. beta=0 is the baseline.",
             "", "This is a ranking heuristic that adds beta*log(objectness) to log(class_score). The scores are correlated and need not be calibrated probabilities. The product is not claimed as an exact probability. This applies known objectness information to this reduced few-shot video decoder; no globally new detection principle is claimed.",
             "", "## Fresh results", "", "| Method | AP mean ± seed SD | AP50 | AP75 |", "|---|---:|---:|---:|"]
    for method, values in summary["summary"].items():
        lines.append(f"| {method} | {values['AP']['mean']:.3f} ± {values['AP']['sample_std']:.3f} | {values['AP50']['mean']:.3f} | {values['AP75']['mean']:.3f} |")
    lines += ["", "| Seed | Baseline AP | Objectness AP | Difference |", "|---|---:|---:|---:|"]
    for row in summary["paired"]:
        lines.append(f"| {row['seed']} | {row['baseline_AP']:.3f} | {row['objectness_AP']:.3f} | {row['difference']:+.3f} |")
    lines += ["", "## Class AP", "", "| Class | Baseline | Objectness | Difference |", "|---|---:|---:|---:|"]
    for row in summary["per_class_AP"]:
        lines.append(f"| {row['class_name']} | {row['baseline']:.3f} | {row['objectness']:.3f} | {row['difference']:+.3f} |")
    largest = max(summary["per_class_AP"], key=lambda r: r["difference"])
    wins = sum(r["difference"] > 0 for r in summary["per_class_AP"])
    lines += ["", f"{wins} of {len(summary['per_class_AP'])} class means improved. The largest gain was {largest['class_name']} ({largest['difference']:+.3f} points). Interpret the aggregate mean together with the class table."]
    lines += ["", "## Protocol and limits", "",
              f"Six unused classes: {', '.join(summary['classes'].values())}; {summary['distinct_videos']} videos; {summary['distinct_frames']} distinct frames; {summary['episodes']} balanced two-way one-shot episodes. All prior classes and videos were excluded. Model hashes, selected beta, data, episodes, and the complete shared feature cache were frozen in Git before scoring.",
              "", "Both methods use the SAME three 300-update-budget baseline heads, with original validation-selected checkpoints (steps 250, 250, 300). No additional head training, encoder pass, trainable parameter, or label is introduced by score calibration. The score uses each proposal's frozen encoder objectness before learned box refinement.",
              "", "Validation AP averaged 59.871 without calibration and 75.138 with beta=1. All three validation pairs improved. These validation scores selected the method; only the separate unused-class set supplies confirmation evidence. beta=0 exactly reproduced all three archived baseline validation AP values.",
              "", "The 1,000 paired bootstrap draws resample query videos within each fixed class and retain all related frames and episodes. The interval is conditional on those classes, fixed support examples, and trained seed pairs. Training-seed SD is not data uncertainty. A video can supply support to another query episode; support sampling uncertainty and new-class uncertainty are not covered. Do not claim a gain on the original paper's full benchmark or every new dataset.",
              "", "## Records and reproduction", "",
              "`protocol.json` and `validation/` preserve the pre-score plan and all seven validation evaluations. `confirmation_protocol/` contains the unused-class selection, episodes, support pool, verified image records, and complete feature hashes. `evaluations/` contains all six raw metric/prediction/ground-truth sets. Every AP/AP50/AP75 value was recomputed exactly. `bootstrap_draws.json` stores all multiplicities and per-model AP draws. `confirmation_artifact_sha256.json` verifies raw source bytes. `per_video.csv` is a diagnostic; its mean is not aggregate COCO AP.",
              "", "![Score calibration results](score_calibration.png)", "",
              "See [the pre-score plan](../../docs/SCORE_CALIBRATION_PLAN.md) and [the runbook](../../docs/RUNBOOK.md). The same trained heads support both methods; evaluation must set `--objectness-beta 0` for baseline or `--objectness-beta 1` for the proposed score. Encoder weights are public and pinned. Head weights are verified locally; public download links remain required before final submission.",
              "", "The shared head training configurations and complete logs are in [fixed seed 42](../revision_v2/source_runs/fixed_tau060_seed42/), [seed 43](../revision_v2/source_runs/fixed_tau060_seed43/), and [seed 44](../revision_v2/source_runs/fixed_tau060_seed44/). They record the original CPU environment, hyperparameters, and validation checkpoint selection. No separate proposed-head training log exists because the proposed method performs no additional training.",
              "", "All earlier unsuccessful memory and centering studies remain in the repository. They are not replaced by this result.", ""]
    (target / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot(target, summary, effects):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained")
    methods = ("baseline", "objectness")
    means = [summary["summary"][m]["AP"]["mean"] for m in methods]
    errors = [summary["summary"][m]["AP"]["sample_std"] for m in methods]
    bars = axes[0].bar(["Baseline", "With objectness"], means, yerr=errors, capsize=4,
                       color=["#2563eb", "#059669"], width=.55)
    axes[0].bar_label(bars, fmt="%.2f", padding=7)
    axes[0].set(title="Unused-class confirmation: box AP", ylabel="AP (%)", ylim=(0, max(means) + 13))
    axes[0].text(.02, .98, "Error bars: training-seed sample SD", transform=axes[0].transAxes, va="top", fontsize=8)
    axes[1].hist(effects, bins=30, color="#059669", alpha=.8)
    axes[1].axvline(0, color="#334155", linestyle="--", label="Zero difference")
    for index, value in enumerate(summary["video_bootstrap"]["percentile_95_interval"]):
        axes[1].axvline(value, color="#b45309", linestyle=":", label="95% interval" if index == 0 else "_nolegend_")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set(title="Paired video bootstrap (1,000 draws)", xlabel="Objectness - baseline (AP points)", ylabel="Draw count")
    rows = summary["per_class_AP"]
    axes[2].barh([r["class_name"] for r in rows], [r["difference"] for r in rows], color="#059669")
    axes[2].axvline(0, color="#334155", linestyle="--")
    axes[2].set(title="Class AP differences", xlabel="Objectness - baseline (AP points)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(target / "score_calibration.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/score_calibration"))
    args = parser.parse_args()
    analyse(args.experiment_dir.resolve(), args.output)
