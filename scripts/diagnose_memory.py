"""Inspect validation memory agreement and a same-checkpoint memory-off ablation.

Token agreement is not detection precision: several selected tokens may match
one object. Final-frame selections are excluded because no next frame consumes
them. Only validation data is used.
"""

import argparse
import json
from pathlib import Path

import torch

from adaptive_fsvod.cache import CachedEpisodes
from adaptive_fsvod.data import VideoManifest, read_json, write_json
from adaptive_fsvod.evaluate import evaluate, load_detector
from adaptive_fsvod.geometry import box_iou, pixel_boxes


def diagnose(root, seed, device):
    manifest = VideoManifest(root / "pilot/val.json")
    episodes = read_json(root / "pilot/val_episodes.json")
    cached = CachedEpisodes(manifest, root / "cache/val")
    results = {}
    for method in ("fixed", "adaptive"):
        model, checkpoint = load_detector(root / "runs" / f"{method}_seed{seed}/best.pt", device)
        counts = {"selected": 0, "correct_iou50": 0, "below_0_7": 0,
                  "below_0_7_correct": 0, "ground_truth": 0, "covered_iou50": 0,
                  "thresholds": 0, "thresholds_at_lower_bound": 0,
                  "consumed_frames": 0}
        cursor = [0]

        def inspect(module, inputs, outputs):
            episode = episodes["episodes"][cursor[0]]
            cursor[0] += 1
            video = manifest.videos[episode["video_id"]]
            for index, output in zip(episode["frame_indices"][:-1], outputs[:-1]):
                frame = video["frames"][index]
                objects = [obj for obj in frame["objects"] if str(obj["category_id"]) in episode["classes"]]
                selected = output["memory_indices"]
                probabilities = output["logits"].softmax(-1)[selected]
                scores, labels = probabilities.max(-1)
                correct = torch.zeros(len(selected), dtype=torch.bool, device=device)
                covered = torch.zeros(len(objects), dtype=torch.bool, device=device)
                if len(objects) and len(selected):
                    boxes = pixel_boxes(output["boxes"][selected], frame["width"], frame["height"])
                    target_boxes = boxes.new_tensor([obj["bbox"] for obj in objects])
                    target_labels = labels.new_tensor([episode["classes"].index(str(obj["category_id"])) for obj in objects])
                    iou, _ = box_iou(boxes, target_boxes)
                    matches = (iou >= 0.5) & (labels[:, None] == target_labels[None, :])
                    correct, covered = matches.any(1), matches.any(0)
                thresholds = output["thresholds"]
                counts["selected"] += len(selected)
                counts["correct_iou50"] += int(correct.sum())
                counts["below_0_7"] += int((scores < .7).sum())
                counts["below_0_7_correct"] += int((correct & (scores < .7)).sum())
                counts["ground_truth"] += len(objects)
                counts["covered_iou50"] += int(covered.sum())
                counts["thresholds"] += len(thresholds)
                counts["thresholds_at_lower_bound"] += int(torch.isclose(
                    thresholds, thresholds.new_tensor(module.config.threshold_min), atol=1e-6).sum())
                counts["consumed_frames"] += 1

        handle = model.register_forward_hook(inspect)
        metrics, _, _ = evaluate(model, cached, episodes, device)
        handle.remove()
        logs = [json.loads(line) for line in (root / "runs" / f"{method}_seed{seed}/training.jsonl").read_text().splitlines()]
        reported = next(r["validation"] for r in logs if r["step"] == checkpoint["step"])
        counts["token_agreement_fraction"] = counts["correct_iou50"] / max(1, counts["selected"])
        counts["object_coverage_fraction"] = counts["covered_iou50"] / max(1, counts["ground_truth"])
        counts["lower_confidence_agreement_fraction"] = (
            counts["below_0_7_correct"] / counts["below_0_7"] if counts["below_0_7"] else None)
        counts["lower_bound_fraction"] = counts["thresholds_at_lower_bound"] / max(1, counts["thresholds"])
        print(method, "validation with memory:", metrics["AP"], "token diagnostics:", counts, flush=True)
        model.config.memory_mode = "none"
        without, _, _ = evaluate(model, cached, episodes, device)
        results[method] = {"seed": seed, "checkpoint_step": checkpoint["step"],
                           "reported_validation_AP": reported["AP"], "validation": metrics,
                           "token_agreement": counts, "memory_off_validation": without,
                           "AP_change_when_memory_disabled": without["AP"] - metrics["AP"]}
        print(method, "validation with memory disabled:", without["AP"], flush=True)
    return {"split": "val", "seed": seed, "methods": results,
            "note": "Same trained weights with memory disabled at inference; not a retrained no-memory baseline. Token agreement is many-to-one IoU50 and correct class, not detection precision."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = diagnose(args.experiment_dir, args.seed, args.device)
    path = args.experiment_dir / "exports" / f"validation_memory_diagnostics_seed{args.seed}.json"
    write_json(path, result)
    print("Saved:", path, flush=True)
