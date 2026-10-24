"""Frame-level COCO box evaluation with unique episode/frame IDs."""

import argparse
import contextlib
import io
import time
from pathlib import Path

import torch

from .cache import CachedEpisodes
from .data import VideoManifest, read_json, validate_episodes, write_json
from .geometry import nms, pixel_boxes
from .model import ModelConfig, TemporalDetector


@torch.no_grad()
def decode(output, classes, width, height, score_floor=0.001, nms_iou=0.5, max_detections=100):
    probabilities = output["logits"].softmax(-1)
    scores, labels = probabilities[:, :-1].max(-1)
    boxes = pixel_boxes(output["boxes"].clone(), width, height)
    winning_foreground = probabilities.argmax(-1) < len(classes)
    valid = winning_foreground & (scores >= score_floor)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    selected = []
    for index in range(len(classes)):
        candidates = torch.where(valid & (labels == index))[0]
        kept = nms(boxes[candidates], scores[candidates], nms_iou)
        selected.extend(candidates[kept].tolist())
    selected = sorted(selected, key=lambda i: float(scores[i]), reverse=True)[:max_detections]
    return [{"bbox": boxes[i].cpu().tolist(), "score": float(scores[i]),
             "category_id": classes[int(labels[i])]} for i in selected]


def coco_metrics(ground_truth, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not ground_truth["annotations"]:
        raise ValueError("No ground truth objects in evaluation episodes")
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = ground_truth
        truth.createIndex()
        detections = truth.loadRes(predictions)
        evaluation = COCOeval(truth, detections, "bbox")
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    return {name: float(evaluation.stats[index] * 100)
            for index, name in enumerate(("AP", "AP50", "AP75"))}


@torch.inference_mode()
def evaluate(model, cached, episode_file, device="cpu", score_floor=0.001, nms_iou=0.5):
    validate_episodes(cached.manifest, episode_file)
    model.eval()
    categories = sorted(cached.manifest.classes)
    category_map = {c: i + 1 for i, c in enumerate(categories)}
    truth = {"info": {"description": "Frame-level few-shot box evaluation"}, "images": [],
             "annotations": [], "categories": [{"id": category_map[c], "name": cached.manifest.classes[c]}
                                                  for c in categories]}
    predictions = []
    memory_counts, thresholds = [], []
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    started = time.perf_counter()
    for episode in episode_file["episodes"]:
        frames, support, _, info = cached.load(episode, device)
        outputs = model(frames, support)  # A fresh call resets temporal memory.
        for output, frame in zip(outputs, info):
            image_id = len(truth["images"]) + 1
            truth["images"].append({"id": image_id, "width": frame["width"], "height": frame["height"],
                                    "file_name": frame["path"], "episode_id": episode["id"]})
            for obj in frame["objects"]:
                category = str(obj["category_id"])
                if category in episode["classes"]:
                    x1, y1, x2, y2 = obj["bbox"]
                    truth["annotations"].append({"id": len(truth["annotations"]) + 1,
                        "image_id": image_id, "category_id": category_map[category], "iscrowd": 0,
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1)})
            for detection in decode(output, episode["classes"], frame["width"], frame["height"], score_floor, nms_iou):
                x1, y1, x2, y2 = detection["bbox"]
                predictions.append({"image_id": image_id, "category_id": category_map[detection["category_id"]],
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": detection["score"]})
            memory_counts.append(output["memory"].shape[0])
            thresholds.extend(output["thresholds"].cpu().tolist())
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    metrics = coco_metrics(truth, predictions)
    metrics.update({"frames": len(truth["images"]), "episodes": len(episode_file["episodes"]),
                    "seconds": elapsed, "cached_pipeline_fps": len(truth["images"]) / max(elapsed, 1e-8),
                    "timing_scope": "cached feature IO, trainable heads, decoding; encoder excluded",
                    "mean_memory_tokens": sum(memory_counts) / len(memory_counts),
                    "mean_threshold": sum(thresholds) / len(thresholds),
                    "synthetic": cached.metadata.get("synthetic", False),
                    "score_floor": score_floor, "nms_iou": nms_iou,
                    "metric": "frame-level COCO bbox AP, IoU 0.50:0.05:0.95, maxDets=100"})
    return metrics, truth, predictions


def load_detector(checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = TemporalDetector(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    cached = CachedEpisodes(VideoManifest(args.manifest), args.cache)
    model, checkpoint = load_detector(args.checkpoint, args.device)
    if checkpoint["cache_signature"] != cache_signature(cached.metadata):
        raise ValueError("Evaluation features differ from the training encoder settings")
    episodes = read_json(args.episodes)
    result, truth, predictions = evaluate(model, cached, episodes, args.device, args.score_floor, args.nms_iou)
    result.update({"checkpoint": str(Path(args.checkpoint).resolve()), "training_step": checkpoint["step"],
                   "model_config": checkpoint["model_config"], "training_git": checkpoint["environment"].get("git")})
    for name, value in [("metrics", result), ("ground_truth", truth), ("predictions", predictions)]:
        write_json(Path(args.output) / f"{name}.json", value)
    print(result)


def cache_signature(metadata):
    return {k: metadata.get(k) for k in ("encoder", "revision", "candidates", "feature_dim", "synthetic")}


if __name__ == "__main__":
    main()
