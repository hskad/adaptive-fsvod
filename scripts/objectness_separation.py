"""How well do class confidence p and encoder objectness q separate correct from incorrect candidates?

Uses every foreground candidate before NMS (argmax is a support class, p >= 0.001, valid
box) from the three shared heads. A candidate is correct when its box has IoU >= 0.5 with
a ground-truth box of its predicted class in that frame. Reports AUROC and the mean log
gap (mean log of correct minus mean log of incorrect) for p, q, and p*q, overall and per
class. This tests the ranking condition beta*log(q_a/q_b) > log(p_b/p_a) in the report.
"""

import argparse
import math
from collections import defaultdict
from pathlib import Path

import torch

from adaptive_fsvod.cache import CachedEpisodes
from adaptive_fsvod.data import VideoManifest, read_json, write_json
from adaptive_fsvod.evaluate import load_detector
from adaptive_fsvod.geometry import pixel_boxes


def auroc(positive, negative):
    values = sorted([(v, 1) for v in positive] + [(v, 0) for v in negative])
    ranks, i = [0.0] * len(values), 0
    while i < len(values):  # average ranks for ties
        j = i
        while j + 1 < len(values) and values[j + 1][0] == values[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, (_, label) in zip(ranks, values) if label)
    n1, n0 = len(positive), len(negative)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def box_iou(a, b):
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area = lambda t: (t[:, 2] - t[:, 0]) * (t[:, 3] - t[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def collect(manifest, cache, episodes_path, checkpoints):
    cached = CachedEpisodes(VideoManifest(manifest), cache)
    episodes = read_json(episodes_path)["episodes"]
    records = defaultdict(lambda: {"p": ([], []), "q": ([], [])})  # (incorrect, correct)
    for checkpoint in checkpoints:
        model, _ = load_detector(checkpoint, "cpu")
        with torch.inference_mode():
            for episode in episodes:
                frames, support, _, info = cached.load(episode, "cpu")
                for output, frame, encoded in zip(model(frames, support), info, frames):
                    probabilities = output["logits"].softmax(-1)
                    p, labels = probabilities[:, :-1].max(-1)
                    q = encoded["objectness"].float()
                    boxes = pixel_boxes(output["boxes"].clone(), frame["width"], frame["height"])
                    foreground = (probabilities.argmax(-1) < len(episode["classes"])) & (p >= 0.001)
                    foreground &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                    for index, category in enumerate(episode["classes"]):
                        candidates = torch.where(foreground & (labels == index))[0]
                        if not len(candidates):
                            continue
                        truth = [o["bbox"] for o in frame["objects"] if str(o["category_id"]) == str(category)]
                        correct = (box_iou(boxes[candidates], torch.tensor(truth, dtype=boxes.dtype)).max(1).values >= 0.5
                                   if truth else torch.zeros(len(candidates), dtype=torch.bool))
                        name = cached.manifest.classes.get(str(category), str(category))
                        for key in ("all", name):
                            for j, good in zip(candidates.tolist(), correct.tolist()):
                                records[key]["p"][int(good)].append(float(p[j]))
                                records[key]["q"][int(good)].append(float(q[j]))
    return records


def summarise(records):
    mean_log = lambda v: sum(math.log(max(x, 1e-12)) for x in v) / len(v)
    result = {}
    for key, r in records.items():
        (p_wrong, p_right), (q_wrong, q_right) = r["p"], r["q"]
        if not p_right or not p_wrong:
            continue
        s_right = [a * b for a, b in zip(p_right, q_right)]
        s_wrong = [a * b for a, b in zip(p_wrong, q_wrong)]
        result[key] = {"correct": len(p_right), "incorrect": len(p_wrong),
                       "AUROC": {"p": auroc(p_right, p_wrong), "q": auroc(q_right, q_wrong),
                                 "pq": auroc(s_right, s_wrong)},
                       "mean_log_gap": {"p": mean_log(p_right) - mean_log(p_wrong),
                                        "q": mean_log(q_right) - mean_log(q_wrong)},
                       "mean": {"p": [sum(p_right) / len(p_right), sum(p_wrong) / len(p_wrong)],
                                "q": [sum(q_right) / len(q_right), sum(q_wrong) / len(q_wrong)]}}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/score_calibration/objectness_separation.json"))
    args = parser.parse_args()
    root = args.experiment_dir
    checkpoints = [(args.weights_dir or root / "weights") / f"shared_seed{s}.pt" for s in (42, 43, 44)]
    confirmation = root / "score_calibration/confirmation"
    record = {
        "definition": "pre-NMS foreground candidates of seeds 42/43/44; correct = IoU>=0.5 with a same-class box",
        "validation": summarise(collect(root / "pilot/val.json", root / "cache/val",
                                        root / "pilot/val_episodes.json", checkpoints)),
        "test": summarise(collect(confirmation / "confirmation.json", confirmation / "cache",
                                  confirmation / "episodes.json", checkpoints)),
    }
    write_json(args.output, record)
    for split in ("validation", "test"):
        print(split, record[split]["all"])
