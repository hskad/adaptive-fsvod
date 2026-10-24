"""Small real-data integration check, not a research training run."""

import argparse
from pathlib import Path

import torch

from adaptive_fsvod.cache import FrozenEncoder
from adaptive_fsvod.data import VideoManifest, checksum, write_json
from adaptive_fsvod.geometry import normalize_boxes
from adaptive_fsvod.loss import DetectionLoss
from adaptive_fsvod.model import ModelConfig, TemporalDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="runs/real_episode_check.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = VideoManifest(args.manifest)
    if manifest.data["split"] != "train":
        raise ValueError("Use training data for this integration check")
    available = {}
    for category, items in manifest.objects.items():
        available[category] = [item for item in items if (
            manifest.root / manifest.videos[item["video_id"]]["frames"][item["frame_index"]]["path"]).exists()]
    episode = None
    for vid, video in manifest.videos.items():
        indices = [i for i, frame in enumerate(video["frames"]) if (manifest.root / frame["path"]).exists()]
        present = {str(o["category_id"]) for i in indices for o in video["frames"][i]["objects"]}
        choices = {c: [item for item in items if item["video_id"] != vid] for c, items in available.items()}
        choices = {c: items for c, items in choices.items() if items}
        anchors = sorted(present & set(choices))
        if len(indices) >= 2 and anchors and len(choices) >= 2:
            anchor = anchors[0]
            classes = sorted([anchor, next(c for c in sorted(choices) if c != anchor)])
            episode = {"video_id": vid, "frame_indices": indices[:2], "classes": classes,
                       "support": {c: choices[c][0] for c in classes}}
            break
    if episode is None:
        raise ValueError("Need two query frames and support examples from other videos for two classes")
    encoder = FrozenEncoder(device=args.device, revision="cfd3195ba4ea9592eec887ded089f4c08eff231d")
    support = []
    for category in episode["classes"]:
        item = episode["support"][category]
        image = manifest.image(item["video_id"], item["frame_index"]).crop(tuple(item["bbox"]))
        support.append(encoder.encode(image, support=True)[None].to(args.device))
    frames, targets = [], []
    for index in episode["frame_indices"]:
        frame = manifest.videos[episode["video_id"]]["frames"][index]
        features = encoder.encode(manifest.image(episode["video_id"], index))
        frames.append({k: features[k].float().to(args.device) for k in ("features", "boxes")})
        objects = [o for o in frame["objects"] if str(o["category_id"]) in episode["classes"]]
        targets.append({"boxes": normalize_boxes([o["bbox"] for o in objects], frame["width"], frame["height"]).to(args.device),
            "labels": torch.tensor([episode["classes"].index(str(o["category_id"])) for o in objects],
                                   dtype=torch.long, device=args.device)})
    results = {}
    for mode in ("fixed", "adaptive"):
        torch.manual_seed(42)
        model = TemporalDetector(ModelConfig(memory_mode=mode)).to(args.device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        outputs = model(frames, support)
        loss = torch.stack([DetectionLoss()(out, target)["total"] for out, target in zip(outputs, targets)]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite real-data loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .1, error_if_nonfinite=True)
        optimizer.step()
        results[mode] = {"loss": float(loss.detach()), "gradient_norm": float(norm),
                         "memory_tokens": [o["memory"].shape[0] for o in outputs], "optimizer_updates": 1}
    result = {"purpose": "real-data integration check only; not research accuracy evidence",
              "manifest_sha256": checksum(args.manifest), "episode": episode, "device": args.device,
              "encoder_revision": encoder.revision, "results": results}
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
