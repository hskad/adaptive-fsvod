"""Synthetic end-to-end check. Outputs are NOT research results."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .cache import CachedEpisodes, save_tensor
from .data import VideoManifest, checksum, frame_key, make_episodes, support_pool, write_json
from .evaluate import evaluate, load_detector
from .geometry import normalize_boxes
from .model import ModelConfig
from .train import parser as training_parser, train


def make_synthetic(root, feature_dim=16):
    root = Path(root)
    generator = torch.Generator().manual_seed(123)
    centers = torch.nn.functional.normalize(torch.randn(6, feature_dim, generator=generator), dim=-1)
    for split_index, split in enumerate(("train", "val", "test")):
        classes = {str(split_index * 2 + i): f"synthetic-{split_index * 2 + i}" for i in range(2)}
        videos = []
        for category in classes:
            for video_index in range(2):
                vid = f"{split}-{category}-{video_index}"
                frames = []
                for frame_index in range(3):
                    path = f"{vid}/{frame_index:03d}.png"
                    bbox = [5 + frame_index, 5, 17 + frame_index, 17]
                    image = Image.new("RGB", (40, 24), (30, 30, 30))
                    objects = [] if frame_index == 2 else [{"category_id": category, "bbox": bbox, "track_id": "0"}]
                    if objects:
                        ImageDraw.Draw(image).rectangle(bbox, fill=(200, 60, 60))
                    image_path = root / "images" / path
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(image_path)
                    frames.append({"path": path, "width": 40, "height": 24, "objects": objects})
                videos.append({"id": vid, "frames": frames})
        manifest_path = root / f"{split}.json"
        write_json(manifest_path, {"schema_version": 1, "dataset": "SYNTHETIC-SOFTWARE-CHECK",
                    "split": split, "images_root": "images", "classes": classes, "videos": videos})
        manifest = VideoManifest(manifest_path)
        cache = root / "cache" / split
        pool = support_pool(manifest, per_class=8)
        for vid, video in manifest.videos.items():
            category = int(vid.split("-")[1])
            for frame in video["frames"]:
                features = torch.randn(8, feature_dim, generator=generator) * 0.2
                if frame["objects"]:
                    features[0] = centers[category] + torch.randn(feature_dim, generator=generator) * 0.02
                boxes = torch.tensor([[0.75, 0.2, 0.1, 0.1]]).repeat(8, 1)
                if frame["objects"]:
                    boxes[0] = normalize_boxes([frame["objects"][0]["bbox"]], 40, 24)[0]
                save_tensor(cache / "frames" / f"{frame_key(vid, frame)}.pt", {"features": features, "boxes": boxes})
        for category, examples in pool.items():
            for item in examples:
                save_tensor(cache / "support" / f"{item['key']}.pt", centers[int(category)])
        write_json(cache / "metadata.json", {"manifest_sha256": checksum(manifest_path), "complete": True,
                    "encoder": "synthetic", "revision": "1", "candidates": 8,
                    "feature_dim": feature_dim, "synthetic": True})
        write_json(cache / "support_pool.json", pool)
        episodes = make_episodes(manifest, pool, count=4, n_way=2, k_shot=1,
                                 max_frames=2 if split == "train" else None)
        write_json(root / f"{split}_episodes.json", episodes)
    for mode in ("fixed", "adaptive"):
        config = ModelConfig(input_dim=feature_dim, hidden_dim=feature_dim, dropout=0, memory_mode=mode)
        write_json(root / f"{mode}.json", asdict(config))
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/synthetic_check")
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = make_synthetic(Path(args.output) / "inputs")
    for mode in ("fixed", "adaptive"):
        command = []
        for flag, value in {
            "train-manifest": root / "train.json", "val-manifest": root / "val.json",
            "train-cache": root / "cache/train", "val-cache": root / "cache/val",
            "train-episodes": root / "train_episodes.json", "val-episodes": root / "val_episodes.json",
            "config": root / f"{mode}.json", "output": Path(args.output) / mode,
            "steps": args.steps, "eval-every": args.steps, "device": "cpu",
        }.items():
            command.extend(["--" + flag, str(value)])
        checkpoint = train(training_parser().parse_args(command))
        detector, _ = load_detector(checkpoint)
        cached = CachedEpisodes(VideoManifest(root / "test.json"), root / "cache/test")
        from .data import read_json
        metrics, _, _ = evaluate(detector, cached, read_json(root / "test_episodes.json"))
        write_json(Path(args.output) / mode / "SYNTHETIC_metrics.json", metrics)
    print("PASS: data loading, training, checkpoint reload, and evaluation for both modes.")
    print("These synthetic outputs are software checks. Do not use them in the project report.")


if __name__ == "__main__":
    main()
