"""Frozen encoder extraction and resumable feature caches."""

import argparse
import os
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .data import VideoManifest, checksum, frame_key, read_json, support_pool, write_json
from .geometry import normalize_boxes


def save_tensor(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class FrozenEncoder:
    def __init__(self, model_id="google/owlv2-base-patch16-ensemble", device="cpu",
                 revision=None, candidates=128, model=None, processor=None):
        if model is None or processor is None:
            os.environ.setdefault("USE_TF", "0")
            from transformers import Owlv2ForObjectDetection, Owlv2Processor
            processor = Owlv2Processor.from_pretrained(model_id, revision=revision)
            model = Owlv2ForObjectDetection.from_pretrained(model_id, revision=revision, use_safetensors=True)
        self.model = model.to(device).eval()
        self.model.requires_grad_(False)
        self.processor = processor
        self.device = torch.device(device)
        self.candidates = candidates
        self.model_id = model_id
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    @torch.inference_mode()
    def encode(self, image, support=False):
        inputs = self.processor(images=image, return_tensors="pt")
        values = inputs["pixel_values"].to(self.device)
        feature_map = self.model.image_embedder(pixel_values=values)[0]
        features = feature_map.flatten(1, 2)
        _, class_features = self.model.class_predictor(features)
        objectness = self.model.objectness_predictor(features)[0]
        if support:
            return class_features[0, objectness.argmax()].float().cpu()
        boxes = self.model.box_predictor(features, feature_map)[0]
        count = min(self.candidates, features.shape[1])
        indices = torch.topk(objectness, count, sorted=True).indices
        return {"features": class_features[0, indices].half().cpu(),
                "boxes": boxes[indices].float().cpu(),
                "objectness": objectness[indices].sigmoid().float().cpu()}


def build_cache(manifest, output, encoder, per_class=32, pool_seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pool = support_pool(manifest, per_class, pool_seed)
    signature = {"schema_version": 1, "manifest_sha256": checksum(manifest.path),
                 "encoder": encoder.model_id, "revision": encoder.revision,
                 "candidates": encoder.candidates, "pool_seed": pool_seed,
                 "support_per_class": per_class, "synthetic": False}
    metadata_file = output / "metadata.json"
    if metadata_file.exists():
        previous = read_json(metadata_file)
        if any(previous.get(k) != v for k, v in signature.items()):
            raise ValueError("Cache settings differ. Use a different output directory.")
    write_json(metadata_file, {**signature, "complete": False})
    write_json(output / "support_pool.json", pool)
    start = time.perf_counter()
    extracted = 0
    total_frames = 0
    feature_dim = None
    for vid, video in tqdm(manifest.videos.items(), desc="Cache query videos"):
        for index, frame in enumerate(video["frames"]):
            destination = output / "frames" / f"{frame_key(vid, frame)}.pt"
            if not destination.exists():
                image = manifest.image(vid, index)
                encoded = encoder.encode(image)
                save_tensor(destination, encoded)
                feature_dim = encoded["features"].shape[-1]
                extracted += 1
            total_frames += 1
    for category, examples in tqdm(pool.items(), desc="Cache support crops"):
        for example in examples:
            destination = output / "support" / f"{example['key']}.pt"
            if not destination.exists():
                image = manifest.image(example["video_id"], example["frame_index"])
                crop = image.crop(tuple(example["bbox"]))
                save_tensor(destination, encoder.encode(crop, support=True))
    if feature_dim is None:
        vid, video = next(iter(manifest.videos.items()))
        first = output / "frames" / f"{frame_key(vid, video['frames'][0])}.pt"
        feature_dim = torch.load(first, map_location="cpu", weights_only=True)["features"].shape[-1]
    metadata = {**signature, "complete": True, "feature_dim": feature_dim,
                "frames": total_frames, "new_frames_in_this_run": extracted,
                "cache_run_seconds": time.perf_counter() - start,
                "timing_scope": "feature cache construction, including support crops and file IO"}
    write_json(metadata_file, metadata)
    return metadata


class CachedEpisodes:
    def __init__(self, manifest, cache_root):
        self.manifest = manifest
        self.root = Path(cache_root)
        self.metadata = read_json(self.root / "metadata.json")
        if self.metadata["manifest_sha256"] != checksum(manifest.path):
            raise ValueError("Feature cache does not match manifest")
        if not self.metadata.get("complete"):
            raise ValueError("Feature cache is incomplete. Resume cache construction first.")
        self.pool = read_json(self.root / "support_pool.json")

    def load(self, episode, device="cpu"):
        classes = episode["classes"]
        support = []
        for category in classes:
            vectors = [torch.load(self.root / "support" / f"{item['key']}.pt",
                                  map_location="cpu", weights_only=True).float()
                       for item in episode["support"][category]]
            support.append(torch.stack(vectors).to(device))
        frames, targets, frame_info = [], [], []
        for index in episode["frame_indices"]:
            frame = self.manifest.videos[episode["video_id"]]["frames"][index]
            path = self.root / "frames" / f"{frame_key(episode['video_id'], frame)}.pt"
            encoded = torch.load(path, map_location="cpu", weights_only=True)
            frames.append({k: encoded[k].float().to(device) for k in ("features", "boxes")})
            objects = [o for o in frame["objects"] if str(o["category_id"]) in classes]
            targets.append({
                "boxes": normalize_boxes([o["bbox"] for o in objects], frame["width"], frame["height"]).to(device),
                "labels": torch.tensor([classes.index(str(o["category_id"])) for o in objects],
                                       dtype=torch.long, device=device),
            })
            frame_info.append(frame)
        return frames, support, targets, frame_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--images-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--support-per-class", type=int, default=32)
    args = parser.parse_args()
    manifest = VideoManifest(args.manifest, args.images_root)
    encoder = FrozenEncoder(args.model, args.device, args.revision, args.candidates)
    print(build_cache(manifest, args.output, encoder, args.support_per_class))


if __name__ == "__main__":
    main()
