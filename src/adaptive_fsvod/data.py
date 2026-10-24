"""Manifest validation and reproducible, video-disjoint support episodes."""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_key(video_id, frame):
    return hashlib.sha256(f"{video_id}|{frame['path']}".encode()).hexdigest()[:24]


def support_key(video_id, frame, obj):
    raw = json.dumps([video_id, frame["path"], obj["category_id"], obj["bbox"]])
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class VideoManifest:
    def __init__(self, path, images_root=None):
        self.path = Path(path).resolve()
        self.data = read_json(self.path)
        self.root = Path(images_root or self.data.get("images_root", "."))
        if not self.root.is_absolute():
            self.root = (self.path.parent / self.root).resolve()
        self.classes = {str(k): v for k, v in self.data["classes"].items()}
        self.videos = {str(v["id"]): v for v in self.data["videos"]}
        if len(self.videos) != len(self.data["videos"]):
            raise ValueError("Duplicate video IDs in manifest")
        self.objects = defaultdict(list)
        for video_id, video in self.videos.items():
            seen_frames = set()
            for index, frame in enumerate(video["frames"]):
                if frame["path"] in seen_frames:
                    raise ValueError(f"Duplicate frame: {frame['path']}")
                seen_frames.add(frame["path"])
                if frame["width"] <= 0 or frame["height"] <= 0:
                    raise ValueError("Image dimensions must be positive")
                for obj in frame["objects"]:
                    category = str(obj["category_id"])
                    if category not in self.classes:
                        raise ValueError(f"Unknown category {category}")
                    x1, y1, x2, y2 = obj["bbox"]
                    if not (0 <= x1 < x2 <= frame["width"] and 0 <= y1 < y2 <= frame["height"]):
                        raise ValueError(f"Invalid pixel xyxy box in {frame['path']}")
                    self.objects[category].append({
                        "video_id": video_id, "frame_index": index,
                        "bbox": obj["bbox"], "category_id": category,
                        "key": support_key(video_id, frame, obj),
                    })
        if not self.videos or not self.objects:
            raise ValueError("Manifest must contain labeled videos")

    def image(self, video_id, frame_index):
        frame = self.videos[str(video_id)]["frames"][frame_index]
        with Image.open(self.root / frame["path"]) as source:
            image = source.convert("RGB")
        if image.size != (frame["width"], frame["height"]):
            raise ValueError(f"Image size differs from annotations: {frame['path']}")
        return image

    def validate_images(self):
        count = 0
        for vid, video in self.videos.items():
            for index in range(len(video["frames"])):
                self.image(vid, index)
                count += 1
        return {"videos": len(self.videos), "frames": count, "classes": len(self.objects)}


def validate_splits(manifests):
    for index, first in enumerate(manifests):
        for second in manifests[index + 1:]:
            if set(first.classes) & set(second.classes):
                raise ValueError("Categories overlap across splits")
            if set(first.videos) & set(second.videos):
                raise ValueError("Videos overlap across splits")


def support_pool(manifest, per_class=32, seed=42):
    """Round-robin across videos so a query video cannot consume the pool."""
    rng = random.Random(seed)
    pool = {}
    for category, examples in sorted(manifest.objects.items()):
        by_video = defaultdict(list)
        for example in examples:
            by_video[example["video_id"]].append(example)
        groups = list(by_video.values())
        rng.shuffle(groups)
        for group in groups:
            rng.shuffle(group)
        chosen = []
        while groups and len(chosen) < per_class:
            remaining = []
            for group in groups:
                if len(chosen) == per_class:
                    break
                chosen.append(group.pop())
                if group:
                    remaining.append(group)
            groups = remaining
        pool[category] = chosen
    return pool


def make_episodes(manifest, pool, count=20, n_way=2, k_shot=1, max_frames=None, seed=42):
    if count < 1 or n_way < 1 or k_shot < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("Episode parameters must be positive")
    rng = random.Random(seed)
    candidates = []
    for vid, video in sorted(manifest.videos.items()):
        present = {str(obj["category_id"]) for f in video["frames"] for obj in f["objects"]}
        available = {
            category: [item for item in items if item["video_id"] != vid]
            for category, items in pool.items()
        }
        available = {c: items for c, items in available.items() if len(items) >= k_shot}
        anchors = sorted(present & set(available))
        if anchors and len(available) >= n_way:
            candidates.append((vid, available, anchors))
    if not candidates:
        raise ValueError("No valid episodes. Need enough support examples outside each query video.")
    episodes = []
    for episode_index in range(count):
        vid, available, anchors = rng.choice(candidates)
        anchor = rng.choice(anchors)
        others = sorted(set(available) - {anchor})
        classes = sorted([anchor] + rng.sample(others, n_way - 1))
        support = {c: rng.sample(available[c], k_shot) for c in classes}
        indices = list(range(len(manifest.videos[vid]["frames"])))
        if max_frames is not None and len(indices) > max_frames:
            start = rng.randrange(len(indices) - max_frames + 1)
            indices = indices[start:start + max_frames]
        episodes.append({"id": episode_index, "video_id": vid, "classes": classes,
                         "support": support, "frame_indices": indices})
    return {"manifest_sha256": checksum(manifest.path), "seed": seed,
            "n_way": n_way, "k_shot": k_shot, "episodes": episodes}


def validate_episodes(manifest, episode_file):
    if episode_file["manifest_sha256"] != checksum(manifest.path):
        raise ValueError("Episodes were generated for a different manifest")
    for episode in episode_file["episodes"]:
        if episode["video_id"] not in manifest.videos:
            raise ValueError("Unknown query video")
        if len(set(episode["classes"])) != episode_file["n_way"]:
            raise ValueError("Invalid episode classes")
        for category in episode["classes"]:
            examples = episode["support"][category]
            if len(examples) != episode_file["k_shot"]:
                raise ValueError("Invalid support count")
            if len({item["key"] for item in examples}) != len(examples):
                raise ValueError("Repeated support example")
            for item in examples:
                if item["video_id"] == episode["video_id"]:
                    raise ValueError("Support/query video leakage")
                if item not in manifest.objects[category]:
                    raise ValueError("Support example does not match manifest")
        indices = episode["frame_indices"]
        if not indices or indices != sorted(set(indices)) or indices[0] < 0:
            raise ValueError("Query frame indices must be unique and ordered")
        if indices[-1] >= len(manifest.videos[episode["video_id"]]["frames"]):
            raise ValueError("Unknown query frame")
