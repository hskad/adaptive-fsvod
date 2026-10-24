"""Prepare the fresh confirmation set specified before revision experiments."""

import argparse
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import torch

from adaptive_fsvod.cache import FrozenEncoder, build_cache
from adaptive_fsvod.data import (VideoManifest, checksum, make_episodes, read_json,
                                validate_splits, write_json)
from adaptive_fsvod.download_frames import SOURCE, RangeReader, download_frames, frame_index


def immutable(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing confirmation selection differs: {path}")
    else:
        write_json(path, value)


def prepare(root):
    output = root / "revision_v2/confirmation"
    annotation = root / "raw/fsytv-boxes.json"
    source = read_json(annotation)
    originals = [VideoManifest(root / f"pilot/{s}.json") for s in ("train", "val", "test")]
    excluded_classes = set().union(*(set(m.classes) for m in originals))
    excluded_videos = set().union(*(set(m.videos) for m in originals))
    categories = {str(c["id"]): c["name"] for c in source["categories"]}
    tracks = defaultdict(list)
    for track in source["annotations"]:
        if not track.get("iscrowd", 0):
            tracks[str(track["video_id"])].append(track)
    with RangeReader(SOURCE["url"], SOURCE["archive_bytes"], root / "download_ranges") as reader:
        with zipfile.ZipFile(reader) as archive:
            available = set(frame_index(archive))
    eligible = defaultdict(list)
    for video in sorted(source["videos"], key=lambda v: str(v["id"])):
        vid = str(video["id"])
        present = {str(t["category_id"]) for t in tracks[vid]}
        if vid in excluded_videos or len(present) != 1 or present & excluded_classes:
            continue
        files = video["file_names"]
        n = min(3, len(files))
        indices = [round(i * (len(files) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        if not files or any(files[i] not in available for i in indices):
            continue
        frames = []
        for index in indices:
            objects = []
            for track in tracks[vid]:
                if index >= len(track["bboxes"]) or track["bboxes"][index] is None:
                    continue
                x, y, width, height = track["bboxes"][index]
                box = [max(0., x), max(0., y), min(video["width"], x + width), min(video["height"], y + height)]
                if box[2] > box[0] and box[3] > box[1]:
                    objects.append({"category_id": str(track["category_id"]), "bbox": box, "track_id": str(track["id"])})
            frames.append({"path": files[index], "source_index": index, "width": video["width"],
                           "height": video["height"], "objects": objects})
        if any(f["objects"] for f in frames):
            eligible[next(iter(present))].append({"id": vid, "frames": frames})
    rng = random.Random(20261024)
    classes = sorted(c for c, videos in eligible.items() if len(videos) >= 2)
    rng.shuffle(classes)
    classes = classes[:3]
    if len(classes) != 3:
        raise ValueError("Insufficient eligible fresh classes")
    videos = []
    for category in classes:
        candidates = eligible[category].copy()
        rng.shuffle(candidates)
        videos.extend(candidates[:2])
    manifest = {"schema_version": 1, "dataset": "FSYTV-40-fresh-confirmation", "split": "confirmation",
                "images_root": "images", "classes": {c: categories[c] for c in sorted(classes)},
                "videos": sorted(videos, key=lambda v: v["id"])}
    path = output / "confirmation.json"
    immutable(path, manifest)
    selected = VideoManifest(path)
    validate_splits(originals + [selected])
    record = {"selection_seed": 20261024, "episode_seed": 20261025, "pool_seed": 42,
              "annotation_sha256": checksum(annotation), "source": SOURCE,
              "original_manifest_sha256": {m.data["split"]: checksum(m.path) for m in originals},
              "classes": selected.classes, "video_ids": sorted(selected.videos),
              "frames": sum(len(v["frames"]) for v in selected.videos.values()),
              "manifest_sha256": checksum(path), "support_per_class": 2,
              "note": "Fresh classes and videos; fixed selection independent of model scores."}
    immutable(output / "selection.json", record)
    print("FRESH SELECTION", record, flush=True)
    download_frames([path], output / "images", root / "download_ranges")
    print("Image validation:", selected.validate_images(), flush=True)


def cache(root):
    output = root / "revision_v2/confirmation"
    manifest = VideoManifest(output / "confirmation.json")
    encoder = FrozenEncoder(device="cpu", revision="cfd3195ba4ea9592eec887ded089f4c08eff231d", candidates=128)
    print(build_cache(manifest, output / "cache", encoder, per_class=2, pool_seed=42), flush=True)
    pool = read_json(output / "cache/support_pool.json")
    episodes = make_episodes(manifest, pool, count=30, n_way=2, k_shot=1, seed=20261025)
    immutable(output / "episodes.json", episodes)
    print("Fresh confirmation cache and 30 fixed episodes ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "cache"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    {"prepare": prepare, "cache": cache}[args.stage](args.experiment_dir.resolve())
