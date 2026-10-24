"""Convert labeled YouTube-VIS 2019 annotations to disjoint manifests."""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .data import VideoManifest, checksum, read_json, validate_splits, write_json


def prepare_ytvis(annotation, images_root, output, class_counts=(20, 10, 10),
                  max_videos_per_class=6, frame_stride=4, seed=42, check_images=True):
    source = read_json(annotation)
    if not source.get("videos") or not source.get("annotations"):
        raise ValueError("Expected labeled YouTube-VIS annotations with videos and track annotations")
    rng = random.Random(seed)
    categories = sorted(source["categories"], key=lambda c: c["id"])
    rng.shuffle(categories)
    if sum(class_counts) > len(categories) or min(class_counts) < 1:
        raise ValueError("Invalid class split sizes")
    if frame_stride < 1 or max_videos_per_class < 2:
        raise ValueError("Need a positive frame stride and at least two videos per class")
    groups = {}
    offset = 0
    for split, count in zip(("train", "val", "test"), class_counts):
        groups[split] = {str(c["id"]): c["name"] for c in categories[offset:offset + count]}
        offset += count
    video_tracks = defaultdict(list)
    for track in source["annotations"]:
        if not track.get("iscrowd", 0):
            video_tracks[str(track["video_id"])].append(track)
    videos = list(source["videos"])
    rng.shuffle(videos)
    counts = {s: defaultdict(int) for s in groups}
    selected = {s: [] for s in groups}
    excluded_mixed = 0
    for video in videos:
        vid = str(video["id"])
        tracks = video_tracks[vid]
        present = {str(t["category_id"]) for t in tracks}
        owners = [s for s, classes in groups.items() if present & set(classes)]
        # A multi-class video can never occur in more than one class split.
        if len(owners) != 1:
            excluded_mixed += bool(owners)
            continue
        split = owners[0]
        assigned = present & set(groups[split])
        if not any(counts[split][c] < max_videos_per_class for c in assigned):
            continue
        frames = []
        for index in range(0, len(video["file_names"]), frame_stride):
            objects = []
            for track in tracks:
                category = str(track["category_id"])
                if category not in groups[split] or index >= len(track["bboxes"]):
                    continue
                box = track["bboxes"][index]
                if box is None:
                    continue
                x, y, width, height = box
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(video["width"], x + width), min(video["height"], y + height)
                if x2 > x1 and y2 > y1:
                    objects.append({"category_id": category, "bbox": [x1, y1, x2, y2],
                                    "track_id": str(track["id"])})
            frames.append({"path": video["file_names"][index], "source_index": index,
                           "width": video["width"], "height": video["height"], "objects": objects})
        if frames and any(f["objects"] for f in frames):
            selected[split].append({"id": vid, "frames": frames})
            for c in assigned:
                counts[split][c] += 1
    manifests = []
    dataset = source.get("source", {}).get("dataset", "YouTube-VIS-2019") + "-subset"
    summary = {"source_sha256": checksum(annotation), "source": source.get("source"), "seed": seed,
               "frame_stride": frame_stride, "excluded_cross_split_videos": excluded_mixed,
               "max_videos_per_class": max_videos_per_class, "splits": {}}
    for split, split_videos in selected.items():
        present_counts = defaultdict(set)
        for video in split_videos:
            for frame in video["frames"]:
                for obj in frame["objects"]:
                    present_counts[obj["category_id"]].add(video["id"])
        usable = {c for c, vids in present_counts.items() if len(vids) >= 2}
        for video in split_videos:
            for frame in video["frames"]:
                frame["objects"] = [o for o in frame["objects"] if o["category_id"] in usable]
        split_videos = [v for v in split_videos if any(f["objects"] for f in v["frames"])]
        if not usable:
            raise ValueError(f"No usable {split} classes. Increase the subset or change the split seed.")
        path = Path(output) / f"{split}.json"
        write_json(path, {"schema_version": 1, "dataset": dataset,
                          "split": split, "images_root": str(Path(images_root).resolve()),
                          "classes": {c: groups[split][c] for c in sorted(usable)},
                          "videos": sorted(split_videos, key=lambda v: v["id"])})
        manifest = VideoManifest(path)
        manifests.append(manifest)
        summary["splits"][split] = {"classes": manifest.classes, "video_ids": sorted(manifest.videos),
            "frames": sum(len(v["frames"]) for v in manifest.videos.values()),
            "manifest_sha256": checksum(path)}
        if check_images:
            manifest.validate_images()
    validate_splits(manifests)
    write_json(Path(output) / "subset.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-counts", nargs=3, type=int, default=[20, 10, 10])
    parser.add_argument("--max-videos-per-class", type=int, default=6)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-image-check", action="store_true")
    args = parser.parse_args()
    summary = prepare_ytvis(args.annotation, args.images_root, args.output, args.class_counts,
                            args.max_videos_per_class, args.frame_stride, args.seed,
                            not args.skip_image_check)
    for split, info in summary["splits"].items():
        print(split, "classes:", len(info["classes"]), "videos:", len(info["video_ids"]), "frames:", info["frames"])


if __name__ == "__main__":
    main()
