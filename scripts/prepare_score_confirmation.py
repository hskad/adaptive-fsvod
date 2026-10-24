"""Prepare a new class/video-disjoint confirmation after the validation gate."""

import argparse
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import torch

from adaptive_fsvod.cache import FrozenEncoder, build_cache
from adaptive_fsvod.data import (VideoManifest, checksum, read_json, support_pool,
                                validate_splits)
from adaptive_fsvod.download_frames import SOURCE, RangeReader, download_frames, frame_index
from prepare_confirmation import immutable, prior_confirmation_path
from run_larger_confirmation import balanced_episodes
from run_score_calibration import STUDY, verify


REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"


def prior_manifests(root):
    repo = Path(__file__).resolve().parents[1]
    result = [VideoManifest(root / f"pilot/{split}.json") for split in ("train", "val", "test")]
    result.append(VideoManifest(prior_confirmation_path(root)))
    for local, published in [("revision_v5/confirmation/confirmation.json", "results/revision_v5/confirmation_protocol/confirmation.json"),
                             ("larger_confirmation/confirmation.json", "results/larger_confirmation/protocol/confirmation.json")]:
        path = root / local
        result.append(VideoManifest(path if path.exists() else repo / published))
    return result


def prepare(root, target):
    verify(root)
    decision = read_json(root / STUDY / "validation_results.json")
    if not decision["validation_gate_passed"]:
        raise ValueError("The validation gate failed. No fresh test is authorized by this study.")
    directory = root / STUDY / "confirmation"
    annotation = root / "raw/fsytv-boxes.json"
    source = read_json(annotation)
    originals = prior_manifests(root)
    excluded_classes = set().union(*(set(m.classes) for m in originals))
    excluded_videos = set().union(*(set(m.videos) for m in originals))
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
        if not files:
            continue
        n = min(8, len(files))
        indices = [round(i * (len(files) - 1) / (n - 1)) for i in range(n)] if n > 1 else [0]
        if any(files[i] not in available for i in indices):
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
                    objects.append({"category_id": str(track["category_id"]), "bbox": box,
                                    "track_id": str(track["id"])})
            frames.append({"path": files[index], "source_index": index, "width": video["width"],
                           "height": video["height"], "objects": objects})
        if any(f["objects"] for f in frames):
            eligible[next(iter(present))].append({"id": vid, "frames": frames})
    rng = random.Random(20261031)
    classes = sorted(c for c, videos in eligible.items() if len(videos) >= 4)
    rng.shuffle(classes)
    classes = classes[:6]
    if len(classes) != 6:
        raise ValueError("Insufficient eligible unused classes")
    videos = []
    for category in classes:
        choices = eligible[category].copy()
        rng.shuffle(choices)
        videos.extend(choices[:4])
    names = {str(c["id"]): c["name"] for c in source["categories"]}
    path = directory / "confirmation.json"
    immutable(path, {"schema_version": 1, "dataset": "FSYTV-40-objectness-confirmation",
              "split": "score_confirmation", "images_root": "images",
              "classes": {c: names[c] for c in sorted(classes)}, "videos": sorted(videos, key=lambda v: v["id"])})
    manifest = VideoManifest(path)
    validate_splits(originals + [manifest])
    pool = support_pool(manifest, per_class=4, seed=42)
    if any(len({e["video_id"] for e in values}) != 4 for values in pool.values()):
        raise ValueError("Expected one support crop per video")
    immutable(directory / "support_pool.json", pool)
    immutable(directory / "episodes.json", balanced_episodes(manifest, pool, seed=20261101))
    protocol = {"selection_seed": 20261031, "episode_seed": 20261101, "pool_seed": 42,
                "bootstrap_seed": 20261102, "bootstrap_draws": 1000,
                "selected_beta": decision["selected_beta"], "classes": manifest.classes,
                "videos": len(videos), "frames": sum(len(v["frames"]) for v in videos),
                "excluded_manifests": [{"path": str(m.path), "sha256": checksum(m.path), "classes": m.classes}
                                       for m in originals],
                "annotation_sha256": checksum(annotation), "source": SOURCE,
                "encoder": "google/owlv2-base-patch16-ensemble", "revision": REVISION,
                "candidates": 128, "feature_dim": 512, "attention_backend": "sdpa", "support_per_class": 4,
                "validation_results_sha256": checksum(root / STUDY / "validation_results.json"),
                "model_protocol_sha256": checksum(root / STUDY / "protocol.json"),
                "files": {name: checksum(directory / name) for name in
                          ("confirmation.json", "support_pool.json", "episodes.json")}}
    immutable(directory / "selection.json", protocol)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("confirmation.json", "support_pool.json", "episodes.json", "selection.json"):
        shutil.copyfile(directory / name, target / name)
    print("UNUSED-CLASS PROTOCOL", protocol, flush=True)


def verify_fresh(root):
    verify(root)
    directory = root / STUDY / "confirmation"
    protocol = read_json(directory / "selection.json")
    for name, digest in protocol["files"].items():
        if checksum(directory / name) != digest:
            raise ValueError("Frozen test input changed")
    if checksum(root / STUDY / "validation_results.json") != protocol["validation_results_sha256"]:
        raise ValueError("Validation selection changed")
    if checksum(root / STUDY / "protocol.json") != protocol["model_protocol_sha256"]:
        raise ValueError("Frozen model protocol changed")
    validate_splits(prior_manifests(root) + [VideoManifest(directory / "confirmation.json")])
    return protocol


def download(root):
    verify_fresh(root)
    directory = root / STUDY / "confirmation"
    download_frames([directory / "confirmation.json"], directory / "images", root / "download_ranges")
    print("IMAGE CHECK", VideoManifest(directory / "confirmation.json").validate_images(), flush=True)


def cache(root, device):
    verify_fresh(root)
    directory = root / STUDY / "confirmation"
    metadata_path = directory / "cache/metadata.json"
    if metadata_path.exists() and read_json(metadata_path).get("complete"):
        from adaptive_fsvod.data import frame_key
        manifest = VideoManifest(directory / "confirmation.json")
        existing = read_json(metadata_path)
        expected = {"manifest_sha256": checksum(manifest.path), "encoder": "google/owlv2-base-patch16-ensemble",
                    "revision": REVISION, "candidates": 128, "support_per_class": 4,
                    "pool_seed": 42, "synthetic": False, "attention_backend": "sdpa"}
        if any(existing.get(k) != v for k, v in expected.items()):
            raise ValueError("Existing complete cache uses different settings")
        pool = read_json(directory / "cache/support_pool.json")
        needed = [directory / "cache/frames" / f"{frame_key(vid, frame)}.pt"
                  for vid, video in manifest.videos.items() for frame in video["frames"]]
        needed += [directory / "cache/support" / f"{item['key']}.pt" for values in pool.values() for item in values]
        if all(p.is_file() for p in needed) and pool == read_json(directory / "support_pool.json"):
            print("Complete matching cache already exists; preserve its metadata and tensors", flush=True)
            return
    encoder = FrozenEncoder(device=device, revision=REVISION, candidates=128, attention_backend="sdpa")
    metadata = build_cache(VideoManifest(directory / "confirmation.json"), directory / "cache", encoder,
                           per_class=4, pool_seed=42)
    if read_json(directory / "cache/support_pool.json") != read_json(directory / "support_pool.json"):
        raise ValueError("Cached support differs from the frozen pool")
    print("COMPLETE FRESH CACHE", metadata, flush=True)


def freeze_cache(root, target):
    protocol = verify_fresh(root)
    directory = root / STUDY / "confirmation"
    cached = directory / "cache"
    metadata = read_json(cached / "metadata.json")
    files = sorted(cached.rglob("*.pt"))
    expected = protocol["frames"] + 4 * len(protocol["classes"])
    if not metadata.get("complete") or metadata["synthetic"] or len(files) != expected:
        raise ValueError("Feature cache is incomplete")
    record = {"selection_sha256": checksum(directory / "selection.json"),
              "metadata_sha256": checksum(cached / "metadata.json"),
              "support_pool_sha256": checksum(cached / "support_pool.json"),
              "files": {p.relative_to(cached).as_posix(): checksum(p) for p in files}}
    immutable(directory / "cache_artifact_sha256.json", record)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(directory / "cache_artifact_sha256.json", target / "cache_artifact_sha256.json")
    output = target / "cache_metadata"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "support_pool.json"):
        shutil.copyfile(cached / name, output / name)
    shutil.copyfile(directory / "images/download_manifest.json", target / "download_manifest.json")
    print("FROZEN", len(files), "new feature tensors before scoring", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "download", "cache", "freeze-cache"), required=True)
    parser.add_argument("--output", type=Path, default=Path("results/score_calibration/confirmation_protocol"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.experiment_dir.resolve()
    if args.stage == "prepare":
        prepare(root, args.output)
    elif args.stage == "cache":
        cache(root, args.device)
    elif args.stage == "freeze-cache":
        freeze_cache(root, args.output)
    else:
        download(root)
