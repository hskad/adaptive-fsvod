"""Prepare and score the predeclared larger confirmation without training."""

import argparse
import random
import shutil
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path

import torch

from adaptive_fsvod.cache import FrozenEncoder, build_cache
from adaptive_fsvod.data import (VideoManifest, checksum, read_json, support_pool,
                                validate_episodes, validate_splits, write_json)
from adaptive_fsvod.download_frames import SOURCE, RangeReader, download_frames, frame_index
from adaptive_fsvod.train import environment
from prepare_confirmation import immutable, prior_confirmation_path
from run_revision import measure


STUDY = "larger_confirmation"
ENCODER_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"


def exclusions(root):
    return [VideoManifest(root / f"pilot/{s}.json") for s in ("train", "val", "test")] + [
        VideoManifest(prior_confirmation_path(root)),
        VideoManifest(root / "revision_v5/confirmation/confirmation.json")]


def balanced_episodes(manifest, pool):
    rng = random.Random(20261029)
    episodes = []
    for vid, video in sorted(manifest.videos.items()):
        present = {str(o["category_id"]) for f in video["frames"] for o in f["objects"]}
        if len(present) != 1:
            raise ValueError("Expected a single-class video")
        anchor = next(iter(present))
        for distractor in sorted(set(manifest.classes) - {anchor}):
            classes = sorted([anchor, distractor])
            support = {c: [rng.choice([x for x in pool[c] if x["video_id"] != vid])]
                       for c in classes}
            episodes.append({"id": len(episodes), "video_id": vid, "classes": classes,
                             "support": support, "frame_indices": list(range(len(video["frames"])))})
    result = {"manifest_sha256": checksum(manifest.path), "seed": 20261029,
              "n_way": 2, "k_shot": 1, "episodes": episodes}
    validate_episodes(manifest, result)
    return result


def model_rows(root):
    rows = []
    for study, methods in [("revision_v5", {"fixed": "fixed_50", "centered": "centered_50"}),
                           ("revision_v2", {"fixed": "fixed_300"})]:
        for row in read_json(root / study / "replications.json"):
            if row["method"] not in methods:
                continue
            checkpoint, config = Path(row["checkpoint"]), Path(row["config"])
            if checksum(checkpoint) != row["checkpoint_sha256"] or checksum(config) != row["config_sha256"]:
                raise ValueError("Frozen source model or configuration changed")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            actual = saved["arguments"].get("stop_after") or saved["arguments"]["steps"]
            expected = 50 if study == "revision_v5" else 300
            if actual != expected or saved["model_config"].get("feature_centering", False) != (row["method"] == "centered"):
                raise ValueError("Source model does not match the declared family")
            rows.append({**row, "method": methods[row["method"]], "training_budget": actual,
                         "checkpoint_step": saved["step"], "source_study": study})
    if len(rows) != 9:
        raise ValueError("Expected all nine source models")
    return rows


def prepare(root):
    directory = root / STUDY
    annotation = root / "raw/fsytv-boxes.json"
    source = read_json(annotation)
    originals = exclusions(root)
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
    rng = random.Random(20261028)
    classes = sorted(c for c, videos in eligible.items() if len(videos) >= 4)
    rng.shuffle(classes)
    classes = classes[:6]
    if len(classes) != 6:
        raise ValueError("Insufficient eligible fresh classes")
    videos = []
    for category in classes:
        candidates = eligible[category].copy()
        rng.shuffle(candidates)
        videos.extend(candidates[:4])
    names = {str(c["id"]): c["name"] for c in source["categories"]}
    manifest_path = directory / "confirmation.json"
    immutable(manifest_path, {"schema_version": 1, "dataset": "FSYTV-40-larger-confirmation",
              "split": "larger_confirmation", "images_root": "images",
              "classes": {c: names[c] for c in sorted(classes)},
              "videos": sorted(videos, key=lambda v: v["id"])})
    manifest = VideoManifest(manifest_path)
    validate_splits(originals + [manifest])
    pool = support_pool(manifest, per_class=4, seed=42)
    for values in pool.values():
        if len({x["video_id"] for x in values}) != 4:
            raise ValueError("Each support pool must cover all four videos")
    immutable(directory / "support_pool.json", pool)
    immutable(directory / "episodes.json", balanced_episodes(manifest, pool))
    rows = model_rows(root)
    immutable(directory / "models.json", rows)
    record = {"selection_seed": 20261028, "episode_seed": 20261029, "pool_seed": 42,
              "bootstrap_seed": 20261030, "bootstrap_draws": 1000,
              "annotation_sha256": checksum(annotation), "source": SOURCE,
              "excluded_manifests": [{"path": str(m.path), "sha256": checksum(m.path),
                                      "classes": m.classes} for m in originals],
              "classes": manifest.classes, "video_ids": sorted(manifest.videos),
              "videos": len(videos), "frames": sum(len(v["frames"]) for v in videos),
              "episodes": 120, "support_per_class": 4,
              "encoder": "google/owlv2-base-patch16-ensemble", "revision": ENCODER_REVISION,
              "candidates": 128, "feature_dim": 512,
              "frozen_files": {name: checksum(directory / name) for name in
                               ("confirmation.json", "episodes.json", "support_pool.json", "models.json")}}
    immutable(directory / "selection.json", record)
    print("FROZEN SELECTION", record, flush=True)


def verify(root):
    directory = root / STUDY
    selection = read_json(directory / "selection.json")
    for name, digest in selection["frozen_files"].items():
        if checksum(directory / name) != digest:
            raise ValueError(f"Frozen protocol changed: {name}")
    validate_splits(exclusions(root) + [VideoManifest(directory / "confirmation.json")])
    for row in read_json(directory / "models.json"):
        if checksum(row["checkpoint"]) != row["checkpoint_sha256"] or checksum(row["config"]) != row["config_sha256"]:
            raise ValueError("Frozen checkpoint or configuration changed")


def download(root):
    verify(root)
    directory = root / STUDY
    download_frames([directory / "confirmation.json"], directory / "images", root / "download_ranges")
    print(VideoManifest(directory / "confirmation.json").validate_images(), flush=True)


def cache(root, device, backend="sdpa"):
    verify(root)
    directory = root / STUDY
    manifest = VideoManifest(directory / "confirmation.json")
    # A separate directory keeps the incomplete native-kernel timing trial intact.
    immutable(directory / "extraction_settings.json", {"attention_backend": backend,
              "device": device, "cache_directory": "cache_sdpa" if backend == "sdpa" else "cache",
              "note": "Equivalent attention equation; fp32 kernel may differ in rounding. Shared by all nine models."})
    cache_path = cache_directory(root)
    encoder = FrozenEncoder(device=device, revision=ENCODER_REVISION, candidates=128, attention_backend=backend)
    metadata = build_cache(manifest, cache_path, encoder, per_class=4, pool_seed=42)
    if read_json(cache_path / "support_pool.json") != read_json(directory / "support_pool.json"):
        raise ValueError("Cache support pool differs from the frozen protocol")
    print("COMPLETE CACHE", metadata, flush=True)


def cache_directory(root):
    directory = root / STUDY
    return directory / read_json(directory / "extraction_settings.json")["cache_directory"]


def score(root):
    verify(root)
    directory = root / STUDY
    frozen_cache = read_json(directory / "cache_artifact_sha256.json")
    if checksum(directory / "extraction_settings.json") != frozen_cache["extraction_settings_sha256"]:
        raise ValueError("Frozen extraction settings changed")
    for name, digest in frozen_cache["files"].items():
        if checksum(cache_directory(root) / name) != digest:
            raise ValueError("Frozen feature cache changed")
    for name, key in (("metadata.json", "metadata_sha256"), ("support_pool.json", "support_pool_sha256")):
        if checksum(cache_directory(root) / name) != frozen_cache[key]:
            raise ValueError("Frozen cache metadata changed")
    execution = {"environment": environment(), "threads": torch.get_num_threads(),
                 "cache_artifact_manifest_sha256": checksum(directory / "cache_artifact_sha256.json")}
    environment_path = directory / "evaluation_environment.json"
    if not environment_path.exists():
        immutable(environment_path, execution)
    elif read_json(environment_path)["cache_artifact_manifest_sha256"] != execution["cache_artifact_manifest_sha256"]:
        raise ValueError("Previous evaluation used different frozen features")
    rows = []
    for row in read_json(directory / "models.json"):
        output = directory / "evaluations" / f"{row['method']}_seed{row['seed']}"
        if all((output / name).exists() for name in ("metrics.json", "ground_truth.json", "predictions.json")):
            metrics = read_json(output / "metrics.json")
            expected = {"checkpoint_sha256": row["checkpoint_sha256"],
                        "manifest_sha256": checksum(directory / "confirmation.json"),
                        "episodes_sha256": checksum(directory / "episodes.json")}
            if any(metrics.get(k) != v for k, v in expected.items()):
                raise ValueError("Stored evaluation used a different model or protocol")
        else:
            metrics = measure(row["checkpoint"], directory / "confirmation.json", cache_directory(root),
                              directory / "episodes.json", output)
            metrics["evaluation_environment"] = execution
            metrics["encoder_attention_backend"] = read_json(directory / "extraction_settings.json")["attention_backend"]
            write_json(output / "metrics.json", metrics)
        rows.append({"method": row["method"], "seed": row["seed"], **metrics})
    summary = {}
    for method in ("fixed_50", "centered_50", "fixed_300"):
        values = [r for r in rows if r["method"] == method]
        summary[method] = {key: {"mean": statistics.mean(r[key] for r in values),
                                "sample_std": statistics.stdev(r[key] for r in values)}
                           for key in ("AP", "AP50", "AP75")}
    write_json(directory / "results.json", {"rows": rows, "summary": summary})
    print("LARGER CONFIRMATION", summary, flush=True)


def freeze_cache(root, target):
    verify(root)
    directory = root / STUDY
    cache_path = cache_directory(root)
    metadata = read_json(cache_path / "metadata.json")
    if not metadata.get("complete") or metadata["synthetic"] or metadata["feature_dim"] != 512:
        raise ValueError("Expected a complete real-data feature cache")
    paths = sorted(cache_path.rglob("*.pt"))
    if len(paths) != 216:
        raise ValueError("Expected 192 query tensors and 24 support tensors")
    record = {"extraction_settings_sha256": checksum(directory / "extraction_settings.json"),
              "files": {path.relative_to(cache_path).as_posix(): checksum(path) for path in paths},
              "metadata_sha256": checksum(cache_path / "metadata.json"),
              "support_pool_sha256": checksum(cache_path / "support_pool.json")}
    immutable(directory / "cache_artifact_sha256.json", record)
    target.mkdir(parents=True, exist_ok=True)
    for name in ("cache_artifact_sha256.json", "extraction_settings.json"):
        shutil.copyfile(directory / name, target / name)
    output = target / "cache_metadata"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "support_pool.json"):
        shutil.copyfile(cache_path / name, output / name)
    print("Frozen", len(paths), "feature tensors before model scoring", flush=True)


def publish_protocol(root, target):
    verify(root)
    directory = root / STUDY
    target.mkdir(parents=True, exist_ok=True)
    for name in ("confirmation.json", "episodes.json", "support_pool.json", "models.json", "selection.json"):
        shutil.copyfile(directory / name, target / name)
    print("Copied frozen protocol to", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "download", "cache", "freeze-cache", "evaluate", "publish-protocol"), required=True)
    parser.add_argument("--output", type=Path, default=Path("results/larger_confirmation/protocol"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attention-backend", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    root = args.experiment_dir.resolve()
    if args.stage == "cache":
        cache(root, args.device, args.attention_backend)
    elif args.stage == "publish-protocol":
        publish_protocol(root, args.output)
    elif args.stage == "freeze-cache":
        freeze_cache(root, args.output)
    else:
        {"prepare": prepare, "download": download, "evaluate": score}[args.stage](root)
