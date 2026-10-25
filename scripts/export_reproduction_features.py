"""Bundle the measured frozen features for CPU-only reproduction; no RGB images."""

import argparse
import io
import zipfile
from pathlib import Path

from adaptive_fsvod.data import VideoManifest, checksum, frame_key, read_json, write_json

import torch


def compact_tensor_file(path):
    """Remove unused backing storage while retaining every tensor value exactly."""
    original = torch.load(path, map_location="cpu", weights_only=True)
    compact = ({key: value.clone() for key, value in original.items()}
               if isinstance(original, dict) else original.clone())
    stream = io.BytesIO()
    torch.save(compact, stream)
    restored = torch.load(io.BytesIO(stream.getvalue()), map_location="cpu", weights_only=True)
    pairs = [(original, restored)] if isinstance(original, torch.Tensor) else [(original[k], restored[k]) for k in original]
    if not all(torch.equal(a, b) for a, b in pairs):
        raise ValueError(f"Feature compaction changed tensor values: {path}")
    return stream.getvalue()


def export(root):
    repo = Path(__file__).resolve().parents[1]
    files = {}
    for split in ("train", "val"):
        manifest_path = root / f"pilot/{split}.json"
        manifest = VideoManifest(manifest_path)
        cache = root / f"cache/{split}"
        metadata = read_json(cache / "metadata.json")
        if not metadata.get("complete") or metadata["manifest_sha256"] != checksum(manifest_path):
            raise ValueError("Incomplete or mismatched training feature cache")
        source = read_json(repo / "results/revision_v2/source_runs/fixed_tau060_seed42/config.json")
        for name in (f"{split}.json", f"{split}_episodes.json"):
            path = root / "pilot" / name
            key = f"{split}_manifest_sha256" if name == f"{split}.json" else f"{split}_episodes_sha256"
            if checksum(path) != source[key]:
                raise ValueError(f"Training input changed: {path}")
            files[path.relative_to(root).as_posix()] = path
        paths = [cache / "metadata.json", cache / "support_pool.json"]
        paths += [cache / "frames" / f"{frame_key(vid, frame)}.pt"
                  for vid, video in manifest.videos.items() for frame in video["frames"]]
        paths += [cache / "support" / f"{item['key']}.pt"
                  for values in read_json(cache / "support_pool.json").values() for item in values]
        for path in paths:
            files[path.relative_to(root).as_posix()] = path
    # Preserve the original exclusion manifest; no original-test features are needed.
    files["pilot/test.json"] = root / "pilot/test.json"
    base = root / "score_calibration/confirmation"
    frozen = read_json(repo / "results/score_calibration/confirmation_protocol/cache_artifact_sha256.json")
    for name, expected in frozen["files"].items():
        path = base / "cache" / name
        if checksum(path) != expected:
            raise ValueError(f"Frozen confirmation feature changed: {name}")
        files[path.relative_to(root).as_posix()] = path
    for name in ("metadata.json", "support_pool.json"):
        path = base / "cache" / name
        key = "metadata_sha256" if name == "metadata.json" else "support_pool_sha256"
        if checksum(path) != frozen[key]:
            raise ValueError(f"Frozen confirmation metadata changed: {name}")
        files[path.relative_to(root).as_posix()] = path
    for name in ("confirmation.json", "episodes.json"):
        path = base / name
        if checksum(path) != checksum(repo / "results/score_calibration/confirmation_protocol" / name):
            raise ValueError(f"Published confirmation input changed: {name}")
        files[path.relative_to(root).as_posix()] = path
    records = {}
    manifest = {"description": "Measured frozen OWLv2 features, manifests, and fixed episodes; no RGB images or encoder weights",
                "dataset_terms": "https://youtube-vos.org/dataset/term/",
                "source": "https://huggingface.co/datasets/JiaHuang01/Youtubevos_train",
                "storage": "Tensor files are lossless storage-compacted exports. Every loaded tensor was checked with torch.equal after serialization. Source byte hashes preserve the link to the measured cache.",
                "files": records}
    archive = root / "exports/score_calibration_features.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    import json
    import hashlib
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, path in sorted(files.items()):
            data = compact_tensor_file(path) if path.suffix == ".pt" else path.read_bytes()
            records[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                             "source_sha256": checksum(path), "source_bytes": path.stat().st_size,
                             "tensor_values_verified_equal": path.suffix == ".pt"}
            bundle.writestr(name, data)
        bundle.writestr("feature_manifest.json", json.dumps(manifest, indent=2) + "\n")
    record = {"file": archive.name, "bytes": archive.stat().st_size, "sha256": checksum(archive), **manifest}
    write_json(archive.with_suffix(".manifest.json"), record)
    print(archive, record["bytes"], record["sha256"], "files", len(files), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    torch.set_num_threads(1)
    export(parser.parse_args().experiment_dir.resolve())
