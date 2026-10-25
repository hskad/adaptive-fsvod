"""Download the public final-study bundles and verify every extracted file."""

import argparse
import hashlib
import shutil
import time
import zipfile
from pathlib import Path, PurePosixPath

import requests

from adaptive_fsvod.data import checksum, read_json

RELEASE = "https://github.com/hskad/adaptive-fsvod/releases/download/v1.0-results"


def download(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and checksum(path) == expected:
        return
    temporary = path.with_suffix(".part")
    for attempt in range(3):
        digest = hashlib.sha256()
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with temporary.open("wb") as target:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            digest.update(block)
                            target.write(block)
            if digest.hexdigest() != expected:
                raise ValueError(f"Download checksum failed: {path.name}")
            temporary.replace(path)
            return
        except requests.RequestException as error:
            status = error.response.status_code if error.response is not None else None
            if attempt == 2 or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            print("Network error; retry", attempt + 2, "of 3:", path.name, flush=True)
            time.sleep(attempt + 1)
        finally:
            if temporary.exists():
                temporary.unlink()


def extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in entry.orig_filename or ":" in entry.filename:
                raise ValueError("Unsafe path in artifact")
            path = destination.joinpath(*name.parts).resolve()
            if not path.is_relative_to(destination):
                raise ValueError("Artifact path leaves destination")
            if entry.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            data = bundle.read(entry)  # ZIP CRC is checked here.
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError(f"Refusing to overwrite a different file: {path}")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)


def prepare(root):
    repo = Path(__file__).resolve().parents[1]
    for kind, destination in (("inference", root / "weights"), ("features", root)):
        record = read_json(repo / f"results/score_calibration/{kind}_bundle.manifest.json")
        archive = root / "downloads" / record["file"]
        print("Download and verify", record["file"], flush=True)
        download(f"{RELEASE}/{record['file']}", archive, record["sha256"])
        extract(archive, destination)
        if kind == "inference":
            items = {row["file"]: row for row in record["models"]}
        else:
            items = record["files"]
        for name, item in items.items():
            if checksum(destination / name) != item["sha256"]:
                raise ValueError(f"Extracted artifact differs: {name}")
        print("Verified", len(items), "files", flush=True)
    # Copy the selected configuration without altering historical source records.
    config = repo / "results/revision_v2/configs/fixed_tau060.json"
    target = root / "fixed_tau060.json"
    if target.exists() and checksum(target) != checksum(config):
        raise ValueError("Existing selected configuration differs")
    if not target.exists():
        shutil.copyfile(config, target)
    print("Ready:", root, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    prepare(parser.parse_args().experiment_dir.resolve())
