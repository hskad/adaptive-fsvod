"""Fetch selected RGB frames from a pinned, public YouTube-VOS ZIP mirror.

FSYTV-40 uses videos derived from YouTube-VOS. We verify exact frame paths,
image dimensions, and archive CRCs before accepting this alternative source.
HTTP Range requests avoid downloading the full 8.96 GB archive.
"""

import argparse
import hashlib
import io
import re
import zipfile
import zlib
from pathlib import Path, PurePosixPath

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from .data import VideoManifest, checksum, write_json


SOURCE = {
    "dataset": "YouTube-VOS training RGB frames",
    "mirror": "JiaHuang01/Youtubevos_train",
    "revision": "7855adf7e2e8e9386b578f58d490f25c1886368d",
    "archive": "Youtube_vos_train.zip",
    "archive_bytes": 8957125671,
}
SOURCE["url"] = (f"https://huggingface.co/datasets/{SOURCE['mirror']}/resolve/"
                 f"{SOURCE['revision']}/{SOURCE['archive']}")


class RangeReader(io.RawIOBase):
    """A seekable ZIP input that refuses full-file HTTP responses."""

    def __init__(self, url, size, cache_directory, block_size=262144, session=None):
        self.url, self.size, self.position = url, size, 0
        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)
        self.source_id = hashlib.sha256(f"{url}|{size}".encode()).hexdigest()[:20]
        self.block_size = block_size
        self.buffer, self.buffer_start = b"", 0
        self.network_bytes, self.network_requests = 0, 0
        self.session = session or requests.Session()
        if session is None:
            retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
            self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek mode")
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def _fetch(self, start, end):
        path = self.cache_directory / f"{self.source_id}_{start}_{end}.bin"
        expected = end - start + 1
        if path.exists() and path.stat().st_size == expected:
            return path.read_bytes()
        # A distinct query prevents shared HTTP caches from reusing another range.
        response = self.session.get(self.url, params={"range": f"{start}-{end}"},
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
            timeout=(15, 90), stream=True)
        try:
            interval = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            actual = tuple(map(int, interval.groups())) if interval else None
            if response.status_code != 206 or actual != (start, end, self.size):
                raise RuntimeError("Server did not honor the exact HTTP byte range. Full download was refused.")
            chunks, received = [], 0
            for chunk in response.iter_content(chunk_size=262144):
                received += len(chunk)
                if received > expected:
                    raise RuntimeError("Range response exceeded the requested size")
                chunks.append(chunk)
            data = b"".join(chunks)
            if len(data) != expected:
                raise RuntimeError("Incomplete byte range. Repeat the download command.")
        finally:
            response.close()
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        self.network_bytes += len(data)
        self.network_requests += 1
        return data

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        offset = self.position - self.buffer_start
        if offset < 0 or offset + size > len(self.buffer):
            end = min(self.size - 1, self.position + max(size, self.block_size) - 1)
            self.buffer_start = self.position
            self.buffer = self._fetch(self.position, end)
            offset = 0
        value = self.buffer[offset:offset + size]
        self.position += len(value)
        return value

    def close(self):
        self.session.close()
        super().close()


def frame_index(archive):
    result = {}
    for entry in archive.infolist():
        parts = PurePosixPath(entry.filename).parts
        if entry.is_dir() or "JPEGImages" not in parts:
            continue
        relative = "/".join(parts[parts.index("JPEGImages") + 1:])
        if relative in result:
            raise ValueError(f"Ambiguous RGB frame in archive: {relative}")
        result[relative] = entry
    return result


def download_frames(manifest_paths, images_root, range_cache, check_only=False):
    images_root = Path(images_root).resolve()
    expected = {}
    for path in manifest_paths:
        manifest = VideoManifest(path)
        for video in manifest.videos.values():
            for frame in video["frames"]:
                relative = PurePosixPath(frame["path"])
                target = (images_root / str(relative)).resolve()
                if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(images_root):
                    raise ValueError("Frame path must stay inside the image root")
                dimensions = (frame["width"], frame["height"])
                if str(relative) in expected and expected[str(relative)] != dimensions:
                    raise ValueError("Inconsistent frame dimensions")
                expected[str(relative)] = dimensions
    reader = RangeReader(SOURCE["url"], SOURCE["archive_bytes"], range_cache)
    records = []
    with reader, zipfile.ZipFile(reader) as archive:
        index = frame_index(archive)
        missing = sorted(set(expected) - set(index))
        if missing:
            raise ValueError(f"Archive is missing {len(missing)} selected frames. Examples: {missing[:5]}")
        print(f"Matched all {len(expected)} selected frame paths in the RGB archive.", flush=True)
        if not check_only:
            for relative, dimensions in tqdm(sorted(expected.items()), desc="Download and verify RGB frames"):
                info = index[relative]
                if info.file_size > 20_000_000:
                    raise ValueError("Unexpectedly large image in archive")
                destination = images_root / relative
                payload = destination.read_bytes() if destination.exists() else None
                if payload is None or zlib.crc32(payload) != info.CRC:
                    payload = archive.read(info)
                with Image.open(io.BytesIO(payload)) as image:
                    if image.size != dimensions or image.mode not in ("RGB", "L"):
                        raise ValueError(f"Frame dimensions or image mode differ from labels: {relative}")
                    image.load()
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists() or zlib.crc32(destination.read_bytes()) != info.CRC:
                    temporary = destination.with_suffix(destination.suffix + ".tmp")
                    temporary.write_bytes(payload)
                    temporary.replace(destination)
                records.append({"path": relative, "width": dimensions[0], "height": dimensions[1],
                                "sha256": hashlib.sha256(payload).hexdigest(), "crc32": info.CRC,
                                "archive_member": info.filename, "bytes": len(payload)})
        summary = {"source": SOURCE, "manifests": {Path(p).name: checksum(p) for p in manifest_paths},
                   "matched_frames": len(expected), "verified_frames": len(records),
                   "check_only": check_only, "network_bytes_this_run": reader.network_bytes,
                   "network_requests_this_run": reader.network_requests, "frames": records}
    if not check_only:
        write_json(images_root / "download_manifest.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--range-cache", default="cache/download_ranges")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = download_frames(args.manifests, args.images_root, args.range_cache, args.check_only)
    print({k: v for k, v in result.items() if k not in ("frames", "manifests")})


if __name__ == "__main__":
    main()
