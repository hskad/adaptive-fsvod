"""A damaged or unsafe public artifact must not replace user files."""

import hashlib
import zipfile

import pytest

from scripts.download_artifacts import download, extract
from scripts.export_reproduction_features import compact_tensor_file


def test_verified_archive_extracts_and_reuses_without_overwrite(tmp_path):
    archive = tmp_path / "valid.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("nested/model.pt", b"model bytes")
    target = tmp_path / "output"
    extract(archive, target)
    extract(archive, target)
    path = target / "nested/model.pt"
    path.write_bytes(b"user file")
    with pytest.raises(ValueError, match="overwrite"):
        extract(archive, target)
    assert path.read_bytes() == b"user file"


@pytest.mark.parametrize("name", ["../outside.pt", "/absolute.pt", "C:/outside.pt", "nested\\outside.pt"])
def test_archive_rejects_unsafe_paths(tmp_path, name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(name.replace("\\", "/"), b"bad")
    # Supply a raw ZIP backslash on both platforms, bypassing write normalization.
    if "\\" in name:
        archive.write_bytes(archive.read_bytes().replace(name.replace("\\", "/").encode(), name.encode()))
    with pytest.raises(ValueError, match="Unsafe"):
        extract(archive, tmp_path / "output")
    assert not (tmp_path / "outside.pt").exists()


def test_bad_download_is_rejected_and_original_file_preserved(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, _):
            yield b"damaged download"

    monkeypatch.setattr("scripts.download_artifacts.requests.get", lambda *args, **kwargs: Response())
    path = tmp_path / "model.zip"
    path.write_bytes(b"existing file")
    with pytest.raises(ValueError, match="checksum"):
        download("https://example.test/weights.zip", path, hashlib.sha256(b"expected").hexdigest())
    assert path.read_bytes() == b"existing file"
    assert not path.with_suffix(".part").exists()


@pytest.mark.parametrize("as_dictionary", [False, True])
def test_feature_export_preserves_values_without_unused_storage(tmp_path, as_dictionary):
    import io
    import torch

    original = torch.arange(51200, dtype=torch.float32)[:512]
    path = tmp_path / "features.pt"
    torch.save({"features": original} if as_dictionary else original, path)
    restored = torch.load(io.BytesIO(compact_tensor_file(path)), weights_only=True)
    value = restored["features"] if as_dictionary else restored
    assert torch.equal(value, original)
    assert value.untyped_storage().nbytes() == value.numel() * value.element_size()
    assert len(compact_tensor_file(path)) < path.stat().st_size / 10
