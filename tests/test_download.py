import io
import zipfile

import pytest

from adaptive_fsvod.download_frames import RangeReader, frame_index


class Response:
    def __init__(self, payload, start, end, status=206):
        self.payload = payload
        self.start, self.end = start, end
        self.status_code = status
        self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
        self.consumed = False

    def iter_content(self, chunk_size):
        self.consumed = True
        yield self.payload[self.start:self.end + 1]

    def close(self):
        pass


class Session:
    def __init__(self, payload, status=206):
        self.payload, self.status = payload, status
        self.responses = []

    def get(self, url, params, headers, **kwargs):
        start, end = map(int, headers["Range"].split("=")[1].split("-"))
        response = Response(self.payload, start, end, self.status)
        self.responses.append(response)
        return response

    def close(self):
        pass


def zip_payload():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("JPEGImages/video/00000.jpg", b"example RGB bytes")
        archive.writestr("Annotations/video/00000.png", b"mask")
    return buffer.getvalue()


def test_remote_zip_reads_rgb_members_and_reuses_disk_ranges(tmp_path):
    payload = zip_payload()
    session = Session(payload)
    with RangeReader("https://example.test/data.zip", len(payload), tmp_path, block_size=64, session=session) as remote:
        with zipfile.ZipFile(remote) as archive:
            index = frame_index(archive)
            assert list(index) == ["video/00000.jpg"]
            assert archive.read(index["video/00000.jpg"]) == b"example RGB bytes"
    assert len(session.responses) > 0
    second_session = Session(payload)
    with RangeReader("https://example.test/data.zip", len(payload), tmp_path, block_size=64, session=second_session) as remote:
        with zipfile.ZipFile(remote) as archive:
            assert archive.read("JPEGImages/video/00000.jpg") == b"example RGB bytes"
        assert remote.network_bytes == 0
    assert second_session.responses == []


def test_full_download_response_is_rejected_before_reading_body(tmp_path):
    session = Session(b"some large archive", status=200)
    reader = RangeReader("https://example.test/data.zip", 18, tmp_path, session=session)
    with pytest.raises(RuntimeError, match="Full download was refused"):
        reader.read(4)
    assert not session.responses[0].consumed


def test_wrong_content_range_is_rejected(tmp_path):
    session = Session(b"0123456789")
    reader = RangeReader("https://example.test/data.zip", 11, tmp_path, session=session)
    with pytest.raises(RuntimeError, match="exact HTTP byte range"):
        reader.read(4)
