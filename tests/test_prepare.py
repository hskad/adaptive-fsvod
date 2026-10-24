import json
import zipfile

from adaptive_fsvod.convert_fsytv import convert
from adaptive_fsvod.data import read_json
from adaptive_fsvod.prepare import prepare_ytvis


def test_fsytv_train_test_category_ids_are_mapped_by_name(tmp_path):
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for split, name in (("train", "cat"), ("test", "dog")):
            source = {"categories": [{"id": 0, "name": name}],
                "images": [{"id": 0, "video": name, "file_name": f"{name}/000.jpg",
                            "width": 40, "height": 24, "frame_index": 0}],
                "annotations": [{"image_id": 0, "category_id": 0, "track_id": 0,
                                 "bbox": [1, 2, 3, 4], "iscrowd": 0}]}
            stream.writestr(f"annotations/fsvis_{split}.json", json.dumps(source))
    target = tmp_path / "converted.json"
    summary = convert(archive, target)
    assert summary["classes"] == 2
    result = read_json(target)
    assert len({track["category_id"] for track in result["annotations"]}) == 2


def test_prepare_keeps_class_and_video_splits_disjoint(tmp_path):
    source = {"categories": [{"id": c, "name": str(c)} for c in range(6)], "videos": [], "annotations": []}
    for category in range(6):
        for index in range(2):
            vid = category * 10 + index
            source["videos"].append({"id": vid, "width": 40, "height": 24, "file_names": [f"{vid}/0.jpg", f"{vid}/1.jpg"]})
            source["annotations"].append({"id": vid, "video_id": vid, "category_id": category,
                                          "bboxes": [[1, 2, 3, 4], None]})
    annotation = tmp_path / "source.json"
    annotation.write_text(json.dumps(source))
    summary = prepare_ytvis(annotation, tmp_path, tmp_path / "out", (2, 2, 2),
                             max_videos_per_class=2, frame_stride=1, check_images=False)
    assert all(len(split["classes"]) == 2 for split in summary["splits"].values())
    assert all(len(split["video_ids"]) == 4 for split in summary["splits"].values())
