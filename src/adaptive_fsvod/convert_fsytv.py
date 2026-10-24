"""Strip FSYTV COCO frame annotations to compact YouTube-VIS box tracks.

Streaming avoids holding the original mask segmentations in memory. Category
IDs are mapped by name because original train/test IDs use separate spaces.
"""

import argparse
import zipfile
from collections import defaultdict

import ijson

from .data import checksum, write_json


def convert(archive, output):
    with zipfile.ZipFile(archive) as source:
        members = ["annotations/fsvis_train.json", "annotations/fsvis_test.json"]
        category_lists = {}
        for member in members:
            with source.open(member) as stream:
                category_lists[member] = {str(c["id"]): c["name"] for c in ijson.items(stream, "categories.item")}
        names = sorted({name for classes in category_lists.values() for name in classes.values()})
        global_ids = {name: index for index, name in enumerate(names)}
        videos = {}
        image_maps = {}
        tracks = defaultdict(dict)
        for member in members:
            image_map = {}
            with source.open(member) as stream:
                for image in ijson.items(stream, "images.item"):
                    vid = image.get("video") or image["file_name"].rsplit("/", 1)[0]
                    index = int(image.get("frame_index", image.get("frame_id", 0)))
                    image_map[str(image["id"])] = (vid, index)
                    video = videos.setdefault(vid, {"id": vid, "width": int(image["width"]),
                                  "height": int(image["height"]), "frames": {}})
                    video["frames"][index] = image["file_name"]
            image_maps[member] = image_map
            with source.open(member) as stream:
                for annotation in ijson.items(stream, "annotations.item"):
                    if annotation.get("iscrowd", 0) or annotation.get("bbox") is None:
                        continue
                    vid, index = image_map[str(annotation["image_id"])]
                    name = category_lists[member][str(annotation["category_id"])]
                    category = global_ids[name]
                    track_id = (member, str(annotation["track_id"]), category)
                    track = tracks[vid].setdefault(track_id, {"category_id": category, "boxes": {}})
                    track["boxes"][index] = [float(v) for v in annotation["bbox"]]
        compact_videos, annotations = [], []
        for vid, video in sorted(videos.items()):
            indices = sorted(video.pop("frames").items())
            frame_map = {original: new for new, (original, _) in enumerate(indices)}
            compact_videos.append({**video, "file_names": [path for _, path in indices]})
            for track in tracks[vid].values():
                boxes = [None] * len(indices)
                for index, box in track["boxes"].items():
                    boxes[frame_map[index]] = box
                annotations.append({"id": len(annotations), "video_id": vid,
                                    "category_id": track["category_id"], "bboxes": boxes, "iscrowd": 0})
    result = {"videos": compact_videos, "annotations": annotations,
              "categories": [{"id": i, "name": name} for i, name in enumerate(names)],
              "source": {"dataset": "FSYTV-40", "archive_sha256": checksum(archive),
                         "members": members, "conversion": "bbox only; original categories mapped by name"}}
    write_json(output, result)
    return {"videos": len(compact_videos), "tracks": len(annotations), "classes": len(names),
            "source": result["source"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(convert(args.archive, args.output))


if __name__ == "__main__":
    main()
