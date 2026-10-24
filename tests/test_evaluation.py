import torch

from adaptive_fsvod.evaluate import coco_metrics
from adaptive_fsvod.geometry import normalize_boxes, pixel_boxes


def ground_truth():
    return {"info": {}, "images": [{"id": 1, "width": 100, "height": 100},
                                    {"id": 2, "width": 100, "height": 100}],
            "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                              "bbox": [0, 0, 10, 10], "area": 100},
                             {"id": 2, "image_id": 2, "category_id": 2, "iscrowd": 0,
                              "bbox": [50, 50, 10, 10], "area": 100}]}


def test_boxes_round_trip_through_nonsquare_padding():
    boxes = torch.tensor([[10., 5., 60., 30.]])
    recovered = pixel_boxes(normalize_boxes(boxes, 100, 40), 100, 40)
    assert torch.allclose(recovered, boxes, atol=1e-5)


def test_evaluation_never_matches_across_frames():
    predictions = [{"image_id": 2, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .99},
                   {"image_id": 1, "category_id": 2, "bbox": [50, 50, 10, 10], "score": .99}]
    assert coco_metrics(ground_truth(), predictions)["AP"] == 0


def test_missing_class_detections_reduce_macro_ap():
    predictions = [{"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": .99}]
    metrics = coco_metrics(ground_truth(), predictions)
    assert abs(metrics["AP"] - 50) < 1e-5
    assert coco_metrics(ground_truth(), [])["AP"] == 0
