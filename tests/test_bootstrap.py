"""The fast bootstrap must match literal COCO evaluation of duplicated images."""

import copy

import numpy as np
import pytest

from adaptive_fsvod.bootstrap import ClusterCocoAP, stratified_video_draws
from adaptive_fsvod.evaluate import coco_metrics


def duplicated(truth, predictions, image_clusters, clusters, multiplicities):
    weights = dict(zip(clusters, multiplicities))
    result = {**truth, "images": [], "annotations": []}
    detections = []
    for image in sorted(truth["images"], key=lambda x: x["id"]):
        for _ in range(weights[image_clusters[image["id"]]]):
            new_id = len(result["images"]) + 1
            result["images"].append({**image, "id": new_id})
            for annotation in truth["annotations"]:
                if annotation["image_id"] == image["id"]:
                    result["annotations"].append({**annotation, "id": len(result["annotations"]) + 1, "image_id": new_id})
            for prediction in predictions:
                if prediction["image_id"] == image["id"]:
                    detections.append({**prediction, "image_id": new_id})
    return result, detections


def test_weighted_ap_matches_literal_duplicates_including_ties_and_empty_frames():
    rng = np.random.default_rng(17)
    truth = {"info": {}, "images": [], "annotations": [],
             "categories": [{"id": 1, "name": "one"}, {"id": 2, "name": "two"}]}
    predictions, mapping = [], {}
    for image_id in range(1, 13):
        mapping[image_id] = f"video{(image_id - 1) // 3}"
        truth["images"].append({"id": image_id, "width": 100, "height": 100})
        for category in (1, 2):
            if image_id % 4:
                truth["annotations"].append({"id": len(truth["annotations"]) + 1, "image_id": image_id,
                    "category_id": category, "iscrowd": 0, "bbox": [10, 10, 20, 20], "area": 400})
            # Deliberate ties, duplicate detections, misses, and false positives.
            for _ in range(4):
                x = 10 + float(rng.choice([0, 1, 3, 6, 40]))
                predictions.append({"image_id": image_id, "category_id": category,
                                    "bbox": [x, 10, 20, 20], "score": float(rng.choice([.3, .6, .9]))})
    clusters = sorted(set(mapping.values()))
    frozen = copy.deepcopy(predictions)
    accumulator = ClusterCocoAP(truth, predictions, mapping, clusters)
    assert predictions == frozen
    for weights in ([1, 1, 1, 1], [0, 3, 1, 0], [2, 0, 0, 2], [0, 0, 1, 3]):
        repeated_truth, repeated_predictions = duplicated(truth, predictions, mapping, clusters, weights)
        direct = coco_metrics(repeated_truth, repeated_predictions)
        fast = accumulator.metrics(weights)
        for name in direct:
            assert fast[name] == pytest.approx(direct[name], abs=1e-10), (weights, name, fast, direct)
    empty = ClusterCocoAP(truth, [], mapping, clusters)
    assert empty.metrics([1, 1, 1, 1]) == {"AP": 0., "AP50": 0., "AP75": 0.}


def test_bootstrap_resamples_videos_within_class_reproducibly():
    mapping = {"a": "cat", "b": "cat", "c": "dog", "d": "dog"}
    clusters, draws = stratified_video_draws(mapping, count=30)
    _, repeated = stratified_video_draws(mapping, count=30)
    assert clusters == ["a", "b", "c", "d"]
    assert np.array_equal(draws, repeated)
    assert np.all(draws[:, :2].sum(1) == 2)
    assert np.all(draws[:, 2:].sum(1) == 2)
    assert np.any(draws != 1)
