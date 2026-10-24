"""Paired video bootstrap using cached COCO per-image matches.

IoU matching is local to each image, so resampling images only changes their
multiplicity. Duplicate cached detection columns in original image order and
retain COCO's stable score sort, including ties within a duplicated image.
No additional IoU matching or model inference is needed for each draw.
"""

import contextlib
import copy
import io

import numpy as np


class ClusterCocoAP:
    def __init__(self, truth, predictions, image_clusters, clusters):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        self.clusters = list(clusters)
        cluster_index = {key: i for i, key in enumerate(self.clusters)}
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = copy.deepcopy(truth)
            gt.createIndex()
            if predictions:
                dt = gt.loadRes(copy.deepcopy(predictions))
            else:
                dt = COCO()
                dt.dataset = {**truth, "annotations": []}
                dt.createIndex()
            evaluation = COCOeval(gt, dt, "bbox")
            evaluation.evaluate()
        self.recall_thresholds = evaluation.params.recThrs.copy()
        self.iou_thresholds = evaluation.params.iouThrs.copy()
        self.categories = []
        self.category_ids = list(evaluation.params.catIds)
        image_ids = evaluation.params.imgIds
        area_count = len(evaluation.params.areaRng)
        image_count = len(image_ids)
        for category_index, _ in enumerate(evaluation.params.catIds):
            scores, matches, ignored, image_ranges = [], [], [], []
            gt_counts = np.zeros(len(clusters), dtype=np.float64)
            # Area index 0 is 'all'; evaluateImg uses the largest maxDets (100).
            for image_index, image_id in enumerate(image_ids):
                item = evaluation.evalImgs[category_index * area_count * image_count + image_index]
                if item is None:
                    continue
                cluster = cluster_index[image_clusters[image_id]]
                gt_counts[cluster] += np.count_nonzero(np.logical_not(item["gtIgnore"]))
                count = len(item["dtScores"])
                start = len(scores)
                scores.extend(item["dtScores"])
                matches.append(item["dtMatches"] != 0)
                ignored.append(item["dtIgnore"])
                image_ranges.append((cluster, np.arange(start, start + count)))
            if matches:
                match = np.concatenate(matches, axis=1)
                ignore = np.concatenate(ignored, axis=1)
                tp = np.logical_and(match, np.logical_not(ignore))
                fp = np.logical_and(np.logical_not(match), np.logical_not(ignore))
            else:
                tp = fp = np.zeros((len(self.iou_thresholds), 0), dtype=bool)
            self.categories.append((tp, fp, np.asarray(scores), image_ranges, gt_counts))

    def metrics(self, multiplicities, by_category=False):
        weights = np.asarray(multiplicities)
        if weights.shape != (len(self.clusters),) or np.any(weights < 0) or np.any(weights != np.floor(weights)):
            raise ValueError("Cluster multiplicities must be nonnegative integers")
        precision = []
        category_precision = {}
        for category, (tp, fp, scores, image_ranges, gt_counts) in zip(self.category_ids, self.categories):
            positive_count = float(gt_counts @ weights)
            if positive_count == 0:
                continue
            parts = [np.tile(indices, int(weights[cluster])) for cluster, indices in image_ranges
                     if weights[cluster] and len(indices)]
            indices = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
            indices = indices[np.argsort(-scores[indices], kind="mergesort")]
            true_positive = np.cumsum(tp[:, indices], axis=1, dtype=np.float64)
            false_positive = np.cumsum(fp[:, indices], axis=1, dtype=np.float64)
            recall = true_positive / positive_count
            pr = true_positive / (true_positive + false_positive + np.spacing(1))
            if pr.shape[1]:
                pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
            sampled = np.zeros((len(self.iou_thresholds), len(self.recall_thresholds)))
            for index in range(len(self.iou_thresholds)):
                positions = np.searchsorted(recall[index], self.recall_thresholds, side="left")
                valid = positions < pr.shape[1]
                sampled[index, valid] = pr[index, positions[valid]]
            precision.append(sampled)
            category_precision[category] = sampled
        if not precision:
            raise ValueError("No ground truth in this bootstrap draw")
        def summarize(values):
            return {"AP": float(values.mean()),
                    "AP50": float(values[np.flatnonzero(np.isclose(self.iou_thresholds, .5))[0]]),
                    "AP75": float(values[np.flatnonzero(np.isclose(self.iou_thresholds, .75))[0]])}
        if by_category:
            return {category: summarize(values.mean(1) * 100) for category, values in category_precision.items()}
        return summarize(np.stack(precision).mean(axis=(0, 2)) * 100)


def stratified_video_draws(video_classes, count=1000, seed=20261030):
    clusters = sorted(video_classes)
    classes = sorted(set(video_classes.values()))
    rng = np.random.default_rng(seed)
    draws = np.zeros((count, len(clusters)), dtype=np.int64)
    for category in classes:
        positions = np.array([i for i, video in enumerate(clusters) if video_classes[video] == category])
        for index in range(count):
            sampled = rng.choice(positions, len(positions), replace=True)
            np.add.at(draws[index], sampled, 1)
    return clusters, draws
