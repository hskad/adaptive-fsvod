"""DETR-style bipartite matching with background supervision."""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .geometry import cxcywh_to_xyxy, generalized_box_iou


class DetectionLoss(nn.Module):
    def __init__(self, classification_weight=2.0, box_weight=5.0, background_weight=0.1):
        super().__init__()
        self.classification_weight = classification_weight
        self.box_weight = box_weight
        self.background_weight = background_weight

    @torch.no_grad()
    def match(self, output, target):
        if target["labels"].numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=output["logits"].device)
            return empty, empty
        class_cost = -output["logits"].log_softmax(-1)[:, target["labels"]]
        distance = torch.cdist(output["boxes"], target["boxes"], p=1)
        giou = generalized_box_iou(cxcywh_to_xyxy(output["boxes"]), cxcywh_to_xyxy(target["boxes"]))
        cost = self.classification_weight * class_cost + self.box_weight * (distance - giou)
        rows, columns = linear_sum_assignment(cost.detach().float().cpu().numpy())
        return (torch.as_tensor(rows, device=cost.device), torch.as_tensor(columns, device=cost.device))

    def forward(self, output, target):
        logits, predicted = output["logits"], output["boxes"]
        rows, columns = self.match(output, target)
        labels = torch.full((logits.shape[0],), logits.shape[-1] - 1,
                            dtype=torch.long, device=logits.device)
        labels[rows] = target["labels"][columns]
        weights = logits.new_ones(logits.shape[-1])
        weights[-1] = self.background_weight
        classification = F.cross_entropy(logits, labels, weight=weights)
        # A graph-connected zero keeps empty frames valid for backward().
        box_loss = predicted.sum() * 0
        if rows.numel():
            matched = predicted[rows]
            truth = target["boxes"][columns]
            l1 = F.l1_loss(matched, truth, reduction="sum") / len(columns)
            giou = generalized_box_iou(cxcywh_to_xyxy(matched), cxcywh_to_xyxy(truth))
            box_loss = l1 + (1 - giou.diag()).mean()
        total = self.classification_weight * classification + self.box_weight * box_loss
        return {"total": total, "classification": classification, "box": box_loss}
