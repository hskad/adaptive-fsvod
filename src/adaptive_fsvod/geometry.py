"""Boxes use normalized coordinates on the encoder's padded square."""

import torch


def cxcywh_to_xyxy(boxes):
    center, size = boxes[..., :2], boxes[..., 2:]
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def xyxy_to_cxcywh(boxes):
    return torch.cat(((boxes[..., :2] + boxes[..., 2:]) / 2,
                      boxes[..., 2:] - boxes[..., :2]), dim=-1)


def normalize_boxes(boxes, width, height):
    """OWL-ViT v2 pads on the right and bottom before resizing."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    return xyxy_to_cxcywh(boxes / max(width, height))


def pixel_boxes(boxes, width, height):
    corners = cxcywh_to_xyxy(boxes) * max(width, height)
    corners[..., 0::2] = corners[..., 0::2].clamp(0, width)
    corners[..., 1::2] = corners[..., 1::2].clamp(0, height)
    return corners


def box_iou(first, second):
    area1 = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(-1)
    area2 = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(-1)
    lo = torch.maximum(first[:, None, :2], second[None, :, :2])
    hi = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    inter = (hi - lo).clamp(min=0).prod(-1)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-8), union


def generalized_box_iou(first, second):
    iou, union = box_iou(first, second)
    lo = torch.minimum(first[:, None, :2], second[None, :, :2])
    hi = torch.maximum(first[:, None, 2:], second[None, :, 2:])
    enclosing = (hi - lo).clamp(min=0).prod(-1)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-8)


def nms(boxes, scores, threshold=0.5):
    """Class-specific callers keep suppression independent between classes."""
    order = torch.argsort(scores, descending=True, stable=True)
    kept = []
    while order.numel():
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        overlap, _ = box_iou(boxes[current:current + 1], boxes[order[1:]])
        order = order[1:][overlap[0] <= threshold]
    return torch.stack(kept) if kept else torch.empty(0, dtype=torch.long, device=boxes.device)
