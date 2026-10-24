# Adaptive FSVOD

A course project on adaptive temporal confidence filtering for few-shot video object detection.

The starting point is [Temporal Object-Aware Vision Transformer for Few-Shot Video Object Detection](https://ojs.aaai.org/index.php/AAAI/article/view/37497) by Yogesh Kumar and Anand Mishra (AAAI 2026). The [authors' implementation](https://github.com/yogesh-iitj/fs-video-vit) is the reference.

This project uses a smaller, documented adaptation. It does not reproduce the paper's full model or claim its published scores. The image encoder is frozen. A small temporal decoder and detection heads are trained. The controlled change is the rule that selects features for temporal memory.

## Status

Project setup is in progress. No experimental accuracy results or trained research checkpoints are available yet. Synthetic checks will be identified separately from experiments on real videos.

## Experiment

- Encoder: `google/owlv2-base-patch16-ensemble`.
- Baseline: a fixed threshold selects object features for the next frame.
- Proposed method: a bounded, class-dependent threshold uses the mean and standard deviation of current confidence scores.
- Controls: the same cached features, episodes, model size, token cap, optimizer, and evaluation procedure.
- Data: a small, labeled YouTube-VIS 2019 subset is the first candidate. Dataset access and the final subset must be checked before a research run.
- Metrics: frame-level COCO box AP, AP50, AP75, and measured run time. This is not video mask AP.

See [the experiment plan](docs/EXPERIMENT_PLAN.md) for the scope and evaluation rules.

## Version control

Use small commits for setup, data preparation, the baseline, the adaptive rule, and evaluation. Record the Git commit, configuration, seed, package versions, and data manifest for each experiment. Keep datasets, credentials, caches, and model weights out of Git. Publish research checkpoints through an accessible download location when they exist.
