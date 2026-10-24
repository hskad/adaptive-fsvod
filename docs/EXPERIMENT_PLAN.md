# Experiment plan

## Question

Does a class-dependent adaptive confidence threshold improve few-shot video detection over a tuned fixed threshold in our reduced implementation?

## Scope

Use a frozen OWLv2 Base encoder. Train a small temporal cross-attention module and detection heads. Cache frozen encoder outputs once. Use the same candidate tokens for every method.

Differences from the assigned paper must be reported: the Base encoder, cached top-objectness candidate tokens, smaller temporal representation, residual box refinement, dataset subset, and experiment budget. These are shared design choices, not the claimed contribution.

## Memory rule

For each frame, let p[j,c] be the predicted class probability for token j and class c. Include a background class in the softmax. Compute class-specific statistics over all candidate tokens:

    tau[c] = clip(mean_j(p[j,c]) + alpha * std_j(p[j,c]), tau_min, tau_max)

Select a token only when its winning class is not background and its winning probability exceeds that class's threshold. Keep at most M tokens per class. The fixed baseline applies one constant tau with the same token cap. Empty memory bypasses cross-attention. Memory is reset for every query video and episode.

The expected benefit is an adjustment to score distributions that vary across classes and frames. A lower threshold can retain useful weak detections. A higher threshold can reject ambiguous detections. Neither effect guarantees an AP gain. Threshold floors and token caps limit weak evidence and memory cost.

Under a score calibration assumption, the expected incorrect fraction of selected tokens is the mean of (1 - p[j,c]) over those tokens. This explains the precision-versus-coverage trade-off. It does not prove that the adaptive rule is better. Validate that claim with experiments. Do not claim that the rule is globally novel without a literature check.

## Data and evaluation rules

1. Obtain labeled video data through the dataset provider's download links and terms.
2. Save the exact source, annotation checksum, selected video IDs, class IDs, and random seed.
3. Split by class for the few-shot study. Use disjoint train, validation, and test classes. Also prevent video overlap across splits, including multi-class videos.
4. Draw support object crops from videos distinct from the query video. Do not use the first query frame as support. K-shot refers to K labeled object examples per class.
5. Fix validation and test episodes in files. Evaluate both methods on the same files and all selected query frames. A short training clip is not a full test video.
6. Use validation data to select hyperparameters for both methods. Do not select thresholds, seeds, epochs, or checkpoints with test AP.
7. Treat query objects outside the episode classes as background under the stated episode protocol. Preserve frame and episode IDs in evaluation to prevent cross-frame box matching.
8. Report frame-level COCO AP across IoU 0.50:0.05:0.95. Also report AP50 and AP75. Do not label the average of AP50 and AP75 as COCO AP.
9. Start with a short overfit run. Then run baseline and proposed training with an equal budget. Use additional paired seeds if GPU time permits. Report seed variability when available.
10. Report actual data counts, run time, hardware, training steps, and negative results. Public encoder pretraining can include categories in this dataset; do not claim complete absence of prior category exposure.

## First milestone

- A validated manifest format and data loader.
- Deterministic episodes without support/query video leakage.
- A finite forward and backward step on CPU with synthetic features.
- Tests for temporal memory, matching, and box coordinates.
- A separate small OWLv2 integration check. This may require a pretrained download.

Synthetic tests establish software behavior. They are not detection accuracy evidence.

## Deliverables

The final repository needs preparation, training, inference, and evaluation commands; a Colab/Kaggle notebook; configurations; version and seed records; logs; and accessible checkpoint links. The one-page report must use the supplied LaTeX template. The template and team details have not yet been supplied.
