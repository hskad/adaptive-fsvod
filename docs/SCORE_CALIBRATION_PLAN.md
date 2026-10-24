# Bounded study: objectness in the final detection score

The user approved this study after the larger centering test failed to confirm
a gain. Preserve every earlier result. Do not tune with any inspected test set.

## Modification

For candidate j, keep its original winning foreground class, background decision,
refined box, and temporal computation. Change only its detection ranking score:

    s_j = p_j * q_j ** beta

p_j is the maximum foreground softmax score. q_j is the frozen OWLv2 objectness
for the same candidate, before learned box refinement. Apply this score before
the existing class-wise NMS, score floor, and global detection cap. beta=0 must
recover the original detections and AP exactly. No score change enters temporal
memory. The previous quality experiment changed memory attention; this study
changes the final detection ranking only.

This is a ranking heuristic. In log space it adds beta*log(q_j) to log(p_j).
It gives extra weight to class-agnostic evidence that a proposal contains an
object. The two signals may be correlated and uncalibrated; their product is
not asserted to be an exact probability. Low objectness can penalize real
objects. No improvement is guaranteed.

For a correct candidate a and an incorrect candidate b with positive scores,
the new ranking places a first exactly when

    beta * (log(q_a) - log(q_b)) > log(p_b) - log(p_a).

Thus objectness must separate the candidates strongly enough to overcome an
incorrect class-score ranking. This gives a testable reason for an AP gain,
without assuming independent or perfectly calibrated probabilities. It also
shows why the rule can fail when a real object has weak objectness.

Reference: the [pinned OWLv2 documentation](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/owlv2)
describes its query-independent objectness head. Combining these scores is not
claimed as a globally new detection principle.

## Fixed validation procedure

- Use the three existing revision_v2 fixed baseline checkpoints, seeds 42/43/44,
  threshold 0.6, 300-update training budget, selected on the original validation
  data. No training, model selection, or feature centering is performed here.
- Keep the original validation manifest, 20 episodes, features, NMS IoU 0.5,
  score floor 0.001, and maximum 100 detections. Use CPU, four threads.
- Before scoring, commit the model, configuration, validation-input hashes, this
  plan, and the implementation to Git.
- At seed 42, evaluate beta in {0, 0.5, 1}. Select by validation AP; ties prefer
  smaller beta. beta=0 must match its archived validation AP within 1e-9.
- If beta=0 wins, stop. Otherwise keep the selected beta fixed and evaluate it
  and beta=0 at seeds 43 and 44. Do not switch beta after those results.
- Continue to a fresh confirmation only if every seed has a positive paired AP
  difference and the mean difference is at least 1.0 AP point. This gate avoids
  another expensive data extraction for a negligible validation gain. Otherwise
  record a negative or weak result and stop this family. Do not automatically
  add Soft-NMS, change thresholds, or expand the beta grid.

## Fresh confirmation, only if the gate passes

Select six unused classes with four single-class videos each and at most eight
uniform frames per video. Exclude the original train/validation/test classes
and videos, both earlier fresh confirmations, and the larger confirmation.
Use selection seed 20261031, episode seed 20261101, and support pool seed 42.
Require four eligible videos per class; shuffle sorted classes and videos once.
Use four support crops per class, one per video. Every query video is tested
against each of the other five classes, with one support per class from other
videos: 120 two-way one-shot episodes. Preserve empty frames and full sequences.

Use one pinned OWLv2 SDPA feature cache (128 candidates, 512 dimensions), shared
by all comparisons. Freeze the selected beta, all three model hashes, manifest,
episodes, support pool, and complete cache hashes in Git before scoring.

Compare beta=0 with the selected beta for each seed. Recompute AP/AP50/AP75 from
saved detections. Report every paired score and class AP. Use the existing
verified COCO accumulator for a paired, class-stratified query-video bootstrap:
1,000 shared draws, seed 20261102. Retain all episodes/frames for each sampled
video; average the three paired AP differences. The percentile 95% interval is
conditional on these classes, support episodes, and fixed trained seed pairs.
Do not treat it as uncertainty over all unseen classes or model development.
Keep every outcome and stop model selection on these new test data.
