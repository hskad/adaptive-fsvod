# Larger confirmation of the frozen feature-centering models

This check follows the small, uncertain +0.612 AP result on six videos. It is
an additional confirmation, not a new method search. Keep all earlier results.

## Decisions made before this set is scored

- Use six previously unused FSYTV classes, four single-class videos per class,
  and at most eight uniformly spaced frames per video. Preserve empty frames.
  Exclude every class and video in the original train/validation/test splits,
  the first fresh confirmation, and the second fresh confirmation.
- Select from the pinned public RGB archive with selection seed 20261028.
  Require four eligible videos per class. Shuffle sorted eligible classes,
  then videos, once. Do not redraw classes or remove difficult videos.
- Cache the same pinned OWLv2 encoder, native resolution, 128 candidates, and
  512 feature dimensions. Use four support crops per class, one from each
  selected video, chosen by the existing round-robin pool with seed 42.
- Use two-way one-shot episodes. Every query video is evaluated once against
  each of the other five classes: 120 episodes in total. Select supports with
  seed 20261029, excluding the query video. Each query uses its entire selected
  sequence. These repeated frames are not independent samples.
- Compare the six already trained revision_v5 models: fixed and centered,
  seeds 42/43/44, 50 actual updates, shared 300-update schedule, threshold 0.8.
  Also evaluate the three existing revision_v2 tuned fixed baseline checkpoints
  (300-update training budget, threshold 0.6, validation-selected checkpoint).
  This is a reference with a different training budget, not the primary pair.
- No training, threshold selection, center fitting, or checkpoint selection
  uses this set. Freeze the manifest, episodes, support pool, and all nine
  checkpoint/configuration hashes in Git before evaluation.

## Analysis and interpretation

Report frame-level COCO box AP/AP50/AP75 for every seed, means and training-seed
sample standard deviations, and all three paired 50-update AP differences.
The primary effect is mean(centered AP - fixed AP) at the matched 50-update
budget. Do not treat training-seed SD as uncertainty over test videos.

Use a paired, class-stratified video bootstrap: 1,000 draws, seed 20261030.
Within each of the six classes, draw four query videos with replacement.
Each draw includes all five episodes and all frames for each sampled video.
Recompute aggregate COCO AP for each model, then average the three paired
seed differences. Report the percentile 95% interval. Use identical draws
for all models. Keep the fixed support galleries and six class identities;
this interval is conditional on those choices and does not cover unseen
classes, changed support examples, or uncertainty from model development.

Bootstrap accumulation may reuse COCO per-image matches, which do not change
when images are duplicated. Verify this implementation against a direct COCO
evaluation on duplicated images before using it on research outputs.

A narrow positive interval would support a benefit on this specific protocol.
An interval containing zero means this test does not establish an improvement.
Do not claim better converged accuracy or a speedup of the frozen encoder.
Whatever the outcome, retain the fixed models, report the result, and stop
method selection on these confirmation data.
