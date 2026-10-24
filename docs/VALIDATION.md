# Software validation

Checks performed locally on 24 October 2026:

- Windows, Python 3.12.3, PyTorch 2.9.1+cpu, Transformers 4.46.3, NumPy 1.26.4.
- 21 automated tests passed. Tests include a simulated disconnect and an exact comparison of resumed versus uninterrupted training weights. HTTP range tests check selective ZIP reading, range-cache reuse, and rejection of invalid or full-archive responses.
- Both fixed and adaptive methods completed four synthetic training updates, validation, checkpoint save/reload, and test evaluation. Losses and gradients were finite. These outputs are not research evidence.
- The real public OWLv2 weights at revision `cfd3195ba4ea9592eec887ded089f4c08eff231d` loaded successfully. A generated test image produced 128 candidate features of dimension 512, with finite features and boxes. This single CPU extraction took about 18.25 seconds with four CPU threads. It is an API check, not an end-to-end speed benchmark or detection accuracy test.
- FSYTV annotations downloaded and converted successfully: 2,235 videos, 3,679 tracks, and 40 category names. Mask segmentations were discarded in the compact box conversion.
- Pilot preparation and frame verification passed: 6/3/3 classes, 18/9/9 videos, and 131/50/68 selected frames for train/validation/test.
- All 249 selected RGB frames were retrieved from the pinned YouTube-VOS ZIP mirror. The first partial download transferred 81,294,147 bytes rather than the full 8,957,125,671-byte archive. Frame paths, dimensions, JPEG decoding, and ZIP CRCs passed. Each image has a recorded SHA-256. Six labeled image overlays were inspected and showed aligned boxes.
- A two-frame, two-way one-shot real training episode completed one optimizer update for each method on CPU. Support examples came from other videos. Fixed/adaptive losses were about 7.663/7.669 with finite gradients. The memory rules selected different token counts. This is an integration check, not evidence of useful learning or improved AP.

Tests cover support/query separation, class/video split overlap, deterministic episodes, full selected evaluation sequences, cache mismatch, insufficient support, background-only frames, no-overlap Hungarian matching, empty temporal memory, memory reset, token caps, pretrained API tensor shapes, non-square image coordinates, cross-frame matching errors, classes with no predictions, and checkpoint resume.

The automation runner adds a synthetic integration test for paired training, resume of completed runs without additional optimizer steps or changed weights, collection of validation diagnostics, aggregate results, archive checksums, source-image exclusion, and rejection of mismatched evaluation protocols. The new test and the existing 21 tests passed locally (22 total). This checks automation behavior; it does not establish a research improvement.

The user's Tesla T4 notebook session subsequently completed feature extraction for all three splits, a successful 50-step real-data overfit check, 300-step training for both methods, and paired test evaluation. Fixed/adaptive test AP was 45.36/45.09. See [the pilot record](PILOT_RESULTS.md) for the supplied outputs, protocol, and limitations. This session required a manual notebook import-path repair; a completely fresh run of the revised notebook remains unverified.

The user then completed paired training seeds 43 and 44 with the same configurations and episodes. Mean fixed/adaptive test AP across seeds 42/43/44 was 45.64/45.34, with sample standard deviations 0.32/0.22. Adaptive AP was lower in every pair. This is a negative result for the proposed rule, not an improvement.

The complete GPU experiment was then downloaded and verified locally. All six best checkpoint hashes match their test evaluation records, and seed, selected step, configurations, source hashes, cache signatures, and final step 300 passed checks. All 249 local image hashes passed. COCO metrics recomputed from the six saved detection files exactly reproduce reported AP/AP50/AP75. CPU inference from both seed-42 checkpoints exactly reproduces validation AP. See [source records and diagnostics](../results/pilot_20261024/README.md). Training revision is the clean `69268ece373f28e468681a4081c9df745d95b33e`; the recorded 300-step training loops including validation took 23.38–24.56 seconds on the T4.

Validation diagnostics show that adaptive thresholds are usually at their lower bound, weak selected tokens have low class/box agreement, and disabling memory at inference increases AP for both seed-42 checkpoints. This is validation evidence for further investigation, not a new method's test improvement.

The later feature-centering study found a small average improvement at equal 50-update budgets: validation AP 52.47/59.71 and fresh confirmation AP 51.43/52.04 for fixed/centered methods. Two of three paired seeds improved. All six confirmation score sets were recomputed exactly, and all exported inference checkpoints matched full checkpoint model tensors. See [the complete record](../results/revision_v5/README.md). This is limited-budget evidence on a small new set, not a full-budget accuracy gain.

The larger frozen-model confirmation used six further unseen classes, 24 videos,
192 distinct frames, and 120 balanced two-way one-shot episodes. At matched
50-update budgets, fixed/centered AP was 38.61/35.14; all three centered runs
were lower. The validation-selected 300-update fixed reference averaged 55.10.
This result does not confirm the earlier small-set centering gain. The pre-score
protocol and all 216 shared feature tensors were hashed and frozen in Git before
scoring. The optional SDPA encoder kernel passed native-output checks on three
images and mask/weight unit checks; every model used the same cache. All 37
software tests passed locally, with the later class-level accumulation change
also passing its two targeted tests. All nine packaged inference checkpoints
passed exact model-tensor reload checks. See [the full larger-set record](../results/larger_confirmation/README.md)
for predictions and the paired, class-stratified query-video bootstrap.

The later score-calibration study used the same three 300-update-budget fixed
heads for baseline and proposed ranking. beta=0 exactly reproduced all three
archived validation AP values. Validation selected beta=1 from {0, 0.5, 1}; its
mean validation gain was 15.267 AP points. On a further class/video-disjoint set
of six classes, 24 videos, and 192 distinct frames, baseline/objectness AP was
47.440/58.298. All three paired gains were positive. The predeclared 1,000-draw
class-stratified query-video bootstrap interval was [+5.960, +14.850] AP points,
conditional on fixed classes, support episodes, and trained seed pairs. All six
class means improved, with the largest effect for bear. All six AP/AP50/AP75
sets were recomputed exactly and the cached COCO accumulator matched the direct
metrics. Every feature, model, setting, and episode was frozen before scoring.

All 48 local software tests passed. The later unit-quality control also passed
its 11 targeted score tests. Three shared inference heads survived export and
reload with every model tensor exactly equal. No new training was performed
for the proposed score. The new notebook's code cells passed syntax checks;
its component stages ran locally, but a new cloud notebook session has not
been tested end to end. [Full records](../results/score_calibration/README.md).

31 tests passed after the new model and update-cap changes. Checks include class-balanced training-only centering, checkpoint mean storage, rejection of changed centers on resume, preservation of the learning-rate schedule prefix under a shorter actual budget, attention-quality ratios, zero-residual identity, and backward compatibility. A real saved baseline reproduced its validation AP exactly after the additions.

Still pending: accessible checkpoint links and peak GPU memory. The final report is deferred at the user's request. Notebook cells are syntax checked locally. GitHub Actions uses CPU checks and does not download public model weights.
