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

The user's Tesla T4 notebook session subsequently completed feature extraction for all three splits, a successful 50-step real-data overfit check, 300-step training for both methods, and paired test evaluation. Fixed/adaptive test AP was 45.36/45.09. See [the pilot record](PILOT_RESULTS.md) for the supplied outputs, protocol, and limitations. This session required a manual notebook import-path repair; a completely fresh run of the revised notebook remains unverified.

Still pending: import and verification of GPU-session source logs and checkpoints, accessible checkpoint links, exact saved environment and training revision, peak GPU memory and training duration, final validation hyperparameter choices, and additional training seeds. Notebook cells are syntax checked locally. GitHub Actions uses CPU checks and does not download public model weights.
