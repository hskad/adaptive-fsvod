# Real-video pilot: three training seeds

These values were supplied from the user's GPU notebook session on 24 October 2026. The full evaluation JSON, logs, and checkpoints remain in the experiment's Google Drive directory and have not yet been imported into this repository. This page is a record of the observed output, not a replacement for those artifacts.

## Protocol

- Frozen public OWLv2 Base encoder, pinned revision `cfd3195ba4ea9592eec887ded089f4c08eff231d`, 128 candidates, 512-dimensional cached class features.
- Custom FSYTV subset: train/validation/test contain 6/3/3 classes, 18/9/9 videos, and 131/50/68 selected frames. Classes and videos are disjoint across splits.
- Two-way, one-shot episodes. Support and query examples use different videos. Fixed episode files contain 300/20/30 train/validation/test episodes, with episode seeds 42/43/44.
- Both methods use training seed 42, 300 optimizer steps, the initial repository configurations, and validation every 50 steps. Both best checkpoints were selected at step 250 by validation AP. Threshold hyperparameters were not tuned; this is a pilot with initial settings, not a comparison against a tuned fixed baseline.
- Hardware reported by the session: Tesla T4. Reported PyTorch version: `2.11.0+cu130`, CUDA version: `13.0`. Exact environment and training Git commit must be taken from the saved run configurations before final publication.
- Frame-level COCO box AP across IoU 0.50:0.05:0.95, not video mask AP. Test data was evaluated after checkpoint selection.

## Learning check

Before comparative training, the fixed method repeated one training episode for 50 steps. Mean loss over the first and last five steps was 5.6242 and 0.5247. On that same episode, AP/AP50/AP75 were 90.64/100.00/100.00. These values check learning on training data and must not be presented as generalization results.

## Comparative results

| Method | Best validation AP | Test AP | Test AP50 | Test AP75 | Mean memory tokens |
|---|---:|---:|---:|---:|---:|
| Fixed threshold | 59.33 | 45.36 | 52.91 | 50.54 | 2.47 |
| Adaptive threshold | 59.79 | 45.09 | 52.78 | 50.15 | 4.82 |
| Adaptive minus fixed | +0.46 | -0.27 | -0.13 | -0.39 | +2.35 |

Differences in AP are percentage points. Rounded entries come from these test outputs:

```csv
method,AP,AP50,AP75,mean_memory_tokens,cached_pipeline_fps,synthetic
fixed,45.361073896666845,52.91139943719391,50.53549691528468,2.4698275862068964,94.86199356350708,False
adaptive,45.094359390040715,52.77727464054397,50.14527699488853,4.818965517241379,105.12604448473408,False
```

The adaptive method improved validation AP slightly but reduced test AP by 0.2667 points. It retained about 1.95 times as many memory tokens. More retained tokens did not produce a test accuracy gain in this run. Whether the added tokens contain false detections requires inspection; token counts alone do not establish the cause.

The reported cached pipeline speeds include cached feature IO, the trainable heads, and decoding. They exclude the image encoder. Each method was timed once, so these values do not establish a speed advantage.

## Additional paired training seeds

After the initial seed-42 result, training seeds 43 and 44 were specified before those runs. Both methods used the same frozen configurations, 300-step budget, fixed episode files, and validation checkpoint selection. Test data was not used to choose a new threshold configuration. The following values are rounded notebook output supplied by the user; source JSON files are still pending import.

| Training seed | Fixed test AP | Adaptive test AP | Adaptive minus fixed |
|---|---:|---:|---:|
| 42 | 45.36 | 45.09 | -0.27 |
| 43 | 45.58 | 45.39 | -0.19 |
| 44 | 45.99 | 45.53 | -0.46 |
| Mean ± sample standard deviation | 45.64 ± 0.32 | 45.34 ± 0.22 | |

The adaptive rule scored lower for all three paired training seeds. The mean loss of test AP is about 0.30 points. These results do not support an improvement claim for this rule. Training-seed variability on the same small test set does not measure uncertainty over new classes or new videos. No statistical significance claim is made.

The user also saved the seed-42 evidence archive as `exports/pilot_seed42.zip` in the experiment Drive directory. Reported size: 140.5 MB. Reported SHA-256: `47e657a3c7b4a72d1e31837837b7c1c98445cc244141799d969dfeb03572939c`. The archive has not been downloaded and independently verified here. It predates the additional seeds and therefore does not contain their runs.

## Limits and next steps

This is three training seeds on one small subset and one fixed collection of test episodes. There is no evidence here of a test improvement or statistical significance. The report must describe this negative test result if it uses this comparison.

Preserve both checkpoints, configurations, training logs, validation and test outputs, comparison files, cache metadata, and all manifest and episode files. Publish accessible checkpoint links and import the source result files before treating the repository as a complete reproducibility package.

Before another method change, inspect validation behavior, including selected token counts and thresholds. Record the current rule as a negative ablation. Any later method or threshold changes must be selected on validation data, and the already-inspected test set must be disclosed as such. A fresh held-out set is preferable for confirmation after exploratory changes. Do not repeatedly select changes on these test scores.
