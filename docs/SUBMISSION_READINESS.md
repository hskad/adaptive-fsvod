# Repository requirements

This page maps the course's non-report requirements to the final score study. The one-page report, team details, template, PDF name, and portal upload are outside this repository check.

| Requirement | Location |
|---|---|
| Project overview, exact modification, original paper and implementation | [README](../README.md), [fixed study plan](SCORE_CALIBRATION_PLAN.md) |
| Python/framework/CUDA/hardware and installation commands | [Reproduction guide, section 1](REPRODUCE.md#1-install), [pinned requirements](../requirements-reproduce.txt), [source training configuration](../results/revision_v2/source_runs/fixed_tau060_seed42/config.json), [evaluation environment](../results/score_calibration/confirmation_environment.json) |
| Dataset links, directory layout, preprocessing and preparation commands | [DATA.md](DATA.md), [reproduction guide, sections 2 and 5](REPRODUCE.md), [frozen labels/episodes/image checksums](../results/score_calibration/confirmation_protocol/) |
| Executable baseline/proposed training and hyperparameters | [Reproduction guide, section 3](REPRODUCE.md#3-reproduce-shared-head-training), [training helper](../scripts/train_fixed_reference.py), [selected configuration](../results/revision_v2/configs/fixed_tau060.json). Both methods share the trained heads; the proposed method requires zero additional updates. |
| Inference/evaluation commands, paths, expected metrics | [Reproduction guide, sections 2 and 4](REPRODUCE.md), [six-run reproduction command](../scripts/reproduce_results.py) |
| Accessible trained weights, configurations, logs and comparative results | [Public release](https://github.com/hskad/adaptive-fsvod/releases/tag/v1.0-results), [head manifest](../results/score_calibration/inference_bundle.manifest.json), [feature manifest](../results/score_calibration/features_bundle.manifest.json), [seed 42 logs/config](../results/revision_v2/source_runs/fixed_tau060_seed42/), [43](../results/revision_v2/source_runs/fixed_tau060_seed43/), [44](../results/revision_v2/source_runs/fixed_tau060_seed44/), [final evidence](../results/score_calibration/README.md) |
| Clear code organization, modification location, reproducibility | [README code map](../README.md#code-organization-and-verification), `decode()` in [evaluate.py](../src/adaptive_fsvod/evaluate.py), [score controls](../tests/test_score_calibration.py), [software validation](VALIDATION.md) |

The public heads are inference exports. They omit optimizer/RNG state. Executable training writes full resumable checkpoints and logs. The proposed scoring rule changes no trained tensor; therefore one shared head bundle covers both baseline and proposed evaluation.

The evidence is for a reduced custom FSYTV setup. It does not establish full-paper benchmark reproduction or a globally new scoring principle. The final report should state this scope and the larger bear-class effect.
