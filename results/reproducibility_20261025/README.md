# Clean reproduction check: 25 October 2026

The public release and instructions were checked in a separate clone and a new
Python 3.12 environment with system packages disabled. See
[verification.json](verification.json), [the complete environment](environment.json),
and [all six reproduced metric sets](evaluation_summary.json).

| Check | Result |
|---|---|
| Public downloads without credentials | Both release archive SHA-256 values matched |
| Extracted files | Three heads and 551 feature/metadata files verified |
| Fresh shared-head training | Three seeds, 300 updates each |
| Selected steps | 250, 250, 300 |
| Model tensors | Every tensor exactly matched the public head for each seed |
| Training loss logs | All 300 losses per seed exactly matched the original logs |
| Final evaluations | All six prediction/ground-truth records and all 18 AP/AP50/AP75 values matched exactly |
| Mean AP | Baseline 47.439707; objectness score 58.297585 |
| Bootstrap | All 1,000 draw multiplicities and per-model AP values matched exactly |
| Conditional 95% interval | [+5.960227, +14.850413] AP points |
| RGB access | All 192 frames downloaded and verified against frozen image hashes |
| Raw inference | Two frames, two support classes; finite detections and inspected annotated images |
| Completed-run reuse | All three completed training runs retained without new updates |

Fresh training logs and configurations: [seed 42](training/seed42/),
[43](training/seed43/), [44](training/seed44/). Full checkpoint byte hashes differ
from the inference exports because fresh run provenance and optimizer/RNG state
differ; the model tensors are exactly equal. No additional optimizer is used for
the proposed score.

Use [the evaluator instructions](../../docs/REPRODUCE.md) to repeat these checks.
Saved-feature evaluation and training require no GPU. The raw inference check
reused the pinned public encoder files already in the model cache. It is a
functional check, not a new accuracy result or a latency benchmark. A full new
cloud notebook run and clean extraction of all 192 encoder frames were not
performed. Cross-hardware bitwise equality is not claimed.
