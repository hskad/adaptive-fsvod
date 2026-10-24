# Final feature-centering study: equal limited update budgets

The 300-update centering family stopped: best validation AP 59.704 versus tuned baseline 59.776. Do not call this a full-budget accuracy improvement. Its centered no-memory control scored 60.148, below the original no-memory control's 62.401.

The validation learning curves reveal a separate efficiency hypothesis: at update 50, best baseline AP was 52.496 and best centered AP was 59.704. The same curve inspection used all three thresholds for both families. No score from the second fresh subset has been read. The adaptation-limit hypothesis is selected using these validation curves and is not claimed to have been specified before the 300-update trials. Preserve those trials and this amendment.

## Fixed limited-budget comparison

- Same centering method as CENTER_PLAN.md; no additional model changes.
- Exactly 50 optimizer updates for BOTH families. Preserve the first 50 updates of the existing schedule: a horizon of 300, 15 warmup steps, lr 1e-4, weight decay .01, and the original cosine schedule. `--steps 300 --stop-after 50` records both actual and schedule budgets. No updates after 50 are executed.
- Train three thresholds 0.6/0.7/0.8 per family at seed 42 and select each family's best final validation AP (ties prefer higher threshold). Then repeat selected configurations at seeds 43 and 44. The same data, episodes, losses, model parameter counts, and CPU threads apply.
- Fit the frozen, class-balanced feature center on training support features only, once per run; store it in each checkpoint. No training or validation labels from novel classes enter the center.
- Before confirmation, verify actual update count 50, matching schedule budgets, same manifests and episodes, parameter counts, and frozen configuration and checkpoint hashes.
- Use the second fresh subset already prepared with selection seed 20261026 and episode seed 20261027. It remains unscored: the quality family stopped at validation, and the full-budget centering family also stopped at validation. Prepare an identical portable copy under revision_v5; do not draw alternative classes based on results.
- Freeze both families and every seed before scoring this set once. Report AP/AP50/AP75 and all paired differences. The subset contains at most 18 distinct frames in six new videos and three new classes, disjoint from the original splits and the first confirmation set. Seed standard deviation does not describe uncertainty over new datasets.
- A positive result supports faster adaptation at this 50-update budget only. Do not claim better fully converged accuracy, overall encoder speed, a reproduction of the original paper, or global novelty for centering. If this check fails, stop this work with no confirmed improvement.
