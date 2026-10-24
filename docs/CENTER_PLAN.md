# Final validation check: class-balanced training feature centering

The objectness-weighted family's best seed-42 validation AP was 58.71, below the reused tuned baseline's 59.78. Stop that family after its three validation trials. Do not evaluate it on the second fresh subset. That subset was prepared but has not been scored by any model.

The following additional family is selected from validation feature diagnostics and prior work, not from the second confirmation set. Preserve and disclose every previous attempt. This is the final validation family in the current work; if it fails validation, stop without claiming an improvement.

## Method and attribution

Adapt the mean-subtraction and normalization idea from [SimpleShot, Wang et al.](https://arxiv.org/abs/1911.04623) to frozen OWLv2 support features and query tokens in the temporal detector. This is a course-project extension of the assigned detector, not an invention of feature centering or a claim of global novelty.

Compute a class-balanced center only from the existing training support pool:

    u_ci = f_ci / ||f_ci||
    mu = mean_c(mean_i(u_ci))
    phi(f) = normalize(project(normalize(f) - mu))

Use phi for both support prototypes and current query features before the existing temporal and detection heads. Keep the original fixed memory rule, box prior, losses, optimizer, and schedule. Disable the earlier confidence and objectness modifications. Store mu as a frozen checkpoint buffer; no trainable parameters are added. Fitting mu rejects non-training data. Validation and confirmation features or labels do not contribute to mu.

Training support features have a substantial shared direction (mean norm about 0.664). The class-balanced training and validation mean directions have cosine about 0.805. For the simple model f = b + s_class + noise, subtracting a training estimate of b removes the common inner-product term ||b||^2 that can inflate similarity between different classes. Subsequent normalization emphasizes differences between object classes. Estimation error and domain shift can still harm AP; this reasoning is not a performance guarantee.

## Fixed procedure

- Three seed-42 centered-memory trials at memory thresholds 0.6, 0.7, 0.8, matched to the existing three-threshold baseline search. Use 300 steps, validation every 50, four CPU threads, and identical data and optimization.
- Select centered memory by best validation AP, ties preferring higher threshold. Reuse the existing tuned fixed baseline and no-memory controls, whose default forward behavior remains unchanged.
- Add a centered no-memory control, 300 steps, without a threshold search. This separates centering from the value of temporal information.
- If centered memory does not beat the tuned baseline on validation, stop. Otherwise replicate centered memory and centered no-memory at seeds 43 and 44, retain every planned seed, and freeze all configurations and checkpoint hashes before confirmation.
- Use the still-uninspected second fresh subset: selection seed 20261026, episode seed 20261027, no class/video overlap with the original three splits or the first fresh subset, three classes, two videos per class, at most three frames per video, two support candidates per class, 30 two-way one-shot episodes. Store an identical copy under revision_v4 with portable relative image paths.
- Report all four frozen methods and all three seeds on this set. No more candidate families are selected from its scores. Report the small data size and earlier failed proposals.
