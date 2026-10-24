# Revision 2: confidence-gated temporal fusion

This protocol was written before the revision experiments. The first adaptive-threshold proposal is retained as a negative result. Its original test set has already been inspected; no later test result from that set will be described as blind confirmation.

## Controlled change

Keep the fixed memory threshold and existing temporal module. Before temporal fusion, predict probabilities from the current frame alone. For candidate j, let q_j be its largest foreground probability. Define

    g_j = 1[foreground wins] * clip((q_j - beta) / (1 - beta), 0, 1)
    z_j = x_j + g_j * (T(x_j, M_previous) - x_j)

T is the original complete temporal update: cross-attention plus the residual feed-forward layer. beta is fixed at 0.7. Gate values are detached during training. Memory selection uses the final predictions as before. The gate has no learned parameters. The original baseline uses g=1 whenever memory is present. Empty memory uses the original single-frame path for both methods.

The gate protects low-confidence and background candidates from a temporal update. If g=0, the current-frame representation is unchanged. In norm,

    ||z_j - x_j|| = g_j ||T(x_j, M_previous) - x_j||.

If the temporal residual has conditional mean b and covariance Sigma for a given x and memory, its conditional squared perturbation is g_j^2 (||b||^2 + tr(Sigma)), because g is fixed conditional on those inputs. Thus the gate bounds disturbance of the current-frame representation. This is a stability argument, not a proof that detection AP will improve. It can also suppress useful temporal evidence for uncertain objects.

## Validation search and controls

- Reuse the original training and validation class/video manifests, feature caches, and episode files. Do not read original test metrics during selection.
- All new training is paired on CPU with four PyTorch threads, seed 42, 300 steps, validation every 50 steps, AdamW lr 1e-4, weight decay 0.01, and the same losses and scheduler.
- Baseline search: fixed memory thresholds 0.6, 0.7, 0.8.
- Proposed search: the same three memory thresholds, with confidence gating and beta fixed at 0.7. No gate-threshold search is performed.
- Select one configuration per method using best validation AP; ties prefer the higher memory threshold. Both families receive three search runs.
- Train a separate no-memory control for 300 steps, to distinguish a useful temporal contribution from simply avoiding memory.
- Freeze selected configurations before testing. Repeat selected baseline, gated, and no-memory methods for training seeds 43 and 44. Report every planned seed.
- Preserve search runs, selected configurations, logs, environments, hashes, and validation outputs. Report negative outcomes.

## Confirmation

If the gated method improves validation AP, evaluate the frozen methods on a fresh class-disjoint confirmation subset. Selection seed is 20261024. Exclude every class and video in the original three splits. Retain classes with at least two otherwise eligible single-class videos in the public RGB source, shuffle sorted eligible classes with the fixed seed, and select the first three. Shuffle each class's sorted eligible videos with the same RNG and select two. Keep up to three uniformly spaced annotated frames per video. Use a round-robin support pool of two examples per class (one per video), two-way one-shot episodes, 30 episodes with seed 20261025, and the original pinned encoder and 128-token cache settings.

This gives at most 18 distinct frames and six distinct videos, a small confirmation set. Use the same support/query separation and box AP implementation, preserve empty frames, and report its size and repeated-episode limitation. If data access prevents this, label any evaluation on the already-inspected test set as exploratory and state the limitation.

The method is a project extension. No claim of global research novelty is made without a literature review.
