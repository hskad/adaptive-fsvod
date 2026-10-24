# Bounded revision 3: objectness-weighted temporal memory

The confidence-gated revision improved validation AP by about 0.38 points across three seeds, but its fresh confirmation AP was 22.54 versus 22.94 for the tuned fixed baseline. The no-memory control scored 19.27 on that confirmation set. Preserve these negative results. Do not select another method using that set.

## Evidence and change

On validation data, selected memory tokens with correct class and IoU >= 0.50 had mean frozen-encoder objectness 0.454; other selected tokens averaged 0.115. Both objectness and class confidence gave token-agreement AUC near 0.83, but objectness had a larger relative separation (approximately 4:1 compared with 1.18:1 for class confidence). These are token agreement diagnostics, not detection precision or calibrated probabilities.

Use the frozen detector's class-agnostic objectness q_k to weight stored memory inside cross-attention:

    a_jk = dot(Q_j, K_k) / sqrt(d)
    w_jk = exp(a_jk + log(max(q_k, 1e-6))) / sum_l exp(a_jl + log(max(q_l, 1e-6)))
         = max(q_k, 1e-6) exp(a_jk) / sum_l max(q_l, 1e-6) exp(a_jl)

Thus, for two equally similar memory keys, their attention-weight ratio is their objectness ratio. Normalized attention is invariant to scaling every quality by the same amount, so it cannot on its own distinguish uniformly weak memory. Also scale the complete temporal residual (attention plus feed-forward layer) by r = max_k(q_k):

    z_j = x_j + r * (T_weighted(x_j, M_previous) - x_j)

The max avoids treating multiple overlapping candidates as independent evidence. If all selected objectness scores are zero, r=0 and the current-frame representation is preserved. The perturbation norm is r times the unscaled temporal residual norm. Together these operations introduce a class-independent quality prior and a conservative residual scale, with no trainable parameters. They leave the temporal module available to new classes rather than switching it off based on current class confidence. They do not guarantee AP improvement; low objectness can also occur on a real small or occluded object.

Keep memory selection, token cap, loss, optimizer, schedule, encoder, and training inputs unchanged. Do not combine this rule with the confidence gate in this study. Raw objectness is already present in real feature caches; carry it into the heads and retain selected scores with memory. Baseline and no-memory models ignore the added input and retain their original computations.

## Fixed experiment budget

- One final family of three validation-search runs: memory thresholds 0.6, 0.7, 0.8; objectness exponent fixed at 1; no further quality hyperparameters.
- Seed 42, 300 steps, validation every 50, four CPU threads, and the same training configuration as revision 2.
- Reuse the revision-2 fixed baseline search (three thresholds) and its paired selected-baseline and no-memory controls. Do not retrain equivalent completed controls merely because this implementation accepts an additional ignored input. Verify default model behavior against the saved baseline validation AP.
- Select the weighted configuration by best validation AP, with higher threshold breaking ties. Repeat that selected model at seeds 43 and 44. Freeze all selected checkpoints before confirmation.
- Keep every search and replication result, including failures. No further model families are selected from confirmation scores in this bounded study.

## New confirmation set

Use the revision-2 deterministic selection procedure with selection seed 20261026 and episode seed 20261027. Exclude all original classes/videos AND the revision-2 confirmation classes/videos. Select three remaining eligible classes, two videos per class, at most three uniformly spaced frames per video, two support candidates per class using pool seed 42, 30 fixed two-way one-shot episodes, and the original pinned encoder cache settings. This is a small confirmation study, not the paper's full benchmark. Evaluate all three frozen methods and all three seeds and report all results. Do not silently reuse a previously inspected test set as fresh confirmation.
