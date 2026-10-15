# Signal screening: next-actor prediction (validation)

Seed means. Single-actor targets; unreachable targets count as misses for top-k.

| condition | single_top1 | single_top3 | single_mrr | single_nll | duel_top1 | duel_nll | player_nll_reachable | best epochs |
|---|---|---|---|---|---|---|---|---|
| M1_summary | 33.52 | 60.84 | 51.48 | 2.0741 | 18.94 | 2.7675 | 2.1664 | [116, 120, 120] |
| M2_pooled | 35.00 | 63.34 | 53.13 | 2.0107 | 19.44 | 2.7017 | 2.1027 | [116, 116, 120] |
| M2_shuffled | 33.81 | 61.23 | 51.78 | 2.0626 | 19.07 | 2.7455 | 2.1536 | [116, 116, 120] |
| M0_ctx_identity | 31.83 | 58.37 | 49.63 | 2.1453 | 17.80 | 2.7825 | 2.2302 | [120, 116, 120] |
| M0_ctx_identity_dot | 32.54 | 59.51 | 50.37 | 2.1197 | 18.17 | 2.7693 | 2.2062 | [112, 116, 120] |

## Contrasts (candidate - reference, match bootstrap 95% CI)

| question | contrast | top-1 (pp) | top-3 (pp) | NLL | decision |
|---|---|---|---|---|---|
| Q1 rerun: dynamic history content over static identity, frequency and recency | M2_pooled - M1_summary | +1.48 [+1.23, +1.73] | +2.50 [+2.17, +2.85] | -0.0634 [-0.0692, -0.0580] | positive |
| Q1 rerun control: history content bound to the right player | M2_pooled - M2_shuffled | +1.19 [+0.93, +1.46] | +2.12 [+1.77, +2.49] | -0.0520 [-0.0588, -0.0458] | positive |
| History channel carrying a teammate's content, over summaries only | M2_shuffled - M1_summary | +0.29 [+0.16, +0.41] | +0.39 [+0.25, +0.53] | -0.0115 [-0.0134, -0.0094] | not positive |
| Dot-product output term in the sequence baseline | M0_ctx_identity_dot - M0_ctx_identity | +0.70 [+0.53, +0.87] | +1.14 [+0.95, +1.34] | -0.0256 [-0.0284, -0.0228] | not positive |
| Explicit frequency and recency per candidate vs identities attached to events (MLP scorer) | M1_summary - M0_ctx_identity | +1.69 [+1.40, +1.96] | +2.47 [+2.12, +2.83] | -0.0712 [-0.0768, -0.0657] | positive |
| Explicit frequency and recency per candidate vs identities attached to events (MLP plus dot-product scorer) | M1_summary - M0_ctx_identity_dot | +0.98 [+0.73, +1.23] | +1.33 [+1.05, +1.63] | -0.0456 [-0.0513, -0.0404] | positive |
| Candidate-side history vs identities attached to events (MLP scorer) | M2_pooled - M0_ctx_identity | +3.17 [+2.77, +3.57] | +4.97 [+4.49, +5.49] | -0.1346 [-0.1444, -0.1254] | positive |
| Candidate-side history vs identities attached to events (MLP plus dot-product scorer) | M2_pooled - M0_ctx_identity_dot | +2.46 [+2.12, +2.82] | +3.84 [+3.40, +4.30] | -0.1091 [-0.1187, -0.1002] | positive |

## Top-1 by when the true next actor last acted

| condition | anchor actor | 1-4 units ago | 5-16 units ago | 17-79 units ago | not in window |
|---|---|---|---|---|---|
| M1_summary | 73.25 | 35.05 | 25.48 | 26.32 | 27.14 |
| M2_pooled | 73.57 | 37.03 | 28.44 | 27.16 | 24.74 |
| M2_shuffled | 73.52 | 34.95 | 26.03 | 26.60 | 27.90 |
| M0_ctx_identity | 67.01 | 29.36 | 26.46 | 26.48 | 28.07 |
| M0_ctx_identity_dot | 69.07 | 31.55 | 26.59 | 26.26 | 27.91 |
