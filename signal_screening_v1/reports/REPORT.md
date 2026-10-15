# Signal screening: next-actor prediction (validation)

Seed means. Single-actor targets; unreachable targets count as misses for top-k.

| condition | single_top1 | single_top3 | single_mrr | single_nll | duel_top1 | duel_nll | player_nll_reachable | best epochs |
|---|---|---|---|---|---|---|---|---|
| M0_static | 25.46 | 51.17 | 43.99 | 2.2937 | 13.35 | 2.8401 | 2.3665 | [116, 116, 116] |
| M1_summary | 32.01 | 59.16 | 50.11 | 2.1117 | 18.19 | 2.8039 | 2.2039 | [120, 116, 120] |
| M2_pooled | 34.60 | 62.90 | 52.73 | 2.0275 | 19.15 | 2.7168 | 2.1194 | [116, 116, 120] |
| M2_shuffled | 33.47 | 61.07 | 51.52 | 2.0730 | 18.59 | 2.7831 | 2.1676 | [116, 120, 120] |
| M3_attention | 34.77 | 63.08 | 52.93 | 2.0182 | 19.03 | 2.7380 | 2.1141 | [120, 116, 120] |
| M0_ctx_identity | 31.08 | 57.51 | 48.97 | 2.1673 | 17.32 | 2.8044 | 2.2522 | [120, 116, 120] |

## Contrasts (candidate - reference, match bootstrap 95% CI)

| question | contrast | top-1 (pp) | top-3 (pp) | NLL | decision |
|---|---|---|---|---|---|
| frequency and recency over static identity | M1_summary - M0_static | +6.54 [+6.13, +6.98] | +7.99 [+7.61, +8.37] | -0.1820 [-0.1906, -0.1736] | positive |
| Q1: dynamic history content over static identity, frequency and recency | M2_pooled - M1_summary | +2.59 [+2.36, +2.83] | +3.73 [+3.39, +4.08] | -0.0842 [-0.0890, -0.0796] | positive |
| Q1 control: history content bound to the right player | M2_pooled - M2_shuffled | +1.13 [+0.87, +1.41] | +1.83 [+1.47, +2.19] | -0.0455 [-0.0519, -0.0397] | positive |
| Q2: candidate-conditioned attention over pooled history | M3_attention - M2_pooled | +0.17 [+0.01, +0.32] | +0.18 [+0.02, +0.35] | -0.0093 [-0.0112, -0.0074] | not positive |
| Q1 via attention: history with attention over summaries only | M3_attention - M1_summary | +2.76 [+2.52, +3.01] | +3.92 [+3.63, +4.21] | -0.0935 [-0.0986, -0.0888] | positive |
| Sequence baseline: player identities attached to the window events, over static candidates alone | M0_ctx_identity - M0_static | +5.62 [+5.27, +5.99] | +6.34 [+5.96, +6.71] | -0.1264 [-0.1333, -0.1194] | positive |
| Explicit frequency and recency per candidate vs identities attached to events | M1_summary - M0_ctx_identity | +0.92 [+0.59, +1.27] | +1.65 [+1.29, +2.00] | -0.0556 [-0.0617, -0.0496] | positive |
| Candidate-side history with attention vs identities attached to events | M3_attention - M0_ctx_identity | +3.69 [+3.26, +4.14] | +5.57 [+5.10, +6.06] | -0.1491 [-0.1584, -0.1403] | positive |

## Top-1 by when the true next actor last acted

| condition | anchor actor | 1-4 units ago | 5-16 units ago | 17-79 units ago | not in window |
|---|---|---|---|---|---|
| M0_static | 30.36 | 24.94 | 24.44 | 24.82 | 26.75 |
| M1_summary | 73.14 | 34.64 | 23.28 | 24.13 | 25.50 |
| M2_pooled | 73.47 | 36.23 | 27.91 | 27.07 | 24.39 |
| M2_shuffled | 73.45 | 35.11 | 25.42 | 26.10 | 27.29 |
| M3_attention | 72.93 | 36.40 | 28.50 | 27.23 | 23.83 |
| M0_ctx_identity | 65.97 | 28.63 | 25.84 | 25.67 | 27.61 |
