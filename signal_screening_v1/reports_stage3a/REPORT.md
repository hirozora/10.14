# Signal screening: next-actor prediction (validation)

Seed means. Single-actor targets; unreachable targets count as misses for top-k.

| condition | single_top1 | single_top3 | single_mrr | single_nll | duel_top1 | duel_nll | player_nll_reachable | best epochs |
|---|---|---|---|---|---|---|---|---|
| M2_pooled (reference: signal_screening_v1_stage2_fixed_schedule) | 35.00 | 63.34 | 53.13 | 2.0107 | 19.44 | 2.7017 | 2.1027 | [116, 116, 120] |
| H_match | 35.14 | 63.63 | 53.30 | 2.0015 | 19.47 | 2.6959 | 2.0940 | [116, 116, 120] |
| H_match_shuffled | 33.54 | 60.91 | 51.51 | 2.0749 | 18.70 | 2.7673 | 2.1671 | [120, 116, 120] |

## Contrasts (candidate - reference, match bootstrap 95% CI)

| question | contrast | top-1 (pp) | top-3 (pp) | NLL | decision |
|---|---|---|---|---|---|
| Q3a: own history and counts from the whole match over the window only | H_match - M2_pooled | +0.14 [-0.01, +0.28] | +0.28 [+0.11, +0.46] | -0.0091 [-0.0119, -0.0064] | not positive |
| Q3a control: whole-match history content bound to the right player | H_match - H_match_shuffled | +1.60 [+1.32, +1.88] | +2.72 [+2.35, +3.09] | -0.0734 [-0.0799, -0.0672] | positive |

## Top-1 by when the true next actor last acted

| condition | anchor actor | 1-4 units ago | 5-16 units ago | 17-79 units ago | not in window |
|---|---|---|---|---|---|
| M2_pooled | 73.57 | 37.03 | 28.44 | 27.16 | 24.74 |
| H_match | 73.45 | 36.35 | 28.62 | 27.89 | 25.25 |
| H_match_shuffled | 73.40 | 35.56 | 25.74 | 25.84 | 26.53 |
