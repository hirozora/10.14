# Oracle union of existing models (validation, merged-duel targets)

N = 83860. Oracle = share of targets that at least one model in the group gets right (uses the label; diagnostic only).

| group | models | best single | oracle union | all agree | probability average | stacking (match CV) |
|---|---|---|---|---|---|---|
| seed replicas: v1 D00 (3 seeds) | 3 | 68.38 | 69.73 | 96.13 | 68.43 | 68.32 |
| seed replicas: v1 B_CXPT (3 seeds) | 3 | 68.41 | 69.87 | 95.84 | 68.4 | 68.41 |
| seed replicas: v1 C2 current_phase_control (3 seeds) | 3 | 68.47 | 69.96 | 95.78 | 68.48 | 68.49 |
| seed replicas: GBDT local (3 split seeds) | 3 | 68.23 | 69.61 | 95.62 | 68.41 | 68.44 |
| diverse architectures (HGT full, sequence, v1 Transformer D00, GBDT) | 4 | 68.52 | 71.47 | 91.12 | 68.58 | 68.6 |
| all five-task HGT conditions (seed 20260715) | 9 | 68.57 | 71.83 | 90.83 | 68.64 | 68.61 |
| all v1 strong runs | 39 | 68.48 | 72.28 | 89.13 | 68.5 | 68.33 |
| everything | 54 | 68.57 | 73.84 | 84.06 | 68.55 | 68.35 |

Top-k accuracy of single models: {"5task:hgt_full_s20260715": {"1": 68.52, "2": 85.89, "3": 94.87}, "v1:D00:s20260715": {"1": 68.28, "2": 85.76, "3": 94.91}, "gbdt:local:s0": {"1": 68.21, "2": 85.49, "3": 94.66}}

Union growth with the number of models (mean of 20 random orders): {"1": 68.35, "2": 69.64, "3": 70.26, "4": 70.68, "6": 71.21, "8": 71.52, "12": 72.03, "16": 72.36, "24": 72.78, "32": 73.12, "48": 73.53, "51": 73.59}

## Where the diverse union gains (relative to HGT full)

| true class | targets | share of gain | HGT full acc | union acc |
|---|---|---|---|---|
| Duel | 13769 | 25.5 | 17.2 | 21.7 |
| Foul | 1194 | 0.7 | 0.1 | 1.5 |
| Free Kick | 5546 | 0.1 | 99.8 | 99.8 |
| GK leaving line | 198 | 0.3 | 0.0 | 4.0 |
| Interruption | 4291 | 14.0 | 30.9 | 38.9 |
| Offside | 253 | 0.3 | 0.0 | 2.8 |
| Others on the ball | 7807 | 7.6 | 14.2 | 16.6 |
| Pass | 49055 | 42.2 | 95.3 | 97.5 |
| Save attempt | 521 | 4.4 | 57.2 | 78.1 |
| Shot | 1226 | 5.0 | 5.3 | 15.3 |

HGT full on samples gained by the union: {"count": 2476, "mean_top1_minus_top2_margin": 0.13, "mean_margin_all": 0.512, "true_class_rank_share": {"2": 87.7, "3": 9.7, "4": 1.9}}
