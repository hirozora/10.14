"""Player top-1 by next-actor recency (seed mean ± SD) and the hgt_full gap decomposed by recency bin (validation)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from units import read_json, write_json
from batches import SplitData
from subsets import target_recency, RECENCY

CONDITIONS = ('hgt_full', 'hgt_no_player', 'hgt_shuffled_player', 'seq_mean', 'seq_anchor')


def main(config):
    runs, reports = Path(config['paths']['runs']), Path(config['paths']['reports'])
    recency = target_recency(SplitData(config, 'validation'))
    share = np.array([(recency == b).sum() for b in range(5)]) / (recency >= 0).sum()
    top1 = {c: np.array([[np.load(runs / f'formal_{c}_s{s}' / 'best_validation_statistics.npz')['player_top1'][recency == b].mean() for b in range(5)]
                         for s in config['training']['seeds']]) for c in CONDITIONS}
    result = dict(bins=list(RECENCY), share_of_single_actor_targets=share.tolist(),
                  top1_mean={c: v.mean(0).tolist() for c, v in top1.items()}, top1_sd={c: v.std(0, ddof=1).tolist() for c, v in top1.items()},
                  gap_from_hgt_full={}, test_evaluated=False)
    print('share', np.round(100 * share, 1))
    for c, v in top1.items():
        print(f'{c:20}', '  '.join(f'{100 * m:5.1f}±{100 * sd:.1f}' for m, sd in zip(v.mean(0), v.std(0, ddof=1))))
    for c in CONDITIONS[1:]:
        gap = top1['hgt_full'].mean(0) - top1[c].mean(0)
        result['gap_from_hgt_full'][c] = dict(by_bin=gap.tolist(), weighted=(gap * share).tolist(), total=float((gap * share).sum()))
        print(f'hgt_full - {c:20} by bin {np.round(100 * gap, 2)}  weighted {np.round(100 * gap * share, 2)}  total {100 * (gap * share).sum():.2f}')
    write_json(reports / 'player_gap_by_recency.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(read_json(parser.parse_args().config))
