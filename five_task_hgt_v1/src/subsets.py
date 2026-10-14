"""Where relations act: per-subset metrics of the selected checkpoints (validation, seed means)."""
import argparse
from pathlib import Path

import numpy as np

from units import read_json, write_json
from batches import SplitData, gather

RAW10 = ('Duel', 'Foul', 'Free Kick', 'Goalkeeper leaving line', 'Interruption', 'Offside', 'Others on the ball', 'Pass', 'Save attempt', 'Shot')
RECENCY = ('anchor actor', '1-4 units ago', '5-16 units ago', '17-79 units ago', 'not in window')


def target_recency(data):
    """Units since the single next actor last acted in the window (non-duel targets with a known player)."""
    t = data.targets
    n = len(t['anchor'])
    out = np.full(n, -1, np.int64)
    roster = data.roster['player']
    for lo in range(0, n, 4096):
        rows = np.arange(lo, min(lo + 4096, n))
        g = gather(data, rows)
        slot = t['player_slot'][rows, 0]
        known = t['player_mask'][rows] & ~t['target_duel'][rows]
        player = np.where(known, roster[t['match_row'][rows], np.maximum(slot, 0)], -1)
        hit = (g['part_player'] == player[:, None, None]).any(-1) & g['valid'] & known[:, None]
        last = np.where(hit.any(1), hit.shape[1] - 1 - hit[:, ::-1].argmax(1), -1)
        offset = np.where(last >= 0, hit.shape[1] - 1 - last, -1)
        bins = np.select([offset == 0, (offset >= 1) & (offset <= 4), (offset >= 5) & (offset <= 16), offset >= 17], [0, 1, 2, 3], 4)
        out[rows] = np.where(known, bins, -1)
    return out


def main(config):
    runs, reports = Path(config['paths']['runs']), Path(config['paths']['reports'])
    data = SplitData(config, 'validation')
    t = data.targets
    recency = target_recency(data)
    anchor_type = data.units['event'][t['anchor']] - 1
    target_type = t['event']
    result = dict(recency_share={RECENCY[b]: float((recency == b).mean()) for b in range(5)}, conditions={})
    for c in config['conditions']:
        stats = [np.load(runs / f'formal_{c}_s{s}' / 'best_validation_statistics.npz') for s in config['training']['seeds']
                 if (runs / f'formal_{c}_s{s}' / 'best_validation_statistics.npz').exists()]
        if len(stats) != len(config['training']['seeds']):
            continue

        def mean(key, mask):
            return float(np.mean([s[key][mask].mean() for s in stats])) if mask.any() else None
        entry = dict(player_top1_by_recency={RECENCY[b]: mean('player_top1', recency == b) for b in range(5)},
                     player_nll_by_recency={RECENCY[b]: mean('player_ce', recency == b) for b in range(5)},
                     team_accuracy_by_anchor={RAW10[k]: mean('team_credit', (anchor_type == k) & ~t['target_duel']) for k in range(10)},
                     team_nll_by_anchor={RAW10[k]: mean('team_ce', (anchor_type == k) & ~t['target_duel']) for k in range(10)},
                     event_accuracy_by_anchor={RAW10[k]: float(np.mean([(s['event_pred'][anchor_type == k] == target_type[anchor_type == k]).mean() for s in stats])) for k in range(10)},
                     position_m_by_target={RAW10[k]: mean('position_m', (target_type == k) & t['pos_mask']) for k in range(10)},
                     time_mae_by_target={RAW10[k]: mean('time_abs', (target_type == k) & t['time_mask']) for k in range(10)})
        result['conditions'][c] = entry
    write_json(reports / 'subsets.json', result)
    lines = ['# Subsets (validation, seed means)', '', 'Share of non-duel targets by next-actor recency: ' +
             ', '.join(f'{k} {100 * v:.1f}%' for k, v in result['recency_share'].items()), '']
    for name, unit in (('player_top1_by_recency', '%'), ('team_accuracy_by_anchor', '%'), ('event_accuracy_by_anchor', '%'),
                       ('position_m_by_target', 'm'), ('time_mae_by_target', 's')):
        groups = list(next(iter(result['conditions'].values()))[name]) if result['conditions'] else []
        lines += [f'## {name}', '', '| condition | ' + ' | '.join(groups) + ' |', '|---' * (len(groups) + 1) + '|']
        for c, entry in result['conditions'].items():
            cells = ['' if entry[name][gname] is None else (f"{100 * entry[name][gname]:.2f}" if unit == '%' else f"{entry[name][gname]:.2f}") for gname in groups]
            lines.append(f'| {c} | ' + ' | '.join(cells) + ' |')
        lines.append('')
    (reports / 'subsets.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(read_json(parser.parse_args().config))
