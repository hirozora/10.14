"""Early-training comparison: HGT vs sequence at equal learning rates, and HGT ablations vs full by epoch."""
import argparse
import json
from pathlib import Path

import numpy as np

from units import read_json, write_json

KEYS = ('five_task_loss', 'event_ce', 'event_accuracy', 'time_mae_seconds', 'position_error_metres', 'team_nll', 'player_nll', 'player_top1')
EPOCHS = (1, 2, 4, 8, 12, 16, 20, 24)


def curve(run, view='all'):
    rows = [json.loads(line) for line in (run / 'curve.jsonl').open()]
    return {r['epoch']: r['validation'][view] for r in rows if 'validation' in r}


def main(config):
    runs, reports = Path(config['paths']['runs']), Path(config['paths']['reports'])
    out = dict(equal_lr={}, ablation_by_epoch={}, note='validation, all targets; difference = sequence (or ablation) minus HGT full')
    for lr in config['lr_search']['grid']:
        h, s = runs / f'lr_hgt_full_lr{lr}', runs / f'lr_seq_mean_lr{lr}'
        if not (h / 'curve.jsonl').exists() or not (s / 'curve.jsonl').exists():
            continue
        ch, cs = curve(h), curve(s)
        out['equal_lr'][str(lr)] = {str(e): {k: dict(hgt=ch[e][k], seq=cs[e][k], seq_minus_hgt=cs[e][k] - ch[e][k]) for k in KEYS}
                                    for e in EPOCHS if e in ch and e in cs}
    seeds = config['training']['seeds']
    for condition in config['conditions']:
        paths = [runs / f'formal_{condition}_s{s}' for s in seeds]
        paths = [p for p in paths if (p / 'curve.jsonl').exists()]
        full = [runs / f'formal_hgt_full_s{s}' for s in seeds if (runs / f'formal_{condition}_s{s}' / 'curve.jsonl').exists()]
        if not paths or condition == 'hgt_full' or not all((p / 'curve.jsonl').exists() for p in full):
            continue
        a, b = [curve(p) for p in paths], [curve(p) for p in full]
        common = sorted(set.intersection(*(set(c) for c in a + b)))
        out['ablation_by_epoch'][condition] = {str(e): {k: float(np.mean([x[e][k] for x in a]) - np.mean([y[e][k] for y in b])) for k in KEYS}
                                               for e in common}
    write_json(reports / 'early_training.json', out)
    lines = ['# Early training', '', 'Validation, all targets. Equal learning rate: sequence minus HGT full (seed 20260715, LR search runs).', '']
    for lr, by_epoch in out['equal_lr'].items():
        lines += [f'## lr {lr}', '', '| epoch | ' + ' | '.join(KEYS) + ' |', '|---' * (len(KEYS) + 1) + '|']
        for e, d in by_epoch.items():
            lines.append(f'| {e} | ' + ' | '.join(f"{d[k]['hgt']:.4f} → {d[k]['seq']:.4f}" for k in KEYS) + ' |')
        lines.append('')
    lines += ['## Ablation minus full by epoch (seed means)', '']
    for c, by_epoch in out['ablation_by_epoch'].items():
        lines += [f'### {c}', '', '| epoch | ' + ' | '.join(KEYS) + ' |', '|---' * (len(KEYS) + 1) + '|']
        for e, d in by_epoch.items():
            if int(e) in (1, 2, 4, 8, 12, 24, 48, 72, 96, 120):
                lines.append(f'| {e} | ' + ' | '.join(f'{d[k]:+.4f}' for k in KEYS) + ' |')
        lines.append('')
    (reports / 'early_training.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(read_json(parser.parse_args().config))
