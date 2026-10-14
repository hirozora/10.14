"""Seed-averaged tables, paired match-bootstrap contrasts and learning curves of the formal runs."""
import argparse
import json
from pathlib import Path

import numpy as np

from units import read_json, write_json
from scoring import VIEWS, metric_array, paired_bootstrap, bootstrap_weights

METRICS = ('five_task_loss', 'event_accuracy', 'event_macro_f1', 'event_ce', 'time_mae_seconds', 'position_error_metres',
           'team_nll', 'team_accuracy', 'player_nll', 'player_top1', 'player_top3')
LOWER_IS_BETTER = {'five_task_loss', 'event_ce', 'time_mae_seconds', 'position_error_metres', 'team_nll', 'player_nll'}
PERCENT = {'event_accuracy', 'event_macro_f1', 'team_accuracy', 'player_top1', 'player_top3'}
CONTRASTS = [
    ('hgt_no_seq', 'hgt_full', 'temporal chain (next + gap edges)'),
    ('hgt_untyped_seq', 'hgt_full', 'gap typing of temporal edges'),
    ('hgt_no_player', 'hgt_full', 'player-event edges'),
    ('hgt_shuffled_player', 'hgt_full', 'correct player identities on player-event edges'),
    ('hgt_no_team', 'hgt_full', 'team-event edges'),
    ('hgt_no_member', 'hgt_full', 'player-team roster edges'),
    ('hgt_no_type', 'hgt_full', 'event-type hubs'),
    ('hgt_no_zone', 'hgt_full', 'start-zone hubs'),
    ('hgt_no_possession', 'hgt_full', 'possession phase nodes'),
    ('hgt_events_only', 'hgt_full', 'all entity and hub relations'),
    ('hgt_anchor_readout', 'hgt_full', 'window-mean readout'),
    ('seq_mean', 'hgt_full', 'graph vs sequence (same inputs and heads)'),
    ('seq_anchor', 'seq_mean', 'window-mean readout in the sequence model'),
]


def scale(metric, value):
    return 100 * value if metric in PERCENT else value


def curve(run):
    return [json.loads(line) for line in (run / 'curve.jsonl').open()]


def main(config):
    runs, reports = Path(config['paths']['runs']), Path(config['paths']['reports'])
    seeds = config['training']['seeds']
    conditions = list(config['conditions'])
    complete = {c: [runs / f'formal_{c}_s{s}' for s in seeds if (runs / f'formal_{c}_s{s}' / 'status.json').exists()
                    and read_json(runs / f'formal_{c}_s{s}' / 'status.json')['status'] == 'complete'] for c in conditions}
    summary = dict(seeds=seeds, conditions={}, contrasts=[], curves={}, thresholds={}, test_evaluated=False)
    tables = {}
    for c in conditions:
        if not complete[c]:
            continue
        per_seed = [read_json(r / 'best_metrics.json') for r in complete[c]]
        status = [read_json(r / 'status.json') for r in complete[c]]
        stats = [np.load(r / 'best_match_statistics.npz') for r in complete[c]]
        matches = stats[0]['match_id']
        for s in stats:
            np.testing.assert_array_equal(s['match_id'], matches)
        tables[c] = {v: np.stack([s[v] for s in stats]) for v in VIEWS}
        entry = dict(runs=[str(r) for r in complete[c]], best_epochs=[s['best_epoch'] for s in status], stop=[s['stop_reason'] for s in status],
                     epochs=[s['epochs'] for s in status], views={})
        for v in VIEWS:
            values = {m: [p['metrics'][v][m] for p in per_seed] for m in METRICS}
            entry['views'][v] = {m: dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)) if len(x) > 1 else None, per_seed=x) for m, x in values.items()}
        summary['conditions'][c] = entry
        rows = {}
        for r in complete[c]:
            for row in curve(r):
                if 'validation' in row:
                    rows.setdefault(row['epoch'], []).append(row['validation'])
        summary['curves'][c] = {str(e): {v: {m: float(np.mean([x[v][m] for x in vals])) for m in METRICS} for v in ('all', 'non_duel_target')}
                                for e, vals in sorted(rows.items()) if len(vals) == len(complete[c])}
    if 'hgt_full' in summary['conditions'] and len(complete['hgt_full']) > 1:
        full = summary['conditions']['hgt_full']['views']['all']
        summary['thresholds'] = {m: 3 * full[m]['sd'] for m in METRICS}
    first = next(iter(tables.values()), None)
    if first is not None:
        weights = bootstrap_weights(first['all'].shape[1], config['evaluation']['bootstrap']['replicates'], config['evaluation']['bootstrap']['seed'])
        for a, b, meaning in CONTRASTS:
            if a not in tables or b not in tables or len(complete[a]) != len(complete[b]):
                continue
            result = {v: paired_bootstrap(tables[a][v], tables[b][v], weights) for v in ('all', 'non_duel_target')}
            summary['contrasts'].append(dict(candidate=a, reference=b, removed_or_changed=meaning, seeds=len(complete[a]), difference='candidate - reference',
                                             views={v: {m: result[v][m] for m in METRICS} for v in result}))
    write_json(reports / 'summary.json', summary)
    write_tables(summary, reports / 'tables.md')


def fmt(metric, value):
    return f'{scale(metric, value):.2f}' if metric in PERCENT else (f'{value:.4f}' if metric in ('five_task_loss', 'event_ce', 'team_nll', 'player_nll') else f'{value:.3f}')


def write_tables(summary, path):
    lines = ['# Five-task HGT: tables', '', 'Validation, England, 3-seed means (± sample SD). Accuracy-type metrics in %, time in s, position in m.', '']
    for view in ('all', 'non_duel_target'):
        lines += [f'## Conditions ({view})', '', '| condition | n | ' + ' | '.join(METRICS) + ' | best epochs |', '|---' * (len(METRICS) + 3) + '|']
        for c, e in summary['conditions'].items():
            cells = []
            for m in METRICS:
                x = e['views'][view][m]
                sd = '' if x['sd'] is None else f" ± {scale(m, x['sd']):.2f}" if m in PERCENT else f" ± {x['sd']:.4f}"
                cells.append(fmt(m, x['mean']) + sd)
            lines.append(f"| {c} | {len(e['runs'])} | " + ' | '.join(cells) + f" | {e['best_epochs']} |")
        lines.append('')
    lines += ['## Contrasts (candidate − reference; paired match bootstrap 95% CI)', '']
    for view in ('all', 'non_duel_target'):
        lines += [f'### {view}', '', '| candidate | reference | change | ' + ' | '.join(METRICS) + ' |', '|---' * (len(METRICS) + 3) + '|']
        for item in summary['contrasts']:
            cells = []
            for m in METRICS:
                r = item['views'][view][m]
                cells.append(f"{scale(m, r['mean']):+.3f} [{scale(m, r['ci95'][0]):+.3f}, {scale(m, r['ci95'][1]):+.3f}]")
            lines.append(f"| {item['candidate']} | {item['reference']} | {item['removed_or_changed']} | " + ' | '.join(cells) + ' |')
        lines.append('')
    lines += ['## Learning curves (all targets; seed means)', '']
    for m in ('five_task_loss', 'event_accuracy', 'position_error_metres', 'team_nll', 'player_top1'):
        epochs = sorted({int(e) for c in summary['curves'].values() for e in c}, key=int)
        shown = [e for e in epochs if e in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 120)]
        lines += [f'### {m}', '', '| condition | ' + ' | '.join(f'ep{e}' for e in shown) + ' |', '|---' * (len(shown) + 1) + '|']
        for c, points in summary['curves'].items():
            lines.append(f'| {c} | ' + ' | '.join(fmt(m, points[str(e)]['all'][m]) if str(e) in points else '' for e in shown) + ' |')
        lines.append('')
    Path(path).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(read_json(parser.parse_args().config))
