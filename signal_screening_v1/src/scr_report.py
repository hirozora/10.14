"""Seed tables, the contrasts listed in the configuration (report.contrasts) with paired match bootstrap, and recency subsets.

Every condition must have all its seeds complete, and every contrast must name configured conditions.
report.reference_runs maps a condition of an earlier stage to that stage's runs directory and protocol
version; its runs must be complete, of that version, with the same seeds and learning rate.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scr_scoring import metrics, paired_bootstrap, bootstrap_weights

SHOW = ('single_top1', 'single_top3', 'single_mrr', 'single_nll', 'duel_top1', 'duel_nll', 'player_nll_reachable')


def decide(result, rule_top1=0.01, rule_nll=-0.03):
    top1, nll = result['single_top1'], result['single_nll']
    same_direction = lambda r, sign: all(sign * x > 0 for x in r['per_seed'])
    top1_ok = top1['mean'] >= rule_top1 and same_direction(top1, 1) and top1['ci95'][0] > 0
    nll_ok = nll['mean'] <= rule_nll and same_direction(nll, -1) and nll['ci95'][1] < 0
    return 'positive' if (top1_ok or nll_ok) else 'not positive'


def main(config):
    runs, reports = Path(config['paths']['runs']), Path(config['paths']['reports'])
    seeds = config['training']['seeds']
    tables, summary = {}, dict(conditions={}, contrasts=[], test_evaluated=False)
    references = config['report']['reference_runs']
    if references:
        summary['references'] = {}
    for c in list(references) + list(config['conditions']):
        paths = [(Path(references[c]['runs']) if c in references else runs) / f'formal_{c}_s{s}' for s in seeds]
        if not all(json.load(open(p / 'status.json'))['status'] == 'complete' for p in paths):
            raise ValueError(f'{c}: not every seed is complete')
        if c in references:
            for p, s in zip(paths, seeds):
                recorded = json.load(open(p / 'config.json'))
                if (recorded['protocol']['version'], recorded['seed'], recorded['lr'], recorded['mode']) != (references[c]['version'], s, config['learning_rate']['value'], 'formal'):
                    raise ValueError(f'{p} is not a formal run of {references[c]["version"]} with seed {s} and the locked learning rate')
            summary['references'][c] = references[c]
        stats = [np.load(p / 'best_match_statistics.npz') for p in paths]
        for s in stats[1:]:
            np.testing.assert_array_equal(s['match_id'], stats[0]['match_id'])
        tables[c] = np.stack([s['table'] for s in stats])
        per_seed = [json.load(open(p / 'best_metrics.json')) for p in paths]
        status = [json.load(open(p / 'status.json')) for p in paths]
        summary['conditions'][c] = dict(best_epochs=[s['best_epoch'] for s in status], stop=[s['stop_reason'] for s in status],
                                        metrics={k: dict(mean=float(np.mean([m['metrics'][k] for m in per_seed])), per_seed=[m['metrics'][k] for m in per_seed])
                                                 for k in per_seed[0]['metrics']})
    weights = bootstrap_weights(next(iter(tables.values())).shape[1], config['evaluation']['bootstrap']['replicates'], config['evaluation']['bootstrap']['seed'])
    for a, b, meaning in config['report']['contrasts']:
        r = paired_bootstrap(tables[a], tables[b], weights)
        summary['contrasts'].append(dict(candidate=a, reference=b, question=meaning, decision=decide(r), result=r))
    (reports / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Signal screening: next-actor prediction (validation)', '', 'Seed means. Single-actor targets; unreachable targets count as misses for top-k.', '',
             '| condition | ' + ' | '.join(SHOW) + ' | best epochs |', '|---' * (len(SHOW) + 2) + '|']
    for c, e in summary['conditions'].items():
        name = f'{c} (reference: {references[c]["version"]})' if c in references else c
        lines.append(f'| {name} | ' + ' | '.join(f"{100 * e['metrics'][k]['mean']:.2f}" if 'nll' not in k else f"{e['metrics'][k]['mean']:.4f}" for k in SHOW) + f" | {e['best_epochs']} |")
    lines += ['', '## Contrasts (candidate - reference, match bootstrap 95% CI)', '', '| question | contrast | top-1 (pp) | top-3 (pp) | NLL | decision |', '|---|---|---|---|---|---|']
    for item in summary['contrasts']:
        r = item['result']
        cell = lambda k, s: f"{s * r[k]['mean']:+.2f} [{s * r[k]['ci95'][0]:+.2f}, {s * r[k]['ci95'][1]:+.2f}]"
        lines.append(f"| {item['question']} | {item['candidate']} - {item['reference']} | {cell('single_top1', 100)} | {cell('single_top3', 100)} | "
                     f"{r['single_nll']['mean']:+.4f} [{r['single_nll']['ci95'][0]:+.4f}, {r['single_nll']['ci95'][1]:+.4f}] | {item['decision']} |")
    recency = [k for k in next(iter(summary['conditions'].values()))['metrics'] if k.startswith('top1[')] if summary['conditions'] else []
    lines += ['', '## Top-1 by when the true next actor last acted', '', '| condition | ' + ' | '.join(k[5:-1] for k in recency) + ' |', '|---' * (len(recency) + 1) + '|']
    for c, e in summary['conditions'].items():
        lines.append(f'| {c} | ' + ' | '.join(f"{100 * e['metrics'][k]['mean']:.2f}" for k in recency) + ' |')
    (reports / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(json.load(open(parser.parse_args().config)))
