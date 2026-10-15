"""Post-hoc contrasts (not part of the pre-registered decisions): same paired match bootstrap as src/scr_report.py."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from scr_scoring import paired_bootstrap, bootstrap_weights

CONTRASTS = [
    ('M2_shuffled', 'M1_summary', 'history channel with content moved to a teammate (within-team shuffle) over summaries only'),
    ('M2_pooled', 'M0_ctx_identity', 'candidate-side pooled history vs identities attached to events'),
    ('M3_attention', 'M2_shuffled', 'attention over own history vs pooled teammate history'),
]
KEYS = ('single_top1', 'single_top3', 'single_nll', 'duel_top1', 'duel_nll')


def main(config):
    runs, reports = Path(config['paths']['runs']), Path(config['paths']['reports'])
    seeds = config['training']['seeds']
    tables = {c: np.stack([np.load(runs / f'formal_{c}_s{s}' / 'best_match_statistics.npz')['table'] for s in seeds]) for c in config['conditions']}
    weights = bootstrap_weights(next(iter(tables.values())).shape[1], config['evaluation']['bootstrap']['replicates'], config['evaluation']['bootstrap']['seed'])
    out = []
    for a, b, meaning in CONTRASTS:
        r = paired_bootstrap(tables[a], tables[b], weights)
        out.append(dict(candidate=a, reference=b, meaning=meaning, result={k: r[k] for k in KEYS}))
    # M1_summary seed 20260716 decayed its learning rate twice before the identity jump and never made it;
    # these seed means leave out the middle seed for every condition.
    summary = json.load(open(reports / 'summary.json'))['conditions']
    keep = [i for i, s in enumerate(seeds) if s != 20260716]
    two_seed = {c: {k: float(np.mean([v['per_seed'][i] for i in keep])) for k, v in summary[c]['metrics'].items()} for c in summary}
    (reports / 'posthoc_contrasts.json').write_text(json.dumps(dict(note='post hoc; not part of the pre-registered decisions', contrasts=out,
                                                                    seed_means_without_20260716=two_seed), indent=2) + '\n')
    show = ['single_top1', 'single_nll'] + [k for k in two_seed['M0_static'] if k.startswith('top1[')]
    print('seed means without 20260716:', ' | '.join(show))
    for c, m in two_seed.items():
        print(f'  {c:16}', ' '.join(f'{m[k]:.4f}' if 'nll' in k else f'{100 * m[k]:6.2f}' for k in show))
    for item in out:
        r = item['result']
        cells = [f"{k} {(100 if 'nll' not in k else 1) * r[k]['mean']:+.3f} [{(100 if 'nll' not in k else 1) * r[k]['ci95'][0]:+.3f}, {(100 if 'nll' not in k else 1) * r[k]['ci95'][1]:+.3f}] seeds {[round((100 if 'nll' not in k else 1) * x, 3) for x in r[k]['per_seed']]}" for k in KEYS]
        print(f"{item['candidate']} - {item['reference']}:\n  " + '\n  '.join(cells))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    main(json.load(open(parser.parse_args().config)))
