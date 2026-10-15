"""Complementarity of existing models on the merged-duel validation targets.

A_oracle = |union of correct sets| / N with three references: the union over seed replicas of one
configuration (prediction instability), single-model top-k accuracy, and legitimate combinations
(probability averages and match-grouped stacking). Selecting the correct model uses the label, so
A_oracle is a diagnostic, not an achievable accuracy.

Usage: python -B analysis/oracle_union.py --config configs/experiment.json --device cuda:3
Outputs: reports/oracle_union.json, reports/oracle_union.md, reports/oracle_union_predictions.npz
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from units import read_json, write_json
from batches import SplitData
from models import build
import fit

RAW10 = ('Duel', 'Foul', 'Free Kick', 'GK leaving line', 'Interruption', 'Offside', 'Others on the ball', 'Pass', 'Save attempt', 'Shot')
V1 = Path('/home/li/football/graph_prediction_research')
V1_STRONG = ('formal_etp_B_CXPT', 'baseline_calibration_etp_D00', 'formal_etp_D01', 'formal_etp_D10', 'formal_etp_D11', 'formal_etp_C1_4',
             'formal_etp_C1_8', 'formal_etp_C1_16', 'formal_etp_C1_32', 'formal_etp_C2_player_relation', 'formal_etp_C2_player_control',
             'formal_etp_C2_current_phase_relation', 'formal_etp_C2_current_phase_control', 'formal_etp_C2_other_phase_team_relation',
             'formal_etp_C2_other_phase_team_control')


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def v1_alignment(val):
    """Index of the v1 target whose target record is the first record of each merged target unit."""
    index = np.load(V1 / 'data/validation_index.npz')
    order_to_record = {int(o): i for i, o in enumerate(index['original_order'])}
    first = val.units['record_order'][val.targets['anchor'] + 1, 0]
    v1 = np.load(V1 / 'runs/baseline_calibration_etp_D00_s20260715_lr0.0027/best_validation_predictions.npz')
    target_record = v1['target_anchor'] + 1
    record_to_target = {int(r): i for i, r in enumerate(target_record)}
    align = np.array([record_to_target[order_to_record[int(o)]] for o in first])
    if not np.array_equal(v1['target_event'][align], val.targets['event']):
        raise ValueError('v1 alignment changes event labels')
    return align


def v1_models(align):
    out = {}
    for run in sorted(V1.glob('runs/*_etp_*')):
        name = run.name.split('_s2026')[0]
        if name not in V1_STRONG or not (run / 'best_validation_predictions.npz').exists():
            continue
        seed = run.name.split('_s')[-1].split('_')[0]
        logits = np.load(run / 'best_validation_predictions.npz')['pred_event'][align]
        out[f'v1:{name.replace("formal_etp_", "").replace("baseline_calibration_etp_", "")}:s{seed}'] = softmax(logits.astype(np.float64))
    return out


@torch.no_grad()
def five_task_probabilities(config, val, run, device):
    status = read_json(run / 'status.json')
    if status['status'] != 'complete':
        raise ValueError(f'{run.name} is not complete')
    cfg = read_json(run / 'config.json')
    condition = config['conditions'][cfg['condition_id']]
    model = build(config, condition, read_json(Path(config['paths']['data']) / 'vocabulary.json')).to(device)
    model.load_state_dict(torch.load(run / 'best.pt', map_location=device, weights_only=True)['model'])
    model.eval()
    n = len(val.targets['anchor'])
    order = [np.arange(lo, min(lo + 256, n)) for lo in range(0, n, 256)]
    logits = []
    for x, _ in fit.loader(val, order, condition, config):
        logits.append(model(fit.move(x, device))['event_logits'].double().cpu().numpy())
    return softmax(np.concatenate(logits))


def gbdt_models(config, seeds):
    """Local-context trees (anchor and three previous units), with different early-stopping splits."""
    def features(split):
        d = SplitData(config, split)
        u, t = d.units, d.targets
        a = t['anchor']
        flip = u['ref_team'][a] != d.matches['home_team'][t['match_row']]
        frame = lambda xy: np.where(flip[:, None], 1 - xy, xy)
        cols = [u['event'][a], u['sub_code'][a], u['merged'][a], u['period'][a], u['seconds'][a] / 60, u['loc_valid'][a]] + list(frame(u['xy_home'][a]).T)
        for lag in (1, 2, 3):
            b = a - lag
            ok = b >= t['match_start']
            bb = np.where(ok, b, a)
            same = np.where(u['merged'][bb], 2, (u['part_team'][bb, 0] == u['ref_team'][a]).astype(int))
            cols += [np.where(ok, u['event'][bb], 0), np.where(ok, u['merged'][bb], 0), np.where(ok, same, 3),
                     np.where(ok, u['seconds'][a] - u['seconds'][bb], -1)] + list(np.where(ok[:, None], frame(u['xy_home'][bb]), -1).T)
        return np.stack(cols, 1).astype(np.float32), t['event']
    xt, yt = features('train')
    xv, _ = features('validation')
    categorical = np.zeros(xt.shape[1], bool)
    categorical[[0, 1]] = True
    out = {}
    for seed in seeds:
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, categorical_features=categorical,
                                             early_stopping=True, validation_fraction=0.1, random_state=seed)
        clf.fit(xt, yt)
        out[f'gbdt:local:s{seed}'] = clf.predict_proba(xv)
    return out


def union_accuracy(correct, names):
    return float(np.mean(np.any([correct[n] for n in names], axis=0)))


def stacking(probs, names, y, match):
    """Two-fold match-grouped logistic stacking on concatenated log-probabilities."""
    x = np.concatenate([np.log(np.clip(probs[n], 1e-9, 1)) for n in names], 1)
    matches = np.unique(match)
    rng = np.random.default_rng(20261014)
    half = set(rng.permutation(matches)[: len(matches) // 2].tolist())
    fold = np.array([m in half for m in match])
    pred = np.zeros(len(y), int)
    for train_fold in (True, False):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(x[fold == train_fold], y[fold == train_fold])
        pred[fold != train_fold] = clf.predict(x[fold != train_fold])
    return float(np.mean(pred == y))


def main(config, device):
    val = SplitData(config, 'validation')
    y = val.targets['event']
    match = val.targets['match_id']
    align = v1_alignment(val)
    probs = v1_models(align)
    runs = Path(config['paths']['runs'])
    for name in ('formal_hgt_full_s20260715', 'formal_hgt_events_only_s20260715', 'formal_hgt_no_player_s20260715', 'formal_seq_mean_s20260715',
                 'lr_hgt_full_lr0.0027', 'lr_seq_mean_lr0.0027'):
        probs[f'5task:{name.replace("formal_", "")}'] = five_task_probabilities(config, val, runs / name, device)
    probs.update(gbdt_models(config, (0, 1, 2)))
    argmax_only = {}
    for run in sorted(runs.glob('formal_hgt_*_s20260715')):
        key = f'5task:{run.name.replace("formal_", "")}'
        if key not in probs and read_json(run / 'status.json')['status'] == 'complete':
            argmax_only[key] = np.load(run / 'best_validation_statistics.npz')['event_pred']
    pred = {n: p.argmax(1) for n, p in probs.items()} | argmax_only
    correct = {n: pred[n] == y for n in pred}
    single = {n: float(c.mean()) for n, c in correct.items()}

    def topk(name, k):
        return float(np.mean((np.argsort(-probs[name], 1)[:, :k] == y[:, None]).any(1)))
    groups = {
        'seed replicas: v1 D00 (3 seeds)': [n for n in pred if n.startswith('v1:D00:')],
        'seed replicas: v1 B_CXPT (3 seeds)': [n for n in pred if n.startswith('v1:B_CXPT:')],
        'seed replicas: v1 C2 current_phase_control (3 seeds)': [n for n in pred if n.startswith('v1:C2_current_phase_control:')],
        'seed replicas: GBDT local (3 split seeds)': [n for n in pred if n.startswith('gbdt:')],
        'diverse architectures (HGT full, sequence, v1 Transformer D00, GBDT)': ['5task:hgt_full_s20260715', '5task:seq_mean_s20260715',
                                                                                 'v1:D00:s20260715', 'gbdt:local:s0'],
        'all five-task HGT conditions (seed 20260715)': [n for n in pred if n.startswith('5task:hgt_') and n.endswith('s20260715')],
        'all v1 strong runs': [n for n in pred if n.startswith('v1:')],
        'everything': list(pred),
    }
    report = dict(n=int(len(y)), single={n: round(100 * v, 2) for n, v in sorted(single.items(), key=lambda kv: -kv[1])}, groups={})
    for g, names in groups.items():
        names = [n for n in names if n in pred]
        if len(names) < 2:
            continue
        with_probs = [n for n in names if n in probs]
        entry = dict(models=len(names), best_single=round(100 * max(single[n] for n in names), 2), oracle_union=round(100 * union_accuracy(correct, names), 2),
                     all_agree_share=round(100 * float(np.mean(np.all([pred[n] == pred[names[0]] for n in names], 0))), 2))
        if len(with_probs) >= 2:
            avg = np.mean([probs[n] for n in with_probs], 0)
            entry.update(probability_average=round(100 * float(np.mean(avg.argmax(1) == y)), 2),
                         stacking_match_cv=round(100 * stacking(probs, with_probs, y, match), 2))
        report['groups'][g] = entry
    # Union growth: random orders of the strong models, mean over 20 orders.
    pool = [n for n in pred if not n.startswith('gbdt:')]
    rng = np.random.default_rng(0)
    growth = {}
    for _ in range(20):
        order = rng.permutation(pool)
        acc = np.zeros(len(y), bool)
        for k, n in enumerate(order, 1):
            acc |= correct[n]
            growth.setdefault(k, []).append(acc.mean())
    report['union_growth'] = {k: round(100 * float(np.mean(v)), 2) for k, v in growth.items() if k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, len(pool))}
    reference = '5task:hgt_full_s20260715'
    report['top_k_reference'] = {name: {k: round(100 * topk(name, k), 2) for k in (1, 2, 3)} for name in (reference, 'v1:D00:s20260715', 'gbdt:local:s0')}
    union = np.any([correct[n] for n in groups['diverse architectures (HGT full, sequence, v1 Transformer D00, GBDT)']], 0)
    best = correct[reference]
    gained = union & ~best
    report['diverse_union_gain_by_true_class'] = {RAW10[k]: dict(targets=int((y == k).sum()), share_of_gain=round(100 * float(gained[y == k].sum() / max(gained.sum(), 1)), 1),
                                                              reference_accuracy=round(100 * float(best[y == k].mean()), 1), union_accuracy=round(100 * float(union[y == k].mean()), 1))
                                                  for k in range(10)}
    # How confident is the reference model on samples that only other models get right?
    p_ref = probs[reference]
    margin = np.sort(p_ref, 1)[:, -1] - np.sort(p_ref, 1)[:, -2]
    true_rank = (p_ref > p_ref[np.arange(len(y)), y][:, None]).sum(1) + 1
    report['reference_on_gained_samples'] = dict(count=int(gained.sum()), mean_top1_minus_top2_margin=round(float(margin[gained].mean()), 3),
                                                 mean_margin_all=round(float(margin.mean()), 3),
                                                 true_class_rank_share={r: round(100 * float(np.mean(true_rank[gained] == r)), 1) for r in (2, 3, 4)})
    reports = Path(config['paths']['reports'])
    write_json(reports / 'oracle_union.json', report)
    np.savez_compressed(reports / 'oracle_union_predictions.npz', names=np.array(list(pred)), predictions=np.stack([pred[n] for n in pred]), target=y,
                        match_id=match)
    lines = ['# Oracle union of existing models (validation, merged-duel targets)', '', f"N = {report['n']}. Oracle = share of targets that at least one model in the group gets right (uses the label; diagnostic only).", '',
             '| group | models | best single | oracle union | all agree | probability average | stacking (match CV) |', '|---|---|---|---|---|---|---|']
    for g, e in report['groups'].items():
        lines.append(f"| {g} | {e['models']} | {e['best_single']} | {e['oracle_union']} | {e['all_agree_share']} | {e.get('probability_average', '')} | {e.get('stacking_match_cv', '')} |")
    lines += ['', 'Top-k accuracy of single models: ' + json.dumps(report['top_k_reference']), '', 'Union growth with the number of models (mean of 20 random orders): ' +
              json.dumps(report['union_growth']), '', '## Where the diverse union gains (relative to HGT full)', '', '| true class | targets | share of gain | HGT full acc | union acc |', '|---|---|---|---|---|']
    for c, e in report['diverse_union_gain_by_true_class'].items():
        lines.append(f"| {c} | {e['targets']} | {e['share_of_gain']} | {e['reference_accuracy']} | {e['union_accuracy']} |")
    lines += ['', 'HGT full on samples gained by the union: ' + json.dumps(report['reference_on_gained_samples'])]
    (reports / 'oracle_union.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', required=True)
    args = parser.parse_args()
    main(read_json(args.config), args.device)
