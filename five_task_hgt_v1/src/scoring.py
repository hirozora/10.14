"""Per-sample statistics, per-match sufficient statistics, metrics and the paired match bootstrap."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from models import task_parts

VIEWS = ('all', 'non_duel_target', 'duel_target', 'duel_anchor')
SCALARS = ('event_n', 'event_ce', 'time_n', 'time_loss', 'time_abs', 'position_n', 'position_loss', 'position_m',
           'team_n', 'team_ce', 'team_credit', 'player_n', 'player_ce', 'player_top1', 'player_top3', 'player_top5')
WIDTH = len(SCALARS) + 100


@torch.no_grad()
def sample_statistics(pred, y):
    """Per-sample values; Team and Player credits use the soft targets (top-k credit = target mass in the top k)."""
    parts = task_parts(pred, y)
    out = dict(event_ce=parts['event'], event_pred=pred['event_logits'].argmax(-1), time_loss=parts['time'],
               time_abs=(pred['time_seconds'] - y['time']).abs(), position_loss=parts['position'],
               position_m=torch.linalg.norm((pred['position_xy'] - y['xy']) * torch.tensor([105.0, 68.0], device=y['xy'].device), dim=-1),
               team_ce=parts['team'], player_ce=parts['player'])
    q = torch.stack((1 - y['team_same'], y['team_same']), -1)
    out['team_credit'] = torch.gather(q, 1, pred['team_logits'].argmax(-1, keepdim=True)).squeeze(1)
    order = pred['player_scores'].argsort(-1, descending=True)
    rank_of = torch.empty_like(order)
    rank_of.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
    slot = y['player_slot'].long()
    rank = torch.gather(rank_of, 1, slot.clamp_min(0))
    for k in (1, 3, 5):
        out[f'player_top{k}'] = ((rank < k) & (slot >= 0)).float().mul(y['player_weight']).sum(-1)
    return {k: v.cpu().numpy() for k, v in out.items()}


def view_mask(y, view):
    n = len(y['event'])
    if view == 'all':
        return np.ones(n, bool)
    if view == 'non_duel_target':
        return ~y['target_duel']
    if view == 'duel_target':
        return y['target_duel'].copy()
    if view == 'duel_anchor':
        return y['anchor_duel'].copy()
    raise ValueError(view)


def match_tables(stats, y):
    """{view: (matches, M x WIDTH table)} of additive per-match statistics."""
    matches, inverse = np.unique(y['match_id'], return_inverse=True)
    tables = {}
    for view in VIEWS:
        keep = view_mask(y, view)
        table = np.zeros((len(matches), WIDTH))

        def add(column, values, mask):
            np.add.at(table[:, SCALARS.index(column)], inverse, np.where(mask, values, 0.0))
        add('event_n', 1.0, keep)
        add('event_ce', stats['event_ce'], keep)
        tm, pm = keep & y['time_mask'], keep & y['pos_mask']
        add('time_n', 1.0, tm); add('time_loss', stats['time_loss'], tm); add('time_abs', stats['time_abs'], tm)
        add('position_n', 1.0, pm); add('position_loss', stats['position_loss'], pm); add('position_m', stats['position_m'], pm)
        team = keep & y['team_mask']
        add('team_n', 1.0, team); add('team_ce', stats['team_ce'], team); add('team_credit', stats['team_credit'], team)
        player = keep & y['player_mask']
        add('player_n', 1.0, player); add('player_ce', stats['player_ce'], player)
        for k in (1, 3, 5):
            add(f'player_top{k}', stats[f'player_top{k}'], player)
        rows = np.flatnonzero(keep)
        np.add.at(table, (inverse[rows], len(SCALARS) + y['event'][rows] * 10 + stats['event_pred'][rows]), 1)
        tables[view] = table
    return matches, tables


def metric_array(table):
    def col(name):
        return table[..., SCALARS.index(name)]

    def ratio(a, b):
        return col(a) / np.maximum(col(b), 1)
    confusion = table[..., len(SCALARS):].reshape(*table.shape[:-1], 10, 10)
    diag = np.diagonal(confusion, axis1=-2, axis2=-1)
    f1 = 2 * diag / np.maximum(confusion.sum(-1) + confusion.sum(-2), 1)
    event_ce, time_loss, position_loss, team_ce, player_ce = (ratio('event_ce', 'event_n'), ratio('time_loss', 'time_n'),
                                                              ratio('position_loss', 'position_n'), ratio('team_ce', 'team_n'),
                                                              ratio('player_ce', 'player_n'))
    return dict(five_task_loss=(0.2 * event_ce / math.log(10) + time_loss + position_loss + 0.05 * team_ce + 0.4 * player_ce) / 3,
                etp_loss=(0.2 * event_ce / math.log(10) + time_loss + position_loss) / 3,
                event_ce=event_ce, event_accuracy=diag.sum(-1) / np.maximum(confusion.sum((-1, -2)), 1), event_macro_f1=f1.mean(-1),
                event_per_class_f1=f1, time_loss=time_loss, time_mae_seconds=ratio('time_abs', 'time_n'), position_loss=position_loss,
                position_error_metres=ratio('position_m', 'position_n'), team_nll=team_ce, team_accuracy=ratio('team_credit', 'team_n'),
                player_nll=player_ce, player_top1=ratio('player_top1', 'player_n'), player_top3=ratio('player_top3', 'player_n'),
                player_top5=ratio('player_top5', 'player_n'), samples=col('event_n'))


def json_metrics(metrics):
    return {k: v.tolist() if isinstance(v, np.ndarray) else float(v) for k, v in metrics.items()}


def bootstrap_weights(matches, replicates, seed):
    return np.random.default_rng(seed).multinomial(matches, np.full(matches, 1 / matches), replicates)


def paired_bootstrap(a, b, weights):
    """a, b: seeds x matches x WIDTH tables of two conditions on the same matches. Difference a - b, seeds weighted equally."""
    if a.shape != b.shape or a.ndim != 3 or a.shape[1] != weights.shape[1]:
        raise ValueError('Bootstrap dimensions differ')
    am, bm = metric_array(a.sum(1)), metric_array(b.sum(1))
    keys = [k for k in am if k not in ('event_per_class_f1', 'samples')]
    draws = {k: [] for k in keys}
    for w in np.array_split(weights, 40):
        aa = metric_array(np.einsum('bm,smk->bsk', w, a))
        bb = metric_array(np.einsum('bm,smk->bsk', w, b))
        for k in keys:
            draws[k].append((aa[k] - bb[k]).mean(1))
    result = {}
    for k in keys:
        delta = am[k] - bm[k]
        draw = np.concatenate(draws[k])
        result[k] = dict(per_seed=delta.tolist(), mean=float(delta.mean()), sd=float(delta.std(ddof=1)) if len(delta) > 1 else None,
                         ci95=np.quantile(draw, [0.025, 0.975]).tolist())
    return result
