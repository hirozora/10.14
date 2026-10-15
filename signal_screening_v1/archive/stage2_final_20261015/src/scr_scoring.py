"""Per-sample player statistics, per-match sufficient statistics and the paired match bootstrap."""
import numpy as np
import torch

from scr_batches import RECENCY_BINS
from scr_models import player_loss

COLUMNS = (['single_n', 'single_top1', 'single_top3', 'single_top5', 'single_rr', 'single_reach_n', 'single_nll', 'single_unreachable',
            'duel_n', 'duel_top1', 'duel_top3', 'duel_reach_n', 'duel_nll']
           + [f'recency{b}_n' for b in range(5)] + [f'recency{b}_top1' for b in range(5)])


@torch.no_grad()
def sample_statistics(scores, y):
    per, mask = player_loss(scores, y)
    order = scores.argsort(-1, descending=True)
    rank_of = torch.empty_like(order)
    rank_of.scatter_(1, order, torch.arange(order.shape[1], device=order.device).expand_as(order))
    slot = y['slot'].long()
    rank = torch.gather(rank_of, 1, slot.clamp_min(0))
    candidate = torch.isfinite(torch.gather(scores, 1, slot.clamp_min(0)))
    hit = lambda k: ((rank < k) & (slot >= 0) & candidate).float().mul(y['eval_weight']).sum(-1)
    rr = torch.where((slot[:, 0] >= 0) & candidate[:, 0], 1.0 / (rank[:, 0].float() + 1), 0.0)
    out = dict(nll=torch.where(mask, per, 0.0), nll_mask=mask, top1=hit(1), top3=hit(3), top5=hit(5), rr=rr)
    return {k: v.cpu().numpy() for k, v in out.items()}


def match_table(stats, y):
    matches, inverse = np.unique(y['match_id'], return_inverse=True)
    table = np.zeros((len(matches), len(COLUMNS)))

    def add(name, values, mask):
        np.add.at(table[:, COLUMNS.index(name)], inverse, np.where(mask, values, 0.0))
    single = y['known'] & ~y['duel']
    duel = y['known'] & y['duel']
    add('single_n', 1.0, single)
    for k in ('top1', 'top3', 'top5'):
        add(f'single_{k}', stats[k], single)
    add('single_rr', stats['rr'], single)
    add('single_reach_n', 1.0, single & stats['nll_mask'])
    add('single_nll', stats['nll'], single & stats['nll_mask'])
    add('single_unreachable', 1.0, single & ~y['reachable'])
    add('duel_n', 1.0, duel)
    add('duel_top1', stats['top1'], duel)
    add('duel_top3', stats['top3'], duel)
    add('duel_reach_n', 1.0, duel & stats['nll_mask'])
    add('duel_nll', stats['nll'], duel & stats['nll_mask'])
    for b in range(5):
        add(f'recency{b}_n', 1.0, single & (y['recency'] == b))
        add(f'recency{b}_top1', stats['top1'], single & (y['recency'] == b))
    return matches, table


def metrics(table):
    def col(name):
        return table[..., COLUMNS.index(name)]

    def ratio(a, b):
        return col(a) / np.maximum(col(b), 1)
    out = dict(single_top1=ratio('single_top1', 'single_n'), single_top3=ratio('single_top3', 'single_n'), single_top5=ratio('single_top5', 'single_n'),
               single_mrr=ratio('single_rr', 'single_n'), single_nll=ratio('single_nll', 'single_reach_n'),
               single_unreachable_share=ratio('single_unreachable', 'single_n'), duel_top1=ratio('duel_top1', 'duel_n'),
               duel_top3=ratio('duel_top3', 'duel_n'), duel_nll=ratio('duel_nll', 'duel_reach_n'),
               player_nll_reachable=(col('single_nll') + col('duel_nll')) / np.maximum(col('single_reach_n') + col('duel_reach_n'), 1))
    for b, name in enumerate(RECENCY_BINS):
        out[f'top1[{name}]'] = ratio(f'recency{b}_top1', f'recency{b}_n')
    return out


def json_metrics(m):
    return {k: float(v) for k, v in m.items()}


def bootstrap_weights(matches, replicates, seed):
    return np.random.default_rng(seed).multinomial(matches, np.full(matches, 1 / matches), replicates)


def paired_bootstrap(a, b, weights):
    """a, b: seeds x matches x columns; difference a - b with seeds weighted equally."""
    am, bm = metrics(a.sum(1)), metrics(b.sum(1))
    draws = {k: [] for k in am}
    for w in np.array_split(weights, 40):
        aa = metrics(np.einsum('bm,smk->bsk', w, a))
        bb = metrics(np.einsum('bm,smk->bsk', w, b))
        for k in draws:
            draws[k].append((aa[k] - bb[k]).mean(1))
    out = {}
    for k in am:
        delta = am[k] - bm[k]
        draw = np.concatenate(draws[k])
        out[k] = dict(per_seed=delta.tolist(), mean=float(delta.mean()), ci95=np.quantile(draw, [0.025, 0.975]).tolist())
    return out
