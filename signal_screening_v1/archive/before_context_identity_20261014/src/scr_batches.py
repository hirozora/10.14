"""Batches for the signal screen: identity-free context tokens and per-candidate inputs.

Player identities enter only through the candidate side. The context sees, for every window
unit, its type, subtype, period, time, location, merged flag, the team relation of its
participants to the reference team and the online phase relations.
"""
from pathlib import Path

import numpy as np

from scr_candidates import setup, anchor_clock, on_pitch

RECENCY_BINS = ('anchor actor', '1-4 units ago', '5-16 units ago', '17-79 units ago', 'not in window')


class ScreenData:
    def __init__(self, config, split):
        setup(config)
        from batches import SplitData
        self.base = SplitData(config, split)
        self.config = config
        self.timing = dict(np.load(Path(config['paths']['candidates']) / f'{split}_candidates.npz'))
        self.K = config['dataset']['history_events_per_candidate']


def phase_relations(g):
    """Per unit: in the anchor's phase, phase owned by the reference team, phase known."""
    phase, valid = g['phase'], g['valid']
    B = phase.shape[0]
    owner = np.zeros_like(phase)
    for k in range(1, int(phase.max()) + 1 if phase.size else 1):
        member = (phase == k) & valid
        first = member.argmax(1)
        team = g['part_team'][np.arange(B), first, 0]
        owner = np.where(member, team[:, None], owner)
    known = phase > 0
    return np.stack([known & (phase == phase[:, -1:]), known & (owner == g['ref'][:, None]), known], -1).astype(np.float32) * valid[..., None]


def membership(g, roster_player):
    """(B, W, S): roster slot s takes part in window unit w."""
    p = g['part_player']
    return ((p[:, :, :, None] == roster_player[:, None, None, :]) & (p[:, :, :, None] > 0)).any(2) & g['valid'][:, :, None]


def history_positions(member, K):
    """(B, S, K) window positions of each slot's most recent K units (most recent first), -1 when absent."""
    B, W, S = member.shape
    rev = member[:, ::-1, :]
    rank = np.cumsum(rev, axis=1) * rev
    b, w_rev, s = np.nonzero((rank >= 1) & (rank <= K))
    pos = np.full((B, S, K), -1, np.int64)
    pos[b, s, rank[b, w_rev, s] - 1] = W - 1 - w_rev
    return pos


def shuffled_positions(pos, same_team_groups, rng_seed, anchors):
    """Swap whole histories among candidates of the same team, independently per sample."""
    out = pos.copy()
    for i, (groups, anchor) in enumerate(zip(same_team_groups, anchors)):
        rng = np.random.default_rng([rng_seed, int(anchor)])
        for slots in groups:
            if len(slots) > 1:
                out[i, slots] = pos[i, rng.permutation(slots)]
    return out


def screen_batch(data, rows, condition):
    from batches import gather, event_numeric, roster_features
    b0 = data.base
    g = gather(b0, rows)
    B, W = g['valid'].shape
    valid = g['valid']
    t = b0.targets
    anchor = t['anchor'][rows]
    roster = roster_features(b0, g)
    roster_player = b0.roster['player'][g['match_row']]
    roster_team = b0.roster['team'][g['match_row']]
    clock = anchor_clock(b0.units, anchor)
    candidate = on_pitch(data.timing, b0.roster, g['match_row'], anchor, clock)
    same_team = roster_team == g['ref'][:, None]
    team_relation = np.where(g['part_team'] > 0, np.where(g['part_team'] == g['ref'][:, None, None], 1, 2), 0) * valid[..., None]
    order = np.where(valid, np.cumsum(valid, 1), 0)
    distance = np.where(valid, g['index'] - g['index'][:, -1:], 0)
    context = dict(kind=np.where(valid, g['event'] - 1, 0), sub=g['sub_code'], period=g['period'], numeric=event_numeric(g), valid=valid,
                   team_relation=team_relation, phase=phase_relations(g), order=order, distance=distance, seconds=g['seconds'], period_raw=g['period'])
    member = membership(g, roster_player)
    count = member.sum(1)
    seen = count > 0
    last = np.where(seen, W - 1 - member[:, ::-1, :].argmax(1), 0)
    units_ago = np.where(seen, W - 1 - last, W)
    seconds_ago = np.where(seen, g['seconds'][:, -1:] - np.take_along_axis(g['seconds'], last, 1), 0.0)
    same_period = np.take_along_axis(g['period'], last, 1) == g['period'][:, -1:]
    seconds_ago = np.where(seen & same_period, np.maximum(seconds_ago, 0.0), 3600.0)
    anchor_actor = member[:, -1, :]
    summary = np.stack([np.log1p(count), units_ago / 80.0, np.log1p(seconds_ago) / 8.0, (~seen).astype(np.float32), anchor_actor.astype(np.float32)], -1).astype(np.float32)
    cand = dict(vocab=roster['vocab'], role=roster['role'], foot=roster['foot'], numeric=roster['numeric'], same_team=same_team.astype(np.float32),
                mask=candidate, summary=summary * candidate[..., None])
    if condition['history'] != 'none':
        pos = history_positions(member, data.K)
        if condition['history'] == 'pooled_shuffled_within_team':
            groups = [[np.flatnonzero(candidate[i] & (roster_team[i] == tm)) for tm in np.unique(roster_team[i][candidate[i]])] for i in range(B)]
            pos = shuffled_positions(pos, groups, data.config['shuffle_seed'], anchor)
        has = pos >= 0
        p = np.maximum(pos, 0)
        bi = np.arange(B)[:, None, None]
        num = event_numeric(g)
        units_before = (W - 1 - p) / 80.0
        secs_before = g['seconds'][:, -1:, None] - g['seconds'][bi, p]
        secs_before = np.where(g['period'][bi, p] == g['period'][:, -1:, None], np.maximum(secs_before, 0.0), 3600.0)
        unit_team = np.where(g['merged'][bi, p], 3, np.where(g['part_team'][bi, p, 0] == g['ref'][:, None, None], 1, 2))
        cand.update(h_kind=np.where(has, g['event'][bi, p] - 1, 0), h_sub=np.where(has, g['sub_code'][bi, p], 0), h_team=np.where(has, unit_team, 0),
                    h_numeric=np.where(has[..., None], np.concatenate([num[bi, p][..., 3:7], units_before[..., None], np.log1p(secs_before)[..., None] / 8.0], -1), 0.0).astype(np.float32),
                    h_mask=has & candidate[..., None])
    # targets: soft weights restricted to reachable candidates
    slot = t['player_slot'][rows]
    weight = t['player_weight'][rows].copy()
    reach = np.where(slot >= 0, np.take_along_axis(candidate, np.maximum(slot, 0), 1), False)
    weight = np.where(reach, weight, 0.0)
    total = weight.sum(1, keepdims=True)
    loss_weight = np.where(total > 0, weight / np.maximum(total, 1e-9), 0.0).astype(np.float32)
    known = t['player_mask'][rows]
    duel = t['target_duel'][rows]
    true_slot = np.maximum(slot[:, 0], 0)
    true_member = np.take_along_axis(member, true_slot[:, None, None].repeat(W, 1), 2)[..., 0]
    seen_true = true_member.any(1)
    last_true = np.where(seen_true, W - 1 - true_member[:, ::-1].argmax(1), -1)
    ago = np.where(seen_true, W - 1 - last_true, -1)
    recency = np.select([ago == 0, (ago >= 1) & (ago <= 4), (ago >= 5) & (ago <= 16), ago >= 17], [0, 1, 2, 3], 4)
    targets = dict(slot=slot, eval_weight=t['player_weight'][rows], loss_weight=loss_weight, known=known, duel=duel, reachable=total[:, 0] > 0,
                   recency=np.where(known & ~duel, recency, -1), match_id=t['match_id'][rows], row=rows)
    return dict(context=context, candidates=cand, batch_size=B), targets
