"""Batches for the signal screen: context tokens and per-candidate inputs.

The context sees, for every window unit, its type, subtype, period, time, location, merged flag,
the team relation of its participants to the reference team and the online phase relations.
Player identities enter only through the candidate side, except in conditions with
context_identity, whose context also carries which roster slots took part in each unit.

Stage 3 adds two condition fields. scope 'match' extends the candidate summary with match-level
counts and gives the history the candidate's latest K acting units of the whole match up to the
anchor (not only the window). pairs 'anchor' / 'window' add player-player relations: transitions
between different players in consecutive units of the match, counted up to the anchor.
"""
from pathlib import Path

import numpy as np

from scr_candidates import setup, anchor_clock, on_pitch

RECENCY_BINS = ('anchor actor', '1-4 units ago', '5-16 units ago', '17-79 units ago', 'not in window')
BIG = 1 << 20  # unit indices of a split stay below this
SCOPES = ('window', 'match')
PAIRS = ('none', 'anchor', 'window', 'window_shuffled_within_team')


class ScreenData:
    def __init__(self, config, split):
        setup(config)
        from batches import SplitData
        self.base = SplitData(config, split)
        self.config = config
        self.timing = dict(np.load(Path(config['paths']['candidates']) / f'{split}_candidates.npz'))
        self.K = config['dataset']['history_events_per_candidate']
        self.match = match_tables(self.base)


def match_tables(base):
    """Match-wide lookups, built once per split.

    slot (n_units, 2): roster slots of each unit's participants, -1 when unknown or not in the roster.
    act_key: sorted keys (match_row * S + slot) * BIG + unit over the acting units of every slot.
    t_unit, t_from, t_to: transitions between different players in consecutive units of a match,
    ordered by the unit they reach; a transition reaching unit i is known from unit i on.
    """
    u, r = base.units, base.roster
    m = u['match_row']
    n, S = len(m), r['player'].shape[1]
    if n >= BIG or np.any(np.diff(m) < 0):
        raise ValueError('Units must be ordered by match and fewer than BIG')
    p = u['part_player']
    hit = (p[:, :, None] == r['player'][m][:, None, :]) & (p[:, :, None] > 0) & r['valid'][m][:, None, :]
    if (hit.sum(2) > 1).any():
        raise ValueError('A participant matches more than one roster slot')
    slot = np.where(hit.any(2), hit.argmax(2), -1)
    if ((slot[:, 0] == slot[:, 1]) & (slot[:, 0] >= 0)).any():
        raise ValueError('A unit lists the same player twice')
    unit = np.arange(n)
    seg = np.concatenate([m * S + slot[:, 0], m * S + slot[:, 1]])
    keep = np.concatenate([slot[:, 0] >= 0, slot[:, 1] >= 0])
    act_key = np.sort(seg[keep] * BIG + np.concatenate([unit, unit])[keep])
    same = m[1:] == m[:-1]
    reach, src, dst = [], [], []
    for j in range(2):
        for k in range(2):
            a, b = slot[:-1, j], slot[1:, k]
            ok = same & (a >= 0) & (b >= 0) & (a != b)
            reach.append(unit[1:][ok])
            src.append(a[ok])
            dst.append(b[ok])
    reach, src, dst = np.concatenate(reach), np.concatenate(src), np.concatenate(dst)
    order = np.argsort(reach, kind='stable')
    return dict(slot=slot, act_key=act_key, t_unit=reach[order], t_from=src[order], t_to=dst[order])


def match_history(data, match_row, anchor, S, K):
    """(B, S, K) each slot's latest K acting units of the match up to the anchor (most recent first, -1 when absent),
    and (B, S) the number of such units."""
    key = data.match['act_key']
    seg = match_row[:, None] * S + np.arange(S)[None, :]
    lo = np.searchsorted(key, seg * BIG, 'left')
    hi = np.searchsorted(key, seg * BIG + anchor[:, None], 'right')
    pos = hi[..., None] - 1 - np.arange(K)
    return np.where(pos >= lo[..., None], key[np.maximum(pos, 0)] % BIG, -1), hi - lo


def pair_counts(data, match_start, anchor, S):
    """count[b, a, s]: transitions from slot a to slot s reaching units of the match up to the anchor;
    last[b, a, s]: the latest unit such a transition reached, -1 when none."""
    t = data.match
    B = len(anchor)
    lo = np.searchsorted(t['t_unit'], match_start, 'left')
    hi = np.searchsorted(t['t_unit'], anchor, 'right')
    n = hi - lo
    idx = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
    bins = (np.repeat(np.arange(B), n) * S + t['t_from'][idx]) * S + t['t_to'][idx]
    count = np.bincount(bins, minlength=B * S * S).reshape(B, S, S)
    last = np.full(B * S * S, -1, np.int64)
    np.maximum.at(last, bins, t['t_unit'][idx])
    return count, last.reshape(B, S, S)


def pair_values(n_in, n_out, last_in, anchor):
    """log1p transitions into the candidate, log1p transitions out of it, log1p(units since the latest one in)/8, none in yet."""
    since = np.where(last_in >= 0, np.log1p(np.maximum(anchor - last_in, 0)) / 8.0, 0.0)
    return np.stack([np.log1p(n_in), np.log1p(n_out), since, last_in < 0], -1).astype(np.float32)


def within_team_permutation(groups, rng_seed, anchors, S):
    """The slot permutation of shuffled_positions as indices: shuffled[i, s] = original[i, perm[i, s]]."""
    perm = np.tile(np.arange(S), (len(anchors), 1))
    for i, (slot_groups, anchor) in enumerate(zip(groups, anchors)):
        rng = np.random.default_rng([rng_seed, int(anchor)])
        for slots in slot_groups:
            if len(slots) > 1:
                perm[i, slots] = rng.permutation(slots)
    return perm


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
    if condition['scope'] not in SCOPES or condition['pairs'] not in PAIRS:
        raise ValueError(f"Unknown scope or pairs: {condition['scope']}, {condition['pairs']}")
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
    if condition['context_identity']:
        context['member'] = member
    count = member.sum(1)
    seen = count > 0
    last = np.where(seen, W - 1 - member[:, ::-1, :].argmax(1), 0)
    units_ago = np.where(seen, W - 1 - last, W)
    seconds_ago = np.where(seen, g['seconds'][:, -1:] - np.take_along_axis(g['seconds'], last, 1), 0.0)
    same_period = np.take_along_axis(g['period'], last, 1) == g['period'][:, -1:]
    seconds_ago = np.where(seen & same_period, np.maximum(seconds_ago, 0.0), 3600.0)
    anchor_actor = member[:, -1, :]
    summary = np.stack([np.log1p(count), units_ago / 80.0, np.log1p(seconds_ago) / 8.0, (~seen).astype(np.float32), anchor_actor.astype(np.float32)], -1).astype(np.float32)
    S = roster_player.shape[1]

    def team_groups():
        return [[np.flatnonzero(candidate[i] & (roster_team[i] == tm)) for tm in np.unique(roster_team[i][candidate[i]])] for i in range(B)]
    if condition['scope'] == 'match':
        acted, acted_count = match_history(data, g['match_row'], anchor, S, data.K)
        since = np.where(acted_count > 0, np.log1p(anchor[:, None] - acted[:, :, 0]) / 8.0, 0.0)
        summary = np.concatenate([summary, np.stack([np.log1p(acted_count), since, acted_count == 0], -1).astype(np.float32)], -1)
    cand = dict(vocab=roster['vocab'], role=roster['role'], foot=roster['foot'], numeric=roster['numeric'], same_team=same_team.astype(np.float32),
                mask=candidate, summary=summary * candidate[..., None])
    if condition['history'] != 'none' and condition['scope'] == 'match':
        # The same event features as the window history, for units anywhere in the match up to the anchor; the
        # units-before feature saturates at 79/80, and two features mark units outside the window and their distance.
        hist = acted
        if condition['history'] == 'pooled_shuffled_within_team':
            hist = shuffled_positions(hist, team_groups(), data.config['shuffle_seed'], anchor)
        elif condition['history'] != 'pooled':
            raise ValueError(f"Match-scope history supports pooling only: {condition['history']}")
        u = b0.units
        has = hist >= 0
        q = np.maximum(hist, 0)
        flip = (g['ref'] != g['home'])[:, None, None, None]
        xy = np.where(flip, 1.0 - u['xy_home'][q], u['xy_home'][q]).astype(np.float32)
        xy = np.where(u['loc_valid'][q][..., None], xy, 0.0).astype(np.float32)
        num = np.stack([xy[..., 0], xy[..., 1], u['loc_valid'][q], u['merged'][q]], -1).astype(np.float32)
        before = anchor[:, None, None] - q
        secs_before = u['seconds'][anchor][:, None, None] - u['seconds'][q]
        secs_before = np.where(u['period'][q] == u['period'][anchor][:, None, None], np.maximum(secs_before, 0.0), 3600.0)
        unit_team = np.where(u['merged'][q], 3, np.where(u['part_team'][q, 0] == g['ref'][:, None, None], 1, 2))
        numeric = np.concatenate([num, np.minimum(before, W - 1)[..., None] / 80.0, np.log1p(secs_before)[..., None] / 8.0,
                                  (before >= W)[..., None], np.log1p(before)[..., None] / 8.0], -1)
        cand.update(h_kind=np.where(has, u['event'][q] - 1, 0), h_sub=np.where(has, u['sub_code'][q], 0), h_team=np.where(has, unit_team, 0),
                    h_numeric=np.where(has[..., None], numeric, 0.0).astype(np.float32), h_mask=has & candidate[..., None])
    if condition['pairs'] != 'none':
        count_ts, last_ts = pair_counts(data, t['match_start'][rows], anchor, S)
        bi = np.arange(B)[:, None]
        actors = data.match['slot'][anchor]
        ok = (actors >= 0)[..., None]
        a = np.maximum(actors, 0)
        n_in = (count_ts[bi, a] * ok).sum(1)
        n_out = (count_ts.transpose(0, 2, 1)[bi, a] * ok).sum(1)
        last_in = np.where(ok, last_ts[bi, a], -1).max(1)
        pair_anchor = pair_values(n_in, n_out, last_in, anchor[:, None]) * candidate[..., None]
        if condition['pairs'] == 'anchor':
            cand.update(pair_anchor=pair_anchor)
        else:
            # One token per other player who acted in the window: the pair values with the candidate, that player's
            # window appearances, units since their latest one, anchor-actor flag and whether they share the candidate's team.
            relation = pair_values(count_ts.transpose(0, 2, 1), count_ts, last_ts.transpose(0, 2, 1), anchor[:, None, None])
            actor = np.stack([np.log1p(count), units_ago / 80.0, anchor_actor], -1).astype(np.float32)
            tokens = np.concatenate([relation, np.broadcast_to(actor[:, None], (B, S, S, 3)),
                                     (roster_team[:, :, None] == roster_team[:, None, :])[..., None]], -1).astype(np.float32)
            pair_mask = seen[:, None, :] & ~np.eye(S, dtype=bool)[None] & candidate[..., None]
            if condition['pairs'] == 'window_shuffled_within_team':
                perm = within_team_permutation(team_groups(), data.config['shuffle_seed'], anchor, S)
                pair_anchor, tokens, pair_mask = pair_anchor[bi, perm], tokens[bi, perm], pair_mask[bi, perm]
            cand.update(pair_anchor=pair_anchor, pair_tokens=np.where(pair_mask[..., None], tokens, 0.0).astype(np.float32), pair_mask=pair_mask)
    if condition['history'] != 'none' and condition['scope'] == 'window':
        pos = history_positions(member, data.K)
        if condition['history'] == 'pooled_shuffled_within_team':
            pos = shuffled_positions(pos, team_groups(), data.config['shuffle_seed'], anchor)
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
