"""Window gathering, online possession phases, heterogeneous graphs and sequence inputs.

All inputs of a sample come from the units up to and including its anchor and from
the pre-match roster of its match. Targets are read only by `targets_of`.
"""
import sys
from pathlib import Path

import numpy as np
import torch

# Official NMSTPP zone centres in percent coordinates (soccer_cohig_publish/src/cohig/data/constants.py).
ZONE_CENTERS = np.array([
    (8.5, 89.45), (25.25, 89.45), (41.75, 89.45), (58.25, 89.45), (74.75, 89.45), (91.5, 89.45),
    (8.5, 10.55), (25.25, 10.55), (41.75, 10.55), (58.25, 10.55), (74.75, 10.55), (91.5, 10.55),
    (33.5, 71.05), (66.5, 71.05), (33.5, 50.0), (66.5, 50.0), (33.5, 28.95), (66.5, 28.95),
    (8.5, 50.0), (91.5, 50.0)], np.float32) / 100.0
GAP_TYPES = ('gap_0_2s', 'gap_2_5s', 'gap_5_15s', 'gap_15_60s', 'gap_60plus', 'period_break')
UNIT_FIELDS = ('event', 'raw_sub', 'sub_code', 'period', 'seconds', 'xy_home', 'loc_valid', 'merged', 'part_player', 'part_team', 'ref_team')


def parent_phases(config):
    if config['parent_src'] not in sys.path:
        sys.path.insert(0, config['parent_src'])
    from data import phases
    return phases


class SplitData:
    """Arrays of one split; read-only after construction."""

    def __init__(self, config, split):
        root = Path(config['paths']['data'])
        self.config = config
        self.split = split
        self.units = dict(np.load(root / f'{split}_units.npz'))
        self.targets = dict(np.load(root / f'{split}_targets.npz'))
        rosters = dict(np.load(root / f'{split}_rosters.npz'))
        self.roster = {k[7:]: v for k, v in rosters.items() if k.startswith('roster_')}
        self.matches = {k: v for k, v in rosters.items() if not k.startswith('roster_')}
        self.window = config['dataset']['window_units']
        self.phases = parent_phases(config)
        # Fixed within-team relabelling of roster slots per match, used only by the shuffled control.
        self.shuffle = np.tile(np.arange(self.roster['player'].shape[1]), (len(self.matches['match_id']), 1))
        for row, mid in enumerate(self.matches['match_id']):
            rng = np.random.default_rng([config['shuffle_seed'], int(mid)])
            for side in ('home_team', 'away_team'):
                slots = np.flatnonzero(self.roster['valid'][row] & (self.roster['team'][row] == self.matches[side][row]))
                self.shuffle[row, slots] = rng.permutation(slots)


def gather(data, rows):
    """Left-padded windows of `window` units ending at each anchor, in the anchor frame."""
    u, t = data.units, data.targets
    W = data.window
    anchor = t['anchor'][rows]
    offsets = np.arange(W - 1, -1, -1)
    index = anchor[:, None] - offsets[None]
    valid = index >= t['match_start'][rows, None]
    safe = np.where(valid, index, anchor[:, None])
    g = {k: u[k][safe].copy() for k in UNIT_FIELDS}
    match_row = t['match_row'][rows]
    home = data.matches['home_team'][match_row]
    ref = u['ref_team'][anchor]
    flip = ref != home
    xy = np.where(flip[:, None, None], 1.0 - g['xy_home'], g['xy_home']).astype(np.float32)
    g['loc_valid'] &= valid
    g['xy'] = np.where(g['loc_valid'][..., None], xy, 0.0).astype(np.float32)
    for k in ('event', 'raw_sub', 'sub_code', 'period', 'seconds', 'part_player', 'part_team', 'ref_team'):
        g[k][~valid] = 0
    g['merged'] &= valid
    previous_valid = np.zeros_like(valid)
    previous_valid[:, 1:] = valid[:, :-1]
    same_period = np.zeros_like(valid)
    same_period[:, 1:] = g['period'][:, 1:] == g['period'][:, :-1]
    g['dt_valid'] = valid & previous_valid & same_period
    dt = np.zeros(valid.shape, np.float32)
    dt[:, 1:] = g['seconds'][:, 1:] - g['seconds'][:, :-1]
    g['dt'] = np.where(g['dt_valid'], np.maximum(dt, 0.0), 0.0).astype(np.float32)
    g['period_break'] = valid & previous_valid & ~same_period
    # Online phases: single-actor units carry their team; merged duels never confirm or close.
    acting = np.where(g['merged'], 0, g['part_team'][..., 0])
    g['phase'] = data.phases(dict(team=acting, event=g['event'], subevent=g['raw_sub'], period=g['period'], valid=valid))
    g.update(valid=valid, index=np.where(valid, index, -1), anchor=anchor, match_row=match_row, home=home, ref=ref, rows=rows)
    return g


def phase_table(g):
    """Per-phase features as of the anchor: duration, member count, latest location, current flag, owner relation."""
    phase, valid = g['phase'], g['valid']
    B, W = phase.shape
    K = int(phase.max()) if phase.size else 0
    if K == 0:
        return np.zeros((B, 0), bool), np.zeros((B, 0, 7), np.float32)
    member = (phase[:, :, None] == np.arange(1, K + 1)[None, None, :]) & valid[:, :, None]
    exists = member.any(1)
    first = member.argmax(1)
    last = W - 1 - member[:, ::-1, :].argmax(1)
    b = np.arange(B)[:, None]
    duration = np.where(exists, g['seconds'][b, last] - g['seconds'][b, first], 0.0)
    count = member.sum(1)
    located = member & g['loc_valid'][:, :, None]
    has_location = located.any(1)
    latest = W - 1 - located[:, ::-1, :].argmax(1)
    position = np.where(has_location[..., None], g['xy'][b, latest], 0.0)
    current = (np.arange(1, K + 1)[None, :] == phase[:, -1:]) & exists
    owner = g['part_team'][b, first, 0]
    owner_same = (owner == g['ref'][:, None]) & exists
    features = np.stack([np.log1p(np.maximum(duration, 0.0)) / 5.0, count / 80.0, position[..., 0], position[..., 1],
                         has_location.astype(np.float32), current.astype(np.float32), owner_same.astype(np.float32)], -1)
    return exists, (features * exists[..., None]).astype(np.float32)


def roster_slots(data, g):
    """Roster slot of each participant (-1 when unknown or outside the roster)."""
    roster = data.roster['player'][g['match_row']]
    rv = data.roster['valid'][g['match_row']]
    p = g['part_player']
    eq = (p[..., None] == roster[:, None, None, :]) & (p[..., None] > 0) & rv[:, None, None, :] & g['valid'][:, :, None, None]
    return np.where(eq.any(-1), eq.argmax(-1), -1)


def gap_type(g):
    """Exclusive relation of each consecutive valid pair (index on the later unit)."""
    kind = np.full(g['valid'].shape, -1, np.int64)
    linked = g['dt_valid'] | g['period_break']
    bounds = np.searchsorted(np.array([2.0, 5.0, 15.0, 60.0]), g['dt'], side='right')
    kind[g['dt_valid']] = bounds[g['dt_valid']]
    kind[g['period_break']] = 5
    kind[~linked] = -1
    return kind


def event_numeric(g):
    return np.stack([g['seconds'] / 3600.0, np.log1p(g['dt']) * g['dt_valid'], g['dt_valid'], g['xy'][..., 0], g['xy'][..., 1],
                     g['loc_valid'], g['merged']], -1).astype(np.float32)


def roster_features(data, g):
    r, m = data.roster, g['match_row']
    numeric = np.stack([r['height'][m] / 200.0, r['weight'][m] / 100.0, r['meta'][m], r['starter'][m]], -1).astype(np.float32)
    return dict(vocab=r['vocab'][m], role=r['role'][m], foot=r['foot'][m], numeric=numeric * r['valid'][m][..., None], valid=r['valid'][m],
                team=r['team'][m])


def targets_of(data, rows):
    t = data.targets
    keys = ('event', 'time', 'time_mask', 'xy', 'pos_mask', 'team_same', 'team_mask', 'player_slot', 'player_weight', 'player_mask',
            'target_duel', 'anchor_duel', 'match_id', 'anchor')
    return {k: t[k][rows] for k in keys}


def hgt_batch(data, rows, condition, relations):
    """Node inputs and typed edges of the heterogeneous graphs of one batch."""
    g = gather(data, rows)
    B, W = g['valid'].shape
    families = set(condition['relations'])
    valid = g['valid']
    event_id = np.full((B, W), -1, np.int64)
    event_id[valid] = np.arange(int(valid.sum()))
    event_batch = np.repeat(np.arange(B), valid.sum(1))
    roster = roster_features(data, g)
    rv = roster['valid']
    player_id = np.full(rv.shape, -1, np.int64)
    player_id[rv] = np.arange(int(rv.sum()))
    player_batch = np.repeat(np.arange(B), rv.sum(1))
    x = dict(event=dict(kind=g['event'][valid] - 1, sub=g['sub_code'][valid], period=g['period'][valid], numeric=event_numeric(g)[valid]),
             player=dict(vocab=roster['vocab'][rv], role=roster['role'][rv], foot=roster['foot'][rv], numeric=roster['numeric'][rv]))
    edges = {}

    def add(edge_type, src, dst, reverse=None):
        edges[edge_type] = np.stack([src, dst]).astype(np.int64)
        if reverse is not None:
            edges[reverse] = np.stack([dst, src]).astype(np.int64)

    if 'seq_next' in families or 'seq_gap' in families:
        pair = valid[:, :-1] & valid[:, 1:]
        src, dst = event_id[:, :-1][pair], event_id[:, 1:][pair]
        if 'seq_next' in families:
            add(('event', 'next', 'event'), src, dst)
        if 'seq_gap' in families:
            kinds = gap_type(g)[:, 1:][pair]
            for k, name in enumerate(GAP_TYPES):
                add(('event', name, 'event'), src[kinds == k], dst[kinds == k])
    use_team_nodes = bool(families & {'team', 'member'})
    if use_team_nodes:
        side = np.stack([np.zeros(B, np.int64), np.ones(B, np.int64)], 1)
        teams = np.stack([g['home'], data.matches['away_team'][g['match_row']]], 1)
        vocab = np.stack([data.matches['home_vocab'][g['match_row']], data.matches['away_vocab'][g['match_row']]], 1)
        kinds = np.stack([data.matches['home_type'][g['match_row']], data.matches['away_type'][g['match_row']]], 1)
        x['team'] = dict(vocab=vocab.reshape(-1), side=side.reshape(-1) + 1, kind=kinds.reshape(-1),
                         is_ref=(teams == g['ref'][:, None]).reshape(-1).astype(np.float32)[:, None])
    if 'player' in families:
        slots = roster_slots(data, g)
        if condition['player_edges'] == 'shuffled_within_team':
            perm = data.shuffle[g['match_row']]
            slots = np.where(slots >= 0, np.take_along_axis(perm[:, None, :].repeat(W, 1), np.maximum(slots, 0), axis=2), -1)
        b, w, k = np.nonzero(slots >= 0)
        add(('player', 'performs', 'event'), player_id[b, slots[b, w, k]], event_id[b, w], ('event', 'performed_by', 'player'))
    if 'team' in families:
        b, w, k = np.nonzero((g['part_team'] > 0) & valid[..., None])
        node = 2 * b + (g['part_team'][b, w, k] != g['home'][b])
        add(('team', 'performs', 'event'), node, event_id[b, w], ('event', 'performed_by_team', 'team'))
    if 'member' in families:
        b, s = np.nonzero(rv)
        node = 2 * b + (roster['team'][b, s] != g['home'][b])
        add(('player', 'member_of', 'team'), player_id[b, s], node, ('team', 'has_member', 'player'))
    if 'type' in families:
        x['event_type'] = dict(kind=np.tile(np.arange(10), B))
        b, w = np.nonzero(valid)
        add(('event', 'has_type', 'event_type'), event_id[b, w], 10 * b + g['event'][b, w] - 1, ('event_type', 'describes', 'event'))
    if 'zone' in families:
        x['zone'] = dict(kind=np.tile(np.arange(20), B), center=np.tile(ZONE_CENTERS, (B, 1)))
        b, w = np.nonzero(g['loc_valid'])
        zone = ((g['xy'][b, w, None, :] - ZONE_CENTERS[None]) ** 2).sum(-1).argmin(-1)
        add(('event', 'starts_in', 'zone'), event_id[b, w], 20 * b + zone, ('zone', 'start_of', 'event'))
    if 'possession' in families:
        exists, features = phase_table(g)
        poss_id = np.full(exists.shape, -1, np.int64)
        poss_id[exists] = np.arange(int(exists.sum()))
        x['possession'] = dict(numeric=features[exists])
        b, w = np.nonzero(g['phase'] > 0)
        add(('event', 'belongs_to', 'possession'), event_id[b, w], poss_id[b, g['phase'][b, w] - 1], ('possession', 'contains', 'event'))
    return dict(family='hgt', x=x, edges=edges, event_batch=event_batch, anchor_event=event_id[:, -1], player_id=player_id,
                player_batch=player_batch, batch_size=B), targets_of(data, rows)


def seq_batch(data, rows):
    """Token inputs for the sequence counterpart: same unit fields, participants and phase relations."""
    g = gather(data, rows)
    B, W = g['valid'].shape
    valid = g['valid']
    roster = roster_features(data, g)
    slots = roster_slots(data, g)
    b = np.arange(B)[:, None, None]
    has = slots >= 0
    take = np.maximum(slots, 0)
    participant = dict(vocab=np.where(has, roster['vocab'][b, take], 0), role=np.where(has, roster['role'][b, take], 0),
                       foot=np.where(has, roster['foot'][b, take], 0), numeric=np.where(has[..., None], roster['numeric'][b, take], 0.0),
                       has=has)
    team_relation = np.where(g['part_team'] > 0, np.where(g['part_team'] == g['ref'][:, None, None], 1, 2), 0) * valid[..., None]
    phase = g['phase']
    anchor_phase = phase[:, -1:]
    owner = np.zeros_like(phase)
    for k in range(1, int(phase.max()) + 1 if phase.size else 1):
        member = (phase == k) & valid
        first = member.argmax(1)
        team = g['part_team'][np.arange(B), first, 0]
        owner = np.where(member, team[:, None], owner)
    phase_numeric = np.stack([(phase > 0) & (phase == anchor_phase), (phase > 0) & (owner == g['ref'][:, None]), phase > 0], -1).astype(np.float32)
    order = np.where(valid, np.cumsum(valid, 1), 0)
    distance = np.where(valid, g['index'] - g['index'][:, -1:], 0)
    x = dict(kind=np.where(valid, g['event'] - 1, 0), sub=g['sub_code'], period=g['period'], numeric=event_numeric(g), valid=valid,
             participant=participant, team_relation=team_relation, phase_numeric=phase_numeric * valid[..., None], order=order,
             distance=distance, seconds=g['seconds'], period_raw=g['period'], candidates=roster)
    return dict(family='seq', x=x, batch_size=B), targets_of(data, rows)


def to_torch(value, device=None):
    if isinstance(value, dict):
        return {k: to_torch(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_torch(v, device) for v in value)
    if isinstance(value, torch.Tensor):
        return value.to(device) if device is not None else value
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(value))
        return tensor.to(device) if device is not None else tensor
    return value
