"""Merged-duel unit tables, five-task targets, match rosters and train-only vocabularies.

Raw fields are read through the parent whitelist (`data.clean_event`): type, subtype,
period, seconds, start position, player and team. Rosters use only player ids of the
lineup and bench; the statistics stored next to them in the match file are never read.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

SLOTS = 40
ROLES = {'Goalkeeper': 1, 'Defender': 2, 'Midfielder': 3, 'Forward': 4}
FEET = {'right': 1, 'left': 2, 'both': 3}
TEAM_TYPES = {'club': 1, 'national': 2}
SINGLE_SUB_OFFSET = 2          # subtype code = raw subtype + 2, so -1 (none) -> 1; 0 is padding
FIRST_PAIR_CODE = 103


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def parent(config):
    if config['parent_src'] not in sys.path:
        sys.path.insert(0, config['parent_src'])
    import data as parent_data
    return parent_data


def is_pair(a, b):
    """The parent duel-pair rule (graph_prediction_research/src/data.py rebuild)."""
    return (a['event'] == b['event'] == 1 and a['team'] > 0 and b['team'] > 0 and a['team'] != b['team']
            and a['period'] == b['period'] and a['xy_valid'] and b['xy_valid']
            and max(abs(a['xy'][0] + b['xy'][0] - 1), abs(a['xy'][1] + b['xy'][1] - 1)) < 0.011)


def home_frame(xy, team, home):
    xy = np.asarray(xy, np.float64)
    return xy if team == home else 1.0 - xy


def match_units(cleaned, home):
    """Greedy left-to-right pairing; each pair becomes one order-free Duel unit."""
    units = []
    i = 0
    last_single_team = 0
    while i < len(cleaned):
        if i + 1 < len(cleaned) and is_pair(cleaned[i], cleaned[i + 1]):
            a, b = cleaned[i], cleaned[i + 1]
            xy = (home_frame(a['xy'], a['team'], home) + home_frame(b['xy'], b['team'], home)) / 2
            parts = sorted([(a['player'], a['team']), (b['player'], b['team'])], key=lambda p: p[1] != home)
            units.append(dict(event=1, raw_sub=-1, pair=(min(a['subevent'], b['subevent']), max(a['subevent'], b['subevent'])),
                              period=a['period'], seconds=a['seconds'], xy=xy, valid=True, merged=True, parts=parts,
                              ref=last_single_team if last_single_team else home, records=(i, i + 1)))
            i += 2
            continue
        r = cleaned[i]
        if r['team'] <= 0:
            raise ValueError('Unknown team on a record')
        xy = home_frame(r['xy'], r['team'], home) if r['xy_valid'] else np.zeros(2)
        units.append(dict(event=r['event'], raw_sub=r['subevent'], pair=None, period=r['period'], seconds=r['seconds'],
                          xy=xy, valid=bool(r['xy_valid']), merged=False, parts=[(r['player'], r['team'])],
                          ref=r['team'], records=(i, -1)))
        last_single_team = r['team']
        i += 1
    return units


def roster_of(match):
    """Home lineup, home bench, away lineup, away bench; player ids and starter flags only."""
    sides = {td['side']: td for td in match['teamsData'].values()}
    rows = []
    for side in ('home', 'away'):
        td = sides[side]
        for group, starter in (('lineup', True), ('bench', False)):
            for entry in td['formation'][group]:
                rows.append((int(entry['playerId']), int(td['teamId']), starter))
    if len(rows) > SLOTS or len({r[0] for r in rows}) != len(rows):
        raise ValueError(('Roster size or duplicate player', match['wyId']))
    return int(sides['home']['teamId']), int(sides['away']['teamId']), rows


def rebuild(config):
    pd = parent(config)
    out = Path(config['paths']['data'])
    for split in ('train', 'validation'):
        if (out / f'{split}_units.npz').exists():
            raise FileExistsError(out / f'{split}_units.npz')
    manifests = {s: read_json(p) for s, p in config['paths']['manifests'].items()}
    ids = {s: [int(r['match_id']) for r in rows] for s, rows in manifests.items()}
    sets = {s: set(v) for s, v in ids.items()}
    if any(len(sets[s]) != len(ids[s]) for s in ids) or any(sets[a] & sets[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise ValueError('Duplicate or overlapping match split')
    matches = {int(m['wyId']): m for m in read_json(config['paths']['raw_matches'])}
    players = {int(p['wyId']): p for p in read_json(config['paths']['raw_players'])}
    teams = {int(t['wyId']): t for t in read_json(config['paths']['raw_teams'])}
    raw = read_json(config['paths']['raw_events'])
    grouped = {mid: [] for s in ('train', 'validation') for mid in ids[s]}
    for order, e in enumerate(raw):
        if e['matchId'] in grouped:
            grouped[e['matchId']].append((order, e))

    built = {}
    for split in ('train', 'validation'):
        split_units, split_rosters = [], []
        for mid in ids[split]:
            events = sorted(grouped[mid], key=lambda t: (pd.PERIODS[t[1]['matchPeriod']], float(t[1]['eventSec']), t[0]))
            home, away, roster = roster_of(matches[mid])
            cleaned = [pd.clean_event(e) for _, e in events]
            units = match_units(cleaned, home)
            for u in units:
                u['record_orders'] = tuple(events[k][0] if k >= 0 else -1 for k in u['records'])
            split_units.append(units)
            split_rosters.append((mid, home, away, roster, len(events)))
        built[split] = (split_units, split_rosters)

    # Train-only vocabularies.
    train_units, train_rosters = built['train']
    player_vocab = sorted({p for _, _, _, roster, _ in train_rosters for p, _, _ in roster})
    team_vocab = sorted({t for _, home, away, _, _ in train_rosters for t in (home, away)})
    pair_kinds = sorted({u['pair'] for units in train_units for u in units if u['merged']})
    player_index = {p: i + 1 for i, p in enumerate(player_vocab)}
    team_index = {t: i + 1 for i, t in enumerate(team_vocab)}
    pair_code = {k: FIRST_PAIR_CODE + i for i, k in enumerate(pair_kinds)}
    unknown_pair_code = FIRST_PAIR_CODE + len(pair_kinds)
    vocab = dict(players=len(player_vocab) + 1, teams=len(team_vocab) + 1, subtype_codes=unknown_pair_code + 1,
                 pair_kinds=[list(k) for k in pair_kinds], unknown_pair_code=unknown_pair_code,
                 roles=ROLES, feet=FEET, team_types=TEAM_TYPES, source='train matches only')
    write_json(out / 'vocabulary.json', vocab)

    report = {}
    for split in ('train', 'validation'):
        split_units, split_rosters = built[split]
        U = dict(match_row=[], event=[], raw_sub=[], sub_code=[], period=[], seconds=[], xy_home=[], loc_valid=[], merged=[],
                 part_player=[], part_team=[], ref_team=[], record_order=[])
        T = dict(anchor=[], match_start=[], match_row=[], match_id=[], event=[], time=[], time_mask=[], xy=[], pos_mask=[],
                 team_same=[], team_mask=[], player_slot=[], player_weight=[], player_mask=[], target_duel=[], anchor_duel=[])
        R = {k: np.zeros((len(split_rosters), SLOTS), dtype) for k, dtype in
             [('player', np.int64), ('team', np.int64), ('starter', bool), ('valid', bool), ('vocab', np.int64), ('role', np.int64),
              ('foot', np.int64), ('height', np.float32), ('weight', np.float32), ('meta', bool)]}
        M = dict(match_id=[], home_team=[], away_team=[], home_vocab=[], away_vocab=[], home_type=[], away_type=[])
        cursor = 0
        pairs = records = masked_duel_players = 0
        for row, (units, (mid, home, away, roster, n_records)) in enumerate(zip(split_units, split_rosters)):
            slot = {p: k for k, (p, _, _) in enumerate(roster)}
            for k, (p, t, starter) in enumerate(roster):
                meta = players.get(p)
                R['player'][row, k] = p; R['team'][row, k] = t; R['starter'][row, k] = starter; R['valid'][row, k] = True
                R['vocab'][row, k] = player_index.get(p, 0)
                if meta is not None:
                    R['role'][row, k] = ROLES.get((meta.get('role') or {}).get('name', ''), 0)
                    R['foot'][row, k] = FEET.get(str(meta.get('foot') or '').lower(), 0)
                    R['height'][row, k] = float(meta.get('height') or 0); R['weight'][row, k] = float(meta.get('weight') or 0)
                    R['meta'][row, k] = True
            for side, team in (('home', home), ('away', away)):
                M[f'{side}_team'].append(team); M[f'{side}_vocab'].append(team_index.get(team, 0))
                M[f'{side}_type'].append(TEAM_TYPES.get(str(teams.get(team, {}).get('type', '')), 0))
            M['match_id'].append(mid)
            records += n_records
            for u in units:
                U['match_row'].append(row); U['event'].append(u['event']); U['raw_sub'].append(u['raw_sub'])
                U['sub_code'].append(pair_code.get(u['pair'], unknown_pair_code) if u['merged'] else u['raw_sub'] + SINGLE_SUB_OFFSET)
                U['period'].append(u['period']); U['seconds'].append(u['seconds']); U['xy_home'].append(u['xy']); U['loc_valid'].append(u['valid'])
                U['merged'].append(u['merged']); U['ref_team'].append(u['ref']); U['record_order'].append(u['record_orders'])
                parts = u['parts'] + [(0, 0)] * (2 - len(u['parts']))
                U['part_player'].append([p for p, _ in parts]); U['part_team'].append([t for _, t in parts])
                pairs += u['merged']
            for j in range(len(units) - 1):
                a, v = units[j], units[j + 1]
                ref = a['ref']
                T['anchor'].append(cursor + j); T['match_start'].append(cursor); T['match_row'].append(row); T['match_id'].append(mid)
                T['event'].append(v['event'] - 1)
                T['time'].append(float(np.clip(v['seconds'] - a['seconds'], 0, 60))); T['time_mask'].append(v['period'] == a['period'])
                T['xy'].append(v['xy'] if ref == home else 1.0 - v['xy']); T['pos_mask'].append(v['valid'])
                if v['merged']:
                    T['team_same'].append(0.5)
                    known = all(p > 0 and p in slot for p, _ in v['parts'])
                    T['player_slot'].append([slot[p] for p, _ in v['parts']] if known else [-1, -1])
                    T['player_weight'].append([0.5, 0.5] if known else [0.0, 0.0])
                    T['player_mask'].append(known); masked_duel_players += not known
                else:
                    p, t = v['parts'][0]
                    T['team_same'].append(float(t == ref))
                    known = p > 0 and p in slot
                    T['player_slot'].append([slot[p], -1] if known else [-1, -1])
                    T['player_weight'].append([1.0, 0.0] if known else [0.0, 0.0])
                    T['player_mask'].append(known)
                T['team_mask'].append(True)
                T['target_duel'].append(v['merged']); T['anchor_duel'].append(a['merged'])
            cursor += len(units)
        units_np = {k: np.asarray(v) for k, v in U.items()}
        units_np['xy_home'] = units_np['xy_home'].astype(np.float32); units_np['seconds'] = units_np['seconds'].astype(np.float32)
        targets_np = {k: np.asarray(v) for k, v in T.items()}
        targets_np['xy'] = targets_np['xy'].astype(np.float32); targets_np['time'] = targets_np['time'].astype(np.float32)
        targets_np['team_same'] = targets_np['team_same'].astype(np.float32); targets_np['player_weight'] = targets_np['player_weight'].astype(np.float32)
        matches_np = {k: np.asarray(v) for k, v in M.items()}
        expected = config['dataset']['expected'][split]
        counts = dict(matches=len(split_rosters), records=records, pairs=int(pairs), units=len(units_np['event']), targets=len(targets_np['anchor']))
        for key, value in counts.items():
            if value != expected[key]:
                raise ValueError(('Unexpected count', split, key, value, expected[key]))
        np.savez(out / f'{split}_units.npz', **units_np)
        np.savez(out / f'{split}_targets.npz', **targets_np)
        np.savez(out / f'{split}_rosters.npz', **{f'roster_{k}': v for k, v in R.items()}, **matches_np)
        report[split] = dict(counts, duel_targets=int(targets_np['target_duel'].sum()), duel_anchors=int(targets_np['anchor_duel'].sum()),
                             player_masked=int((~targets_np['player_mask']).sum()), duel_targets_player_masked=int(masked_duel_players),
                             position_valid=int(targets_np['pos_mask'].sum()), time_valid=int(targets_np['time_mask'].sum()),
                             unseen_roster_players=int(((R['vocab'] == 0) & R['valid']).sum()), roster_slots_mean=float(R['valid'].sum(1).mean()),
                             merged_duels_without_preceding_single=int(sum(1 for units in split_units for k, u in enumerate(units)
                                                                            if u['merged'] and not any(not w['merged'] for w in units[:k]))))
        print(split, report[split], flush=True)
    write_json(Path(config['paths']['reports']) / 'data_counts.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    rebuild(read_json(parser.parse_args().config))
