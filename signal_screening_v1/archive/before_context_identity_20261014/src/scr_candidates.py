"""Per-roster-slot timing used to form on-pitch candidate sets at prediction time.

For every match and roster slot: the first unit in which the player acts, the substitution
minutes (in / out), and the unit of a red-card foul. Candidate sets are computed per anchor from
these values and the anchor clock (`on_pitch`), so only information available at the anchor enters.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

NEVER = np.iinfo(np.int64).max


def setup(config):
    for p in (config['five_task_src'], config['parent_src']):
        if p not in sys.path:
            sys.path.insert(0, p)


def build(config):
    setup(config)
    from units import read_json, write_json
    from batches import SplitData
    out = Path(config['paths']['candidates'])
    matches = {int(m['wyId']): m for m in read_json(config['paths']['raw_matches'])}
    raw = read_json(config['paths']['raw_events'])
    report = {}
    for split in ('train', 'validation'):
        dest = out / f'{split}_candidates.npz'
        if dest.exists():
            raise FileExistsError(dest)
        d = SplitData(config, split)
        u, roster = d.units, d.roster
        M, S = roster['player'].shape
        first_seen = np.full((M, S), NEVER, np.int64)
        sub_in = np.full((M, S), -1, np.int64)
        sub_out = np.full((M, S), -1, np.int64)
        red_unit = np.full((M, S), NEVER, np.int64)
        for row, mid in enumerate(d.matches['match_id']):
            slot = {int(p): s for s, p in enumerate(roster['player'][row]) if roster['valid'][row, s]}
            for td in matches[int(mid)]['teamsData'].values():
                subs = td['formation'].get('substitutions')
                if isinstance(subs, list):
                    for x in subs:
                        if int(x['playerIn']) in slot:
                            sub_in[row, slot[int(x['playerIn'])]] = int(x['minute'])
                        if int(x['playerOut']) in slot:
                            sub_out[row, slot[int(x['playerOut'])]] = int(x['minute'])
                elif subs != 'null':
                    raise ValueError(('Unexpected substitutions field', mid, subs))
            units = np.flatnonzero(u['match_row'] == row)
            for k in units:
                for p in u['part_player'][k]:
                    if p > 0 and int(p) in slot:
                        s = slot[int(p)]
                        first_seen[row, s] = min(first_seen[row, s], k)
                for rec in u['record_order'][k]:
                    if rec < 0:
                        continue
                    e = raw[int(rec)]
                    if int(e['eventId']) == 2 and any(int(x['id']) in (1701, 1703) for x in e['tags']) and int(e['playerId']) in slot:
                        s = slot[int(e['playerId'])]
                        red_unit[row, s] = min(red_unit[row, s], k)
        np.savez(dest, first_seen=first_seen, sub_in=sub_in, sub_out=sub_out, red_unit=red_unit)
        report[split] = dict(substitutions_in=int((sub_in >= 0).sum()), substitutions_out=int((sub_out >= 0).sum()),
                             red_cards=int((red_unit < NEVER).sum()))
    write_json(Path(config['paths']['reports']) / 'candidate_timing.json', report)
    print(report)


def anchor_clock(units, anchor):
    period = units['period'][anchor]
    if (period > 2).any():
        raise ValueError('Extra-time periods are not handled')
    return units['seconds'][anchor] + np.where(period == 2, 2700.0, 0.0)


def on_pitch(timing, roster, match_row, anchor, clock):
    """(B, S) candidate mask at the anchor; uses only events up to the anchor and finished substitution minutes."""
    first_seen, sub_in, sub_out, red = (timing[k][match_row] for k in ('first_seen', 'sub_in', 'sub_out', 'red_unit'))
    valid, starter = roster['valid'][match_row], roster['starter'][match_row]
    c = clock[:, None]
    entered = (sub_in >= 0) & ((c >= (sub_in + 1) * 60) | (first_seen <= anchor[:, None]))
    left = (sub_out >= 0) & (c >= (sub_out + 1) * 60)
    sent_off = red < anchor[:, None]
    return valid & (starter | entered) & ~left & ~sent_off


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    import json
    build(json.load(open(parser.parse_args().config)))
