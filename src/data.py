"""Explicit raw-event whitelist, independent targets and input-only window selection."""
import argparse
import json
from pathlib import Path
import numpy as np

PERIODS = {'1H': 1, '2H': 2, 'E1': 3, 'E2': 4, 'P': 5}
PLACEHOLDERS = {34, 40, 51, 90, 91}
FIELDS = ('event', 'subevent', 'period', 'seconds', 'player', 'team', 'xy', 'xy_valid')


def read_config(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def clean_event(e):
    """Read only named raw fields; end coordinates and tags are never accessed."""
    event = int(e['eventId'])
    sub = -1 if e['subEventId'] == '' else int(e['subEventId'])
    if not 1 <= event <= 10 or not -1 <= sub <= 100:
        raise ValueError(('Unexpected event taxonomy', event, sub))
    seconds = float(e['eventSec'])
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError('Invalid period seconds')
    start = e['positions'][0] if e['positions'] else None
    valid = start is not None and sub not in PLACEHOLDERS
    xy = [float(start['x']) / 100, float(start['y']) / 100] if valid else [0., 0.]
    if valid and (not np.isfinite(xy).all() or min(xy) < 0 or max(xy) > 1):
        raise ValueError('Invalid start coordinates')
    return dict(event=event, subevent=sub, period=PERIODS[e['matchPeriod']], seconds=seconds,
                player=max(0, int(e['playerId'])), team=max(0, int(e['teamId'])), xy=xy, xy_valid=valid)


def anchor_frame(xy, team, anchor_team, valid):
    known = valid & (team > 0) & (anchor_team > 0)
    transformed = np.where((team == anchor_team)[..., None], xy, 1 - xy)
    return np.where(known[..., None], transformed, 0).astype(np.float32), known


def phases(w):
    """Online membership. A closing action belongs to the phase it closes."""
    n, length = w['team'].shape
    owner = np.zeros(n, np.int64)
    phase = np.zeros(n, np.int64)
    count = np.zeros(n, np.int64)
    out = np.zeros((n, length), np.int64)
    previous = np.zeros(n, np.int64)
    for j in range(length):
        valid, period = w['valid'][:, j], w['period'][:, j]
        boundary = valid & (period != previous)
        owner[boundary] = 0; phase[boundary] = 0
        e, sub, team = w['event'][:, j], w['subevent'][:, j], w['team'][:, j]
        confirm = valid & ((e == 3) | (e == 8) | (e == 10) | ((e == 7) & ((sub == 70) | (sub == 71)))) & (team > 0)
        new = confirm & ((owner == 0) | (owner != team))
        count[new] += 1; phase[new] = count[new]; owner[confirm] = team[confirm]
        out[:, j] = np.where(valid, phase, 0)
        close = valid & ((e == 2) | (e == 5) | (e == 6))
        owner[close] = 0; phase[close] = 0
        previous = np.where(valid, period, previous)
    return out


def local_identity(values, valid):
    """Contiguous numbering by first appearance, with unknown=0."""
    n, length = values.shape
    result = np.zeros((n, length), np.int64)
    count = np.zeros(n, np.int64)
    rows = np.arange(n)
    for j in range(length):
        known = valid[:, j] & (values[:, j] > 0)
        equal = (values[:, :j + 1] == values[:, j:j + 1]) & valid[:, :j + 1]
        first = equal.argmax(1)
        new = known & (first == j)
        count[new] += 1
        result[:, j] = np.where(known, np.where(new, count, result[rows, first]), 0)
    return result


def windows(inputs, targets, rows, length=80, offsets=None):
    anchor = targets['anchor'][rows]
    if offsets is None:
        offsets = np.broadcast_to(np.arange(length - 1, -1, -1), (len(rows), length))
    indices = anchor[:, None] - offsets
    valid = (offsets >= 0) & (indices >= targets['match_start'][rows, None])
    safe = np.where(valid, indices, anchor[:, None])
    w = {key: value[safe].copy() for key, value in inputs.items() if key in FIELDS}
    w['valid'], w['indices'] = valid, np.where(valid, indices, -1)
    w['xy'], w['xy_valid'] = anchor_frame(w['xy'], w['team'], inputs['team'][anchor, None], w['xy_valid'] & valid)
    for key in ['event', 'subevent', 'period', 'seconds', 'player', 'team']:
        w[key][~valid] = 0
    w['phase'] = phases(w)
    return w


def features(w):
    """Derived fields are rebuilt exclusively from the supplied selected records."""
    out = {k: w[k] for k in ('event', 'subevent', 'period', 'seconds', 'xy', 'xy_valid', 'valid')}
    out['player'] = local_identity(w['player'], w['valid'])
    out['team'] = local_identity(w['team'], w['valid'])
    out['phase'] = phases(w)
    # Original event distances are observable order metadata, not hidden event contents.
    out['order'] = np.where(w['valid'], w['indices'] - w['indices'][:, -1:], 0)
    return out


def paired_select(w, rule, priorities):
    """Vectorized nearest-first matching, no relaxation of the two distance bins."""
    n, length = w['valid'].shape
    assert length == 80 and priorities.shape == (n, 64)
    p, t, z = w['player'], w['team'], w['phase']
    pk = (p > 0) & (p[:, -1:] > 0)
    tk = (t > 0) & (t[:, -1:] > 0)
    zk = (z > 0) & (z[:, -1:] > 0)
    same_p, same_t, same_z = p == p[:, -1:], t == t[:, -1:], z == z[:, -1:]
    base = w['valid'] & (w['period'] == w['period'][:, -1:])
    if rule == 'player':
        rel = base & pk & same_p & zk
        ctrl = base & tk & same_t & pk & ~same_p & zk
    elif rule == 'current_phase':
        rel = base & pk & ~same_p & tk & same_t & zk & same_z
        ctrl = base & pk & ~same_p & tk & same_t & zk & ~same_z
    elif rule == 'other_phase_team':
        rel = base & pk & ~same_p & tk & same_t & zk & ~same_z
        ctrl = base & tk & ~same_t & zk & ~same_z
    else:
        raise ValueError(rule)
    rel, ctrl = rel[:, :64], ctrl[:, :64].copy()
    dt = w['seconds'][:, -1:] - w['seconds'][:, :64]
    dn = np.broadcast_to(np.arange(79, 15, -1), dt.shape)
    bins = ((dn - 16) // 16) * 5 + np.searchsorted([5, 15, 30, 60], dt, side='right')
    q = np.zeros(n, np.int64)
    chosen_r, chosen_c = np.full((n, 8), -1, np.int64), np.full((n, 8), -1, np.int64)
    delta_n_error, delta_t_error = np.zeros((n, 8)), np.zeros((n, 8))
    for j in range(63, -1, -1):
        eligible = ctrl & (bins == bins[:, j:j + 1])
        if rule == 'player':
            eligible &= same_z[:, :64] == same_z[:, j:j + 1]
        eligible &= (rel[:, j] & (q < 8))[:, None]
        use = eligible.any(1)
        rows = np.flatnonzero(use)
        if not len(rows):
            continue
        col = np.where(eligible[rows], priorities[rows], np.inf).argmin(1)
        slot = q[rows]
        chosen_r[rows, slot] = 79 - j
        chosen_c[rows, slot] = 79 - col
        delta_n_error[rows, slot] = np.abs(col - j)
        delta_t_error[rows, slot] = np.abs(dt[rows, col] - dt[rows, j])
        ctrl[rows, col] = False
        q[rows] += 1
    def assemble(chosen):
        recent = np.broadcast_to(np.arange(15, -1, -1), (n, 16)).copy()
        recent[recent >= (16 - q[:, None])] = -1
        combined = np.sort(np.concatenate([recent, chosen], axis=1), axis=1)[:, ::-1][:, :16].copy()
        # Right align short prefixes; padding offset=-1 is always excluded.
        available = w['valid'].sum(1)
        combined[combined >= available[:, None]] = -1
        combined = np.sort(combined, axis=1)[:, ::-1]
        # Padding must be on the left so the anchor remains the last slot.
        order = np.argsort(np.where(combined < 0, 1000, combined), axis=1)[:, ::-1]
        return np.take_along_axis(combined, order, axis=1).astype(np.int16)
    return assemble(chosen_r), assemble(chosen_c), q, delta_n_error, delta_t_error


def rebuild(config):
    out = Path(config['paths']['data'])
    for split in ['train', 'validation']:
        if (out / f'{split}_inputs.npz').exists():
            raise FileExistsError(out / f'{split}_inputs.npz')
    manifest = {s: read_config(p) for s, p in config['paths']['manifests'].items()}
    ids = {s: {int(r['match_id']) for r in rows} for s, rows in manifest.items()}
    if any(len(ids[s]) != len(manifest[s]) for s in ids) or any(ids[a] & ids[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
        raise ValueError('Duplicate or overlapping match split')
    for s in ids:
        if len(ids[s]) != config['dataset']['expected'][s]['matches']:
            raise ValueError(('Unexpected match count',s))
    raw = read_config(config['paths']['raw_events'])
    grouped = {mid: [] for s in ['train','validation'] for mid in ids[s]}
    for original_order,e in enumerate(raw):
        if e['matchId'] in grouped:
            grouped[e['matchId']].append((original_order,e))
    reports = {}
    for split in ['train','validation']:
        rows_out = []; index = []; all_targets = []; cursor = 0
        for m in manifest[split]:
            mid = int(m['match_id'])
            events = sorted(grouped[mid], key=lambda item: (PERIODS[item[1]['matchPeriod']],float(item[1]['eventSec']),item[0]))
            if len(events) != int(m['events']):
                raise ValueError(('Manifest event count differs',mid))
            cleaned = [clean_event(e) for _,e in events]
            arr = {k: np.array([v[k] for v in cleaned]) for k in FIELDS}
            opening, closing = np.zeros(len(events),bool),np.zeros(len(events),bool)
            i = 0
            while i < len(events)-1:
                j=i+1
                paired = arr['event'][i] == arr['event'][j] == 1 and arr['team'][i] > 0 and arr['team'][j] > 0 and arr['team'][i] != arr['team'][j] and arr['period'][i] == arr['period'][j] and arr['xy_valid'][i] and arr['xy_valid'][j] and np.max(np.abs(arr['xy'][i]+arr['xy'][j]-1)) < 0.011
                if paired:
                    opening[i]=closing[j]=True; i+=2
                else:
                    i+=1
            target_xy, target_mask = anchor_frame(arr['xy'][1:],arr['team'][1:],arr['team'][:-1],arr['xy_valid'][1:])
            for j in range(len(events)-1):
                all_targets.append(dict(anchor=cursor+j,match_start=cursor,match_id=mid,local_anchor=j,event=arr['event'][j+1]-1,
                    time=np.clip(arr['seconds'][j+1]-arr['seconds'][j],0,60),time_mask=arr['period'][j]==arr['period'][j+1],xy=target_xy[j],position_mask=target_mask[j],
                    team=int(arr['team'][j]==arr['team'][j+1]),team_mask=arr['team'][j]>0 and arr['team'][j+1]>0,
                    duel_first=opening[j+1],duel_second=closing[j+1]))
            rows_out.extend(cleaned)
            index.extend(dict(match_id=mid,event_id=int(e['id']),original_order=o) for o,e in events)
            cursor+=len(events)
        inputs={k:np.array([e[k] for e in rows_out],dtype=np.float32 if k in ['seconds','xy'] else bool if k=='xy_valid' else np.int64) for k in FIELDS}
        targets={k:np.array([t[k] for t in all_targets]) for k in all_targets[0]}
        if len(all_targets)!=config['dataset']['expected'][split]['targets']:
            raise ValueError(('Unexpected target count',split,len(all_targets)))
        np.savez(out/f'{split}_inputs.npz',**inputs)
        np.savez(out/f'{split}_targets.npz',**targets)
        np.savez(out/f'{split}_index.npz',**{k:np.array([r[k] for r in index]) for k in index[0]})
        reports[split]=dict(matches=len(ids[split]),events=len(rows_out),targets=len(all_targets),time_valid=int(targets['time_mask'].sum()),position_valid=int(targets['position_mask'].sum()),team_valid=int(targets['team_mask'].sum()),unknown_player=int((inputs['player']==0).sum()),unknown_team=int((inputs['team']==0).sum()),placeholder_starts=int(np.isin(inputs['subevent'],list(PLACEHOLDERS)).sum()),paired_first_targets=int(targets['duel_first'].sum()),paired_second_targets=int(targets['duel_second'].sum()))
        print(split,reports[split],flush=True)
    write_json(Path(config['paths']['reports'])/'data_counts.json',reports)


def build_selections(config):
    out=Path(config['paths']['data']); summary={}
    for split in ['train','validation']:
        inputs=dict(np.load(out/f'{split}_inputs.npz')); targets=dict(np.load(out/f'{split}_targets.npz'))
        n=len(targets['anchor']); summary[split]={}
        phase_counts=[]
        for seed in config['selection']['seed_map'].values():
            for rule in config['selection']['rules']:
                dest=out/f'{split}_{rule}_{seed}.npz'
                if dest.exists(): raise FileExistsError(dest)
                rng=np.random.default_rng(seed)
                relation=np.empty((n,16),np.int16); control=np.empty_like(relation); qs=np.empty(n,np.int8)
                errors_n=errors_t=0.; total_pairs=0
                for start in range(0,n,4096):
                    stop=min(start+4096,n); rows=np.arange(start,stop)
                    w=windows(inputs,targets,rows)
                    a,b,q,dn,dt=paired_select(w,rule,rng.random((len(rows),64)))
                    relation[start:stop]=a;control[start:stop]=b;qs[start:stop]=q
                    errors_n+=dn.sum();errors_t+=dt.sum();total_pairs+=q.sum()
                    if seed==20261014 and rule=='player':
                        phase_counts.extend(w['phase'].max(1).tolist())
                np.savez(dest,relation=relation,control=control,q=qs)
                summary[split][f'{rule}_{seed}']=dict(coverage=float((qs>0).mean()),q_counts=np.bincount(qs,minlength=9).tolist(),pairs=int(total_pairs),mean_abs_delta_n_error=float(errors_n/max(1,total_pairs)),mean_abs_delta_t_error=float(errors_t/max(1,total_pairs)))
                print(split,rule,seed,summary[split][f'{rule}_{seed}'],flush=True)
        summary[split]['phase_count_per_window']={'mean':float(np.mean(phase_counts)),'max':int(max(phase_counts)),'unknown_only_windows':int(np.count_nonzero(np.array(phase_counts)==0))}
    write_json(Path(config['paths']['reports'])/'selection_coverage.json',summary)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--step',choices=['rebuild','selections'],required=True)
    args=parser.parse_args();config=read_config(args.config)
    if args.step=='rebuild': rebuild(config)
    else: build_selections(config)
