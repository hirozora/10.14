"""Acceptance tests for the signal screen; writes reports/acceptance.json and reports/acceptance.log."""
import copy
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
C = json.load(open(ROOT / 'configs/experiment.json'))
from scr_candidates import setup, anchor_clock, on_pitch, NEVER
setup(C)
from units import read_json
from train import deterministic
from scr_batches import ScreenData, screen_batch, history_positions, membership
from scr_models import Screen, player_loss
from scr_fit import to_torch
from scr_scoring import sample_statistics, match_table, metrics
V = read_json(Path(C['paths']['data']) / 'vocabulary.json')
ARCHIVE = ROOT / 'archive/before_context_identity_20261014'
IDENTITY_FREE = [name for name, cond in C['conditions'].items() if not cond['context_identity']]


def equal(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    return np.array_equal(a, b)


class ScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deterministic(20260715)
        cls.d = ScreenData(C, 'validation')

    def test_01_incoming_substitute_needs_finished_minute_or_observation(self):
        d = self.d
        u, t, r = d.base.units, d.base.targets, d.base.roster
        checked = 0
        for row in range(len(d.base.matches['match_id'])):
            for s in np.flatnonzero(d.timing['sub_in'][row] >= 0):
                m = d.timing['sub_in'][row, s]
                rows = np.flatnonzero(t['match_row'] == row)
                a = t['anchor'][rows]
                clock = anchor_clock(u, a)
                mask = on_pitch(d.timing, r, np.full(len(rows), row), a, clock)[:, s]
                early = (clock < (m + 1) * 60) & (d.timing['first_seen'][row, s] > a)
                self.assertFalse(mask[early].any())
                late = (clock >= (m + 1) * 60) & ~((d.timing['sub_out'][row, s] >= 0) & (clock >= (d.timing['sub_out'][row, s] + 1) * 60)) & (d.timing['red_unit'][row, s] >= a)
                self.assertTrue(mask[late].all())
                checked += 1
        self.assertGreater(checked, 100)

    def test_02_candidate_set_ignores_the_future(self):
        d = self.d
        t, u = d.base.targets, d.base.units
        for row_index in range(0, len(t['anchor']), 97):
            a, mr = t['anchor'][[row_index]], t['match_row'][[row_index]]
            before = on_pitch(d.timing, d.base.roster, mr, a, anchor_clock(u, a))
            timing = {k: v.copy() for k, v in d.timing.items()}
            row = mr[0]
            later = (timing['first_seen'][row] > a[0]) & (timing['first_seen'][row] < NEVER)
            timing['first_seen'][row, later] += 1
            red_later = (timing['red_unit'][row] >= a[0]) & (timing['red_unit'][row] < NEVER)
            timing['red_unit'][row, red_later] += 1
            self.assertTrue(np.array_equal(before, on_pitch(timing, d.base.roster, mr, a, anchor_clock(u, a))), row_index)

    def test_03_sent_off_player_leaves_after_the_foul(self):
        d = self.d
        found = 0
        for row, s in zip(*np.nonzero(d.timing['red_unit'] < NEVER)):
            j = d.timing['red_unit'][row, s]
            rows = np.flatnonzero((d.base.targets['match_row'] == row) & (np.abs(d.base.targets['anchor'] - j) <= 2))
            a = d.base.targets['anchor'][rows]
            mask = on_pitch(d.timing, d.base.roster, np.full(len(rows), row), a, anchor_clock(d.base.units, a))[:, s]
            self.assertFalse(mask[a > j].any())
            found += 1
        self.assertGreater(found, 0)

    def test_04_context_identity_only_where_declared(self):
        d = copy.copy(self.d)
        d.base = copy.copy(self.d.base)
        d.base.units = {k: v.copy() for k, v in self.d.base.units.items()}
        rows = np.arange(3000, 3064)
        players = d.base.units['part_player']
        known = np.unique(players[players > 0])
        mapping = dict(zip(known, np.random.default_rng(0).permutation(known)))
        d.base.units['part_player'] = np.vectorize(lambda p: mapping.get(p, p))(players)
        for name in IDENTITY_FREE:
            x, _ = screen_batch(self.d, rows, C['conditions'][name])
            y, _ = screen_batch(d, rows, C['conditions'][name])
            self.assertNotIn('member', x['context'], name)
            self.assertTrue(equal(x['context'], y['context']), name)
        # With context identity, the context adds exactly the roster slots taking part in each unit.
        x, _ = screen_batch(self.d, rows, C['conditions']['M0_ctx_identity'])
        free, _ = screen_batch(self.d, rows, C['conditions']['M0_static'])
        self.assertTrue(equal({k: v for k, v in x['context'].items() if k != 'member'}, free['context']))
        u, r = self.d.base.units, self.d.base.roster
        mr = self.d.base.targets['match_row'][rows]
        anchor = self.d.base.targets['anchor'][rows]
        W = C['dataset']['window_units']
        for b in range(len(rows)):
            start = self.d.base.targets['match_start'][rows[b]]
            for w in range(W):
                unit = anchor[b] - (W - 1 - w)
                acted = set(int(q) for q in u['part_player'][unit] if q > 0) if unit >= start else set()
                expect = [bool(r['valid'][mr[b], s] and int(r['player'][mr[b], s]) in acted) for s in range(r['player'].shape[1])]
                self.assertEqual(expect, x['context']['member'][b, w].tolist(), (b, w))
        self.assertGreater(int(x['context']['member'].sum()), len(rows) * 40)

    def test_05_condition_firewall(self):
        d = copy.copy(self.d)
        d.base = copy.copy(self.d.base)
        d.base.units = {k: v.copy() for k, v in self.d.base.units.items()}
        rows = np.arange(7000, 7064)
        x0, _ = screen_batch(self.d, rows, C['conditions']['M0_static'])
        x1, _ = screen_batch(self.d, rows, C['conditions']['M1_summary'])
        # Reassign who acted in the past window units of these samples.
        lo, hi = int(d.base.targets['anchor'][rows].min()) - 80, int(d.base.targets['anchor'][rows].max())
        d.base.units['part_player'][max(lo, 0):hi] = d.base.units['part_player'][max(lo, 0):hi][::-1]
        y0, _ = screen_batch(d, rows, C['conditions']['M0_static'])
        y1, _ = screen_batch(d, rows, C['conditions']['M1_summary'])
        xi, _ = screen_batch(self.d, rows, C['conditions']['M0_ctx_identity'])
        yi, _ = screen_batch(d, rows, C['conditions']['M0_ctx_identity'])
        static = ('vocab', 'role', 'foot', 'numeric', 'same_team', 'mask')
        self.assertTrue(all(np.array_equal(x0['candidates'][k], y0['candidates'][k]) for k in static))
        self.assertTrue(all(np.array_equal(xi['candidates'][k], yi['candidates'][k]) for k in static))
        self.assertFalse(np.array_equal(x1['candidates']['summary'], y1['candidates']['summary']))
        self.assertFalse(np.array_equal(xi['context']['member'], yi['context']['member']))
        self.assertNotIn('h_mask', x1['candidates'])
        self.assertNotIn('h_mask', xi['candidates'])

    def test_06_history_positions_match_a_reference(self):
        from batches import gather
        rows = np.arange(11000, 11032)
        g = gather(self.d.base, rows)
        member = membership(g, self.d.base.roster['player'][g['match_row']])
        pos = history_positions(member, self.d.K)
        for b in range(len(rows)):
            for s in range(member.shape[2]):
                expect = list(np.flatnonzero(member[b, :, s])[::-1][:self.d.K])
                got = [p for p in pos[b, s] if p >= 0]
                self.assertEqual(expect, got)

    def test_07_shuffled_control(self):
        rows = np.arange(15000, 15128)
        x, _ = screen_batch(self.d, rows, C['conditions']['M2_pooled'])
        y, _ = screen_batch(self.d, rows, C['conditions']['M2_shuffled'])
        for k in ('summary', 'vocab', 'mask'):
            self.assertTrue(np.array_equal(x['candidates'][k], y['candidates'][k]))
        team = self.d.base.roster['team'][self.d.base.targets['match_row'][rows]]
        changed = 0
        for i in range(len(rows)):
            for tm in np.unique(team[i]):
                s = np.flatnonzero(x['candidates']['mask'][i] & (team[i] == tm))
                a = sorted(map(tuple, x['candidates']['h_kind'][i, s].tolist()))
                b = sorted(map(tuple, y['candidates']['h_kind'][i, s].tolist()))
                self.assertEqual(a, b)
                changed += int(not np.array_equal(x['candidates']['h_kind'][i, s], y['candidates']['h_kind'][i, s]))
        self.assertGreater(changed, 50)

    def test_08_future_invariance_of_inputs(self):
        for row in (2000, 40000, 80000):
            d = copy.copy(self.d)
            d.base = copy.copy(self.d.base)
            d.base.units = {k: v.copy() for k, v in self.d.base.units.items()}
            cut = int(d.base.targets['anchor'][row]) + 1
            for k in ('event', 'sub_code', 'raw_sub', 'part_player', 'part_team', 'xy_home', 'seconds', 'merged', 'loc_valid', 'ref_team'):
                d.base.units[k][cut:] = d.base.units[k][cut:][::-1]
            for name in ('M1_summary', 'M3_attention', 'M0_ctx_identity'):
                a, _ = screen_batch(self.d, np.array([row]), C['conditions'][name])
                b, _ = screen_batch(d, np.array([row]), C['conditions'][name])
                self.assertTrue(equal(a, b), (row, name))

    def test_09_targets_and_candidates_match_a_slow_reference(self):
        """Candidate sets recomputed per sample from the raw match file and the unit table."""
        d = self.d
        t, u, r = d.base.targets, d.base.units, d.base.roster
        matches = {int(m['wyId']): m for m in read_json(C['paths']['raw_matches'])}
        raw = read_json(C['paths']['raw_events'])
        n = len(t['anchor'])
        unreachable_fast = 0
        for lo in range(0, n, 8192):
            rows = np.arange(lo, min(lo + 8192, n))
            x, y = screen_batch(d, rows, C['conditions']['M0_static'])
            w = y['loss_weight'][y['known'] & y['reachable']]
            np.testing.assert_allclose(w.sum(1), 1.0, rtol=1e-6)
            slot = y['slot']
            ok = (slot >= 0) & np.take_along_axis(x['candidates']['mask'], np.maximum(slot, 0), 1)
            np.testing.assert_array_equal(y['reachable'], (ok & (y['eval_weight'] > 0)).any(1))
            unreachable_fast += int((y['known'] & ~y['duel'] & ~y['reachable']).sum())
        unreachable_slow = 0
        for row, mid in enumerate(d.base.matches['match_id']):
            m = matches[int(mid)]
            subs = [x for td in m['teamsData'].values() if isinstance(td['formation'].get('substitutions'), list) for x in td['formation']['substitutions']]
            sub_in = {int(x['playerIn']): int(x['minute']) for x in subs}
            sub_out = {int(x['playerOut']): int(x['minute']) for x in subs}
            starters = {int(e['playerId']) for td in m['teamsData'].values() for e in td['formation']['lineup']}
            units = np.flatnonzero(u['match_row'] == row)
            first, red = {}, {}
            for k in units:
                for p in u['part_player'][k]:
                    first.setdefault(int(p), k)
                for rec in u['record_order'][k]:
                    if rec >= 0:
                        e = raw[int(rec)]
                        if int(e['eventId']) == 2 and any(int(q['id']) in (1701, 1703) for q in e['tags']):
                            red.setdefault(int(e['playerId']), k)
            rows = np.flatnonzero(t['match_row'] == row)
            fast = on_pitch(d.timing, r, t['match_row'][rows], t['anchor'][rows], anchor_clock(u, t['anchor'][rows]))
            for i, rr in enumerate(rows[::7]):
                a = t['anchor'][rr]
                clock = float(anchor_clock(u, np.array([a]))[0])
                for s, p in enumerate(r['player'][row]):
                    if not r['valid'][row, s]:
                        continue
                    p = int(p)
                    on = (p in starters or (p in sub_in and (clock >= (sub_in[p] + 1) * 60 or first.get(p, 10 ** 12) <= a)))
                    on = on and not (p in sub_out and clock >= (sub_out[p] + 1) * 60) and not red.get(p, 10 ** 12) < a
                    self.assertEqual(on, bool(fast[i * 7, s]), (row, rr, p))
            for rr in rows:
                if t['player_mask'][rr] and not t['target_duel'][rr]:
                    a = t['anchor'][rr]
                    clock = float(anchor_clock(u, np.array([a]))[0])
                    p = int(r['player'][row, t['player_slot'][rr, 0]])
                    on = (p in starters or (p in sub_in and (clock >= (sub_in[p] + 1) * 60 or first.get(p, 10 ** 12) <= a)))
                    on = on and not (p in sub_out and clock >= (sub_out[p] + 1) * 60) and not red.get(p, 10 ** 12) < a
                    unreachable_slow += not on
        self.assertEqual(unreachable_fast, unreachable_slow)
        print('unreachable single-actor validation targets:', unreachable_fast)

    def test_10_models_train_reload_and_mask(self):
        rows = np.arange(256)
        for name, cond in C['conditions'].items():
            torch.manual_seed(1)
            model = Screen(C, cond, V)
            x, y = to_torch(screen_batch(self.d, rows, cond))
            scores = model(x)
            self.assertTrue(torch.isinf(scores[~x['candidates']['mask']]).all())
            per, mask = player_loss(scores, y)
            loss = (per * mask).sum() / mask.sum()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
            model.eval()
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            other = Screen(C, cond, V)
            other.load_state_dict(torch.load(buffer, weights_only=True))
            other.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x), other(x), rtol=0, atol=0)

    def test_12_context_identity_reaches_the_scores(self):
        d = copy.copy(self.d)
        d.base = copy.copy(self.d.base)
        d.base.units = {k: v.copy() for k, v in self.d.base.units.items()}
        rows = np.arange(7000, 7064)
        lo, hi = int(d.base.targets['anchor'][rows].min()) - 80, int(d.base.targets['anchor'][rows].max())
        d.base.units['part_player'][max(lo, 0):hi] = d.base.units['part_player'][max(lo, 0):hi][::-1]
        for name, changes in (('M0_static', False), ('M0_ctx_identity', True)):
            torch.manual_seed(3)
            model = Screen(C, C['conditions'][name], V).eval()
            x, _ = to_torch(screen_batch(self.d, rows, C['conditions'][name]))
            y, _ = to_torch(screen_batch(d, rows, C['conditions'][name]))
            with torch.no_grad():
                a, b = model(x), model(y)
            self.assertEqual(changes, not torch.equal(a, b), name)

    def test_13_existing_conditions_match_the_archived_code(self):
        """The first formal wave ran with the archived code; the current code must reproduce it exactly."""
        old = {}
        for name in ('scr_batches', 'scr_models'):
            spec = importlib.util.spec_from_file_location(f'archived_{name}', ARCHIVE / 'src' / f'{name}.py')
            old[name] = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(old[name])
        old_conditions = json.load(open(ARCHIVE / 'configs/experiment.json'))['conditions']
        self.assertEqual(set(old_conditions), set(IDENTITY_FREE))
        rows = np.arange(20000, 20128)
        for name in IDENTITY_FREE:
            new_cond, old_cond = C['conditions'][name], old_conditions[name]
            self.assertEqual({k: v for k, v in new_cond.items() if k != 'context_identity'}, old_cond)
            x_new, y_new = screen_batch(self.d, rows, new_cond)
            x_old, y_old = old['scr_batches'].screen_batch(self.d, rows, old_cond)
            self.assertTrue(equal(x_new, x_old) and equal(y_new, y_old), name)
            x, y = to_torch((x_new, y_new))
            torch.manual_seed(5)
            new_model = Screen(C, new_cond, V)
            torch.manual_seed(5)
            old_model = old['scr_models'].Screen(C, old_cond, V)
            for (k, a), (k2, b) in zip(new_model.state_dict().items(), old_model.state_dict().items()):
                self.assertEqual(k, k2)
                self.assertTrue(torch.equal(a, b), (name, k))
            outputs = []
            for model in (new_model, old_model):
                torch.manual_seed(9)
                scores = model(x)
                per, mask = player_loss(scores, y)
                ((per * mask).sum() / mask.sum()).backward()
                outputs.append((scores.detach(), [p.grad.clone() for p in model.parameters() if p.grad is not None]))
            self.assertTrue(torch.equal(outputs[0][0], outputs[1][0]), name)
            self.assertEqual(len(outputs[0][1]), len(outputs[1][1]))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(outputs[0][1], outputs[1][1])), name)

    def test_11_scoring(self):
        scores = torch.tensor([[3.0, 2.0, 1.0, float('-inf')], [0.0, 1.0, 2.0, 3.0], [5.0, 1.0, 0.0, -1.0]])
        y = dict(slot=torch.tensor([[1, 3], [0, -1], [3, -1]]), eval_weight=torch.tensor([[0.5, 0.5], [1.0, 0.0], [1.0, 0.0]]),
                 loss_weight=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]), known=torch.tensor([True, True, True]),
                 reachable=torch.tensor([True, True, True]), duel=torch.tensor([True, False, False]))
        s = sample_statistics(scores, y)
        np.testing.assert_allclose(s['top1'], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(s['top3'], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(s['top5'], [0.5, 1.0, 1.0])
        np.testing.assert_allclose(s['rr'], [1 / 2, 1 / 4, 1 / 4])
        yy = {k: v.numpy() for k, v in y.items()}
        yy.update(match_id=np.array([1, 1, 2]), recency=np.array([-1, 2, 0]))
        _, table = match_table(s, yy)
        m = metrics(table.sum(0))
        self.assertAlmostEqual(m['single_top5'], 1.0)
        self.assertAlmostEqual(m['duel_top3'], 0.5)


if __name__ == '__main__':
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScreeningTests))
    (ROOT / 'reports/acceptance.log').write_text(stream.getvalue())
    print(stream.getvalue())
    (ROOT / 'reports/acceptance.json').write_text(json.dumps(dict(status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
                                                                  failures=len(result.failures), errors=len(result.errors), test_evaluated=False), indent=2) + '\n')
    sys.exit(0 if result.wasSuccessful() else 1)
