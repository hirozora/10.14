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
C2 = json.load(open(ROOT / 'configs/stage2_fixed_schedule.json'))
C3 = json.load(open(ROOT / 'configs/stage3a_match_history.json'))
C3B = json.load(open(ROOT / 'configs/stage3b_pair_relations.json'))
from scr_candidates import setup, anchor_clock, on_pitch, NEVER
setup(C)
from units import read_json
from train import deterministic
from scr_batches import ScreenData, screen_batch, history_positions, membership, shuffled_positions
from scr_batches import match_tables, match_history, pair_counts, within_team_permutation, BIG
from scr_models import Screen, player_loss
from scr_fit import to_torch, fixed_step_lr
from scr_scoring import sample_statistics, match_table, metrics
V = read_json(Path(C['paths']['data']) / 'vocabulary.json')
ARCHIVE = ROOT / 'archive/before_context_identity_20261014'
STAGE1_ARCHIVE = ROOT / 'archive/stage1_final_20261015'
STAGE2_ARCHIVE = ROOT / 'archive/stage2_final_20261015'
IDENTITY_FREE = [name for name, cond in C['conditions'].items() if not cond['context_identity']]
# Pair-relation conditions of question 3b, built on H_match (the user's choice, 2026-10-15).
PAIR_CONDITIONS = C3B['conditions']
ALL_CONDITIONS = {**C['conditions'], **C2['conditions'], **C3['conditions'], **PAIR_CONDITIONS}
NEUTRAL = dict(scope='window', pairs='none')
STAGE3 = {**C3['conditions'], **PAIR_CONDITIONS}
PLATEAU_ONLY = {'warmup_no_decay_epochs', 'min_stop_epoch', 'relative_improvement', 'patience_validations', 'lr_decay_factor', 'max_lr_decays'}


def archived(folder, name, tag):
    spec = importlib.util.spec_from_file_location(f'{tag}_{name}', folder / 'src' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        for name, cond in ALL_CONDITIONS.items():
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
        for name, changes in (('M0_static', False), ('M0_ctx_identity', True), ('M0_ctx_identity_dot', True)):
            torch.manual_seed(3)
            model = Screen(C, ALL_CONDITIONS[name], V).eval()
            if name == 'M0_ctx_identity_dot':
                torch.nn.init.normal_(model.dot.weight, std=0.1)
            x, _ = to_torch(screen_batch(self.d, rows, ALL_CONDITIONS[name]))
            y, _ = to_torch(screen_batch(d, rows, ALL_CONDITIONS[name]))
            with torch.no_grad():
                a, b = model(x), model(y)
            self.assertEqual(changes, not torch.equal(a, b), name)

    def assert_same_as_archive(self, old, new_cond, old_cond, name):
        """Batches, initial parameters, training-mode scores and gradients equal those of archived code, bit for bit."""
        rows = np.arange(20000, 20128)
        x_new, y_new = screen_batch(self.d, rows, new_cond)
        x_old, y_old = old['scr_batches'].screen_batch(self.d, rows, old_cond)
        self.assertTrue(equal(x_new, x_old) and equal(y_new, y_old), name)
        x, y = to_torch((x_new, y_new))
        torch.manual_seed(5)
        new_model = Screen(C, new_cond, V)
        torch.manual_seed(5)
        old_model = old['scr_models'].Screen(C, old_cond, V)
        self.assertEqual(list(new_model.state_dict()), list(old_model.state_dict()), name)
        for (k, a), (k2, b) in zip(new_model.state_dict().items(), old_model.state_dict().items()):
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

    def test_13_existing_conditions_match_the_archived_code(self):
        """The first formal wave ran with the archived code; the current code must reproduce it exactly."""
        old = {name: archived(ARCHIVE, name, 'archived') for name in ('scr_batches', 'scr_models')}
        old_conditions = json.load(open(ARCHIVE / 'configs/experiment.json'))['conditions']
        self.assertEqual(set(old_conditions), set(IDENTITY_FREE))
        for name in IDENTITY_FREE:
            new_cond, old_cond = C['conditions'][name], old_conditions[name]
            self.assertEqual({k: v for k, v in new_cond.items() if k not in ('context_identity', 'scorer', 'scope', 'pairs')}, old_cond)
            self.assertEqual((new_cond['scorer'], new_cond['scope'], new_cond['pairs']), ('mlp', 'window', 'none'))
            self.assert_same_as_archive(old, new_cond, old_cond, name)

    def test_14_stage1_conditions_match_the_final_stage1_code(self):
        """Stage 1 finished with the code in archive/stage1_final_20261015; the current code must reproduce all six conditions exactly."""
        old = {name: archived(STAGE1_ARCHIVE, name, 'stage1') for name in ('scr_batches', 'scr_models')}
        old_conditions = json.load(open(STAGE1_ARCHIVE / 'configs/experiment.json'))['conditions']
        self.assertEqual(list(old_conditions), list(C['conditions']))
        for name, new_cond in C['conditions'].items():
            self.assertEqual({k: v for k, v in new_cond.items() if k not in ('scorer', 'scope', 'pairs')}, old_conditions[name])
            self.assertEqual((new_cond['scorer'], new_cond['scope'], new_cond['pairs']), ('mlp', 'window', 'none'))
            self.assert_same_as_archive(old, new_cond, old_conditions[name], name)

    def test_15_dot_scorer_starts_as_the_mlp_scorer(self):
        rows = np.arange(30000, 30256)
        plain, dot = C2['conditions']['M0_ctx_identity'], C2['conditions']['M0_ctx_identity_dot']
        self.assertEqual({k: v for k, v in dot.items() if k != 'scorer'}, {k: v for k, v in plain.items() if k != 'scorer'})
        self.assertEqual((plain['scorer'], dot['scorer']), ('mlp', 'mlp_plus_dot'))
        x, y = to_torch(screen_batch(self.d, rows, dot))
        torch.manual_seed(7)
        a = Screen(C2, plain, V)
        torch.manual_seed(7)
        b = Screen(C2, dot, V)
        self.assertEqual(list(b.state_dict()), list(a.state_dict()) + ['dot.weight'])
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        self.assertTrue(torch.equal(b.dot.weight, torch.zeros_like(b.dot.weight)))
        torch.manual_seed(11)
        sa = a(x)
        torch.manual_seed(11)
        sb = b(x)
        self.assertTrue(torch.equal(sa, sb))
        # One optimiser step moves the dot term away from zero, and the scores then differ.
        per, mask = player_loss(sb, y)
        ((per * mask).sum() / mask.sum()).backward()
        self.assertGreater(float(b.dot.weight.grad.abs().sum()), 0.0)
        torch.optim.AdamW(b.parameters(), lr=2.7e-3, weight_decay=1e-4).step()
        self.assertGreater(float(b.dot.weight.abs().sum()), 0.0)
        a.eval(), b.eval()
        with torch.no_grad():
            self.assertFalse(torch.equal(a(x), b(x)))

    def test_16_fixed_step_schedule(self):
        s = C2['training']['schedule']
        self.assertEqual((s['kind'], s['milestones'], s['factor']), ('fixed_steps', [96, 108], 0.3))
        lr = C2['learning_rate']['value']
        got = [fixed_step_lr(s, lr, e) for e in range(1, C2['training']['max_epochs'] + 1)]
        self.assertEqual(got[:96], [lr] * 96)
        np.testing.assert_allclose(got[96:108], [lr * 0.3] * 12, rtol=1e-12)
        np.testing.assert_allclose(got[108:], [lr * 0.09] * 12, rtol=1e-12)
        self.assertEqual(C['training']['schedule'], {'kind': 'plateau'})

    def test_17_stage2_config_differs_only_where_declared(self):
        for k in ('root', 'five_task_src', 'parent_src', 'dataset', 'candidates', 'context', 'candidate_features', 'shuffle_seed', 'evaluation', 'decision'):
            self.assertEqual(C[k], C2[k], k)
        for k in C['paths']:
            if k in ('runs', 'reports'):
                self.assertNotEqual(C['paths'][k], C2['paths'][k], k)
            else:
                self.assertEqual(C['paths'][k], C2['paths'][k], k)
        self.assertEqual(set(C['paths']), set(C2['paths']))
        new = set(C2['conditions']) - set(C['conditions'])
        self.assertEqual(new, {'M0_ctx_identity_dot'})
        for name in set(C2['conditions']) - new:
            self.assertEqual(C['conditions'][name], C2['conditions'][name], name)
        t1, t2 = C['training'], C2['training']
        self.assertEqual(set(t1) - PLATEAU_ONLY, set(t2))
        for k in t2:
            if k != 'schedule':
                self.assertEqual(t1[k], t2[k], k)
        locked = json.load(open(C2['learning_rate']['locked_by']))
        self.assertEqual((locked['status'], locked['selected']), ('locked', C2['learning_rate']['value']))
        for k in ('python', 'gpus', 'jobs_per_gpu'):
            self.assertEqual(C['execution'][k], C2['execution'][k], k)
        for config in (C, C2):
            for a, b, _ in config['report']['contrasts']:
                self.assertIn(a, config['conditions'])
                self.assertIn(b, config['conditions'])
        self.assertFalse(C2['test_evaluated'])

    def test_18_match_tables_match_a_slow_reference(self):
        d = self.d
        u, r, mt = d.base.units, d.base.roster, d.match
        S = r['player'].shape[1]
        for row in (0, 13, 29, 56):
            slot_of = {int(p): s for s, p in enumerate(r['player'][row]) if r['valid'][row, s]}
            units = np.flatnonzero(u['match_row'] == row)
            slots = {int(i): [slot_of[int(p)] for p in u['part_player'][i] if p > 0 and int(p) in slot_of] for i in units}
            for i in units:
                self.assertEqual(sorted(slots[int(i)]), sorted(int(s) for s in mt['slot'][i] if s >= 0), int(i))
            for s in range(S):
                expect = [int(i) for i in units if s in slots[int(i)]]
                lo, hi = np.searchsorted(mt['act_key'], [(row * S + s) * BIG, (row * S + s + 1) * BIG])
                self.assertEqual(expect, (mt['act_key'][lo:hi] % BIG).tolist(), (row, s))
            expect = sorted((int(i), a, b) for i in units[1:] for a in slots[int(i) - 1] for b in slots[int(i)] if a != b)
            inside = (mt['t_unit'] >= units[0]) & (mt['t_unit'] <= units[-1])
            got = sorted(zip(mt['t_unit'][inside].tolist(), mt['t_from'][inside].tolist(), mt['t_to'][inside].tolist()))
            self.assertEqual(expect, got, row)
            self.assertGreater(len(expect), 500)

    def slow_match_state(self, row_index):
        """Per-slot acting units and the pair count / latest-unit matrices of one sample, by scanning its match."""
        d = self.d
        u, r, t = d.base.units, d.base.roster, d.base.targets
        mr, a, start = int(t['match_row'][row_index]), int(t['anchor'][row_index]), int(t['match_start'][row_index])
        S = r['player'].shape[1]
        slot_of = {int(p): s for s, p in enumerate(r['player'][mr]) if r['valid'][mr, s]}
        slots = [[slot_of[int(p)] for p in u['part_player'][i] if p > 0 and int(p) in slot_of] for i in range(start, a + 1)]
        acts = [[start + k for k, x in enumerate(slots) if s in x] for s in range(S)]
        count, last = np.zeros((S, S), np.int64), np.full((S, S), -1, np.int64)
        for k in range(1, len(slots)):
            for p in slots[k - 1]:
                for q in slots[k]:
                    if p != q:
                        count[p, q] += 1
                        last[p, q] = start + k
        return acts, count, last, slots[-1]

    def test_19_match_history_and_summary(self):
        d = self.d
        rows = np.arange(500, len(d.base.targets['anchor']), 2003)
        t = d.base.targets
        S, K, W = d.base.roster['player'].shape[1], d.K, C['dataset']['window_units']
        units, counts = match_history(d, t['match_row'][rows], t['anchor'][rows], S, K)
        xm, _ = screen_batch(d, rows, C3['conditions']['H_match'])
        x2, _ = screen_batch(d, rows, C2['conditions']['M2_pooled'])
        cm, c2 = xm['candidates'], x2['candidates']
        np.testing.assert_array_equal(cm['summary'][..., :5], c2['summary'])
        checked_outside = 0
        for i, row_index in enumerate(rows):
            acts, _, _, _ = self.slow_match_state(row_index)
            a = int(t['anchor'][row_index])
            for s in range(S):
                expect = acts[s][::-1][:K]
                self.assertEqual(expect, [int(v) for v in units[i, s] if v >= 0], (row_index, s))
                self.assertEqual(len(acts[s]), counts[i, s])
                if cm['mask'][i, s]:
                    n = len(acts[s])
                    since = np.log1p(a - acts[s][-1]) / 8.0 if n else 0.0
                    np.testing.assert_allclose(cm['summary'][i, s, 5:], [np.log1p(n), since, float(n == 0)], rtol=1e-6, atol=1e-7)
                    # The window part of the match history repeats the window history exactly.
                    n2 = int(c2['h_mask'][i, s].sum())
                    for k in ('h_kind', 'h_sub', 'h_team'):
                        np.testing.assert_array_equal(cm[k][i, s, :n2], c2[k][i, s, :n2])
                    np.testing.assert_array_equal(cm['h_numeric'][i, s, :n2, :6], c2['h_numeric'][i, s, :n2])
                    self.assertTrue((cm['h_numeric'][i, s, :n2, 6] == 0).all())
                    outside = [v for v in expect if a - v >= W]
                    self.assertEqual(int(cm['h_numeric'][i, s, :, 6].sum()), len(outside))
                    checked_outside += len(outside)
        self.assertGreater(checked_outside, 1000)

    def test_20_pairs_match_a_slow_reference(self):
        d = self.d
        t, r = d.base.targets, d.base.roster
        rows = np.arange(700, len(t['anchor']), 2999)
        S = r['player'].shape[1]
        count, last = pair_counts(d, t['match_start'][rows], t['anchor'][rows], S)
        xa, _ = screen_batch(d, rows, PAIR_CONDITIONS['P_anchor'])
        xw, _ = screen_batch(d, rows, PAIR_CONDITIONS['P_window'])
        np.testing.assert_array_equal(xa['candidates']['pair_anchor'], xw['candidates']['pair_anchor'])
        cw = xw['candidates']
        W = C['dataset']['window_units']
        g_member = membership(*self.window_of(rows))
        tokens_checked = 0
        for i, row_index in enumerate(rows):
            acts, cnt, lst, anchor_slots = self.slow_match_state(row_index)
            np.testing.assert_array_equal(count[i], cnt)
            np.testing.assert_array_equal(last[i], lst)
            a = int(t['anchor'][row_index])
            value = lambda n_in, n_out, latest: [np.log1p(n_in), np.log1p(n_out), np.log1p(a - latest) / 8.0 if latest >= 0 else 0.0, float(latest < 0)]
            team = r['team'][t['match_row'][row_index]]
            seen = g_member[i].any(0)
            window_count = g_member[i].sum(0)
            for s in range(S):
                if not cw['mask'][i, s]:
                    self.assertFalse(cw['pair_mask'][i, s].any())
                    continue
                expect = value(sum(cnt[p, s] for p in anchor_slots), sum(cnt[s, p] for p in anchor_slots), max([lst[p, s] for p in anchor_slots], default=-1))
                np.testing.assert_allclose(cw['pair_anchor'][i, s], expect, rtol=1e-6, atol=1e-7)
                np.testing.assert_array_equal(cw['pair_mask'][i, s], seen & (np.arange(S) != s))
                for p in np.flatnonzero(cw['pair_mask'][i, s]):
                    last_seen = W - 1 - int(np.flatnonzero(g_member[i, :, p])[-1])
                    expect = value(cnt[p, s], cnt[s, p], lst[p, s]) + [np.log1p(window_count[p]), last_seen / 80.0, float(p in anchor_slots and g_member[i, -1, p]), float(team[p] == team[s])]
                    np.testing.assert_allclose(cw['pair_tokens'][i, s, p], expect, rtol=1e-6, atol=1e-7)
                    tokens_checked += 1
        self.assertGreater(tokens_checked, 2000)

    def window_of(self, rows):
        from batches import gather
        g = gather(self.d.base, rows)
        return g, self.d.base.roster['player'][g['match_row']]

    def test_21_stage3_inputs_ignore_the_future(self):
        for row in (2000, 40000, 80000):
            d = copy.copy(self.d)
            d.base = copy.copy(self.d.base)
            d.base.units = {k: v.copy() for k, v in self.d.base.units.items()}
            cut = int(d.base.targets['anchor'][row]) + 1
            for k in ('event', 'sub_code', 'raw_sub', 'part_player', 'part_team', 'xy_home', 'seconds', 'merged', 'loc_valid', 'ref_team'):
                d.base.units[k][cut:] = d.base.units[k][cut:][::-1]
            d.match = match_tables(d.base)
            for name, cond in STAGE3.items():
                a, _ = screen_batch(self.d, np.array([row]), cond)
                b, _ = screen_batch(d, np.array([row]), cond)
                self.assertTrue(equal(a, b), (row, name))
            # Positive control: who acted earlier in the match does change the pair inputs.
            lo = int(d.base.targets['match_start'][row])
            d.base.units['part_player'][lo:cut] = d.base.units['part_player'][lo:cut][::-1]
            d.match = match_tables(d.base)
            a, _ = screen_batch(self.d, np.array([row]), PAIR_CONDITIONS['P_window'])
            b, _ = screen_batch(d, np.array([row]), PAIR_CONDITIONS['P_window'])
            self.assertFalse(equal(a['candidates']['pair_tokens'], b['candidates']['pair_tokens']), row)

    def test_22_stage3_shuffled_controls(self):
        d = self.d
        rows = np.arange(15000, 15128)
        t = d.base.targets
        S = d.base.roster['player'].shape[1]
        x, _ = screen_batch(d, rows, C3['conditions']['H_match'])
        y, _ = screen_batch(d, rows, C3['conditions']['H_match_shuffled'])
        cx, cy = x['candidates'], y['candidates']
        for k in ('summary', 'vocab', 'mask'):
            np.testing.assert_array_equal(cx[k], cy[k])
        team = d.base.roster['team'][t['match_row'][rows]]
        groups = [[np.flatnonzero(cx['mask'][i] & (team[i] == tm)) for tm in np.unique(team[i][cx['mask'][i]])] for i in range(len(rows))]
        perm = within_team_permutation(groups, C3['shuffle_seed'], t['anchor'][rows], S)
        bi = np.arange(len(rows))[:, None]
        # Unit features do not depend on whose history holds the unit, so the shuffled inputs are the real ones permuted.
        for k in ('h_kind', 'h_sub', 'h_team', 'h_numeric', 'h_mask'):
            np.testing.assert_array_equal(cy[k], cx[k][bi, perm])
        self.assertGreater(int((perm != np.arange(S)).sum()), 1000)
        # The same permutation as the window-history control.
        pos = np.random.default_rng(0).integers(-1, 80, (len(rows), S, 4))
        np.testing.assert_array_equal(shuffled_positions(pos, groups, C3['shuffle_seed'], t['anchor'][rows]), pos[bi, perm])
        x, _ = screen_batch(d, rows, PAIR_CONDITIONS['P_window'])
        y, _ = screen_batch(d, rows, PAIR_CONDITIONS['P_window_shuffled'])
        cx, cy = x['candidates'], y['candidates']
        for k in ('pair_anchor', 'pair_tokens', 'pair_mask'):
            np.testing.assert_array_equal(cy[k], cx[k][bi, perm])
        for k in ('summary', 'h_kind', 'h_numeric', 'mask'):
            np.testing.assert_array_equal(cx[k], cy[k])

    def test_23_pair_branches_start_at_zero(self):
        rows = np.arange(30000, 30256)
        base = C3['conditions']['H_match']
        for name, cond in PAIR_CONDITIONS.items():
            self.assertEqual({k: v for k, v in cond.items() if k != 'pairs'}, {k: v for k, v in base.items() if k != 'pairs'})
            x, y = to_torch(screen_batch(self.d, rows, cond))
            torch.manual_seed(7)
            a = Screen(C3, base, V)
            torch.manual_seed(7)
            b = Screen(C3, cond, V)
            extra = [k for k in b.state_dict() if k not in a.state_dict()]
            self.assertEqual(list(b.state_dict())[:len(a.state_dict())], list(a.state_dict()))
            self.assertTrue(all(k.startswith('pair_') for k in extra) and extra)
            self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
            a.eval(), b.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(a(x), b(x)), name)
            b.train()
            per, mask = player_loss(b(x), y)
            ((per * mask).sum() / mask.sum()).backward()
            torch.optim.AdamW(b.parameters(), lr=2.7e-3, weight_decay=1e-4).step()
            b.eval()
            with torch.no_grad():
                self.assertFalse(torch.equal(a(x), b(x)), name)

    def test_24_stage2_conditions_match_the_stage2_code(self):
        """Stage 2 ran with the code in archive/stage2_final_20261015; the current code must reproduce its conditions exactly."""
        old = {name: archived(STAGE2_ARCHIVE, name, 'stage2') for name in ('scr_batches', 'scr_models')}
        old_conditions = json.load(open(STAGE2_ARCHIVE / 'configs/stage2_fixed_schedule.json'))['conditions']
        self.assertEqual(list(old_conditions), list(C2['conditions']))
        for name, new_cond in C2['conditions'].items():
            self.assertEqual({k: v for k, v in new_cond.items() if k not in NEUTRAL}, old_conditions[name])
            self.assertEqual({k: new_cond[k] for k in NEUTRAL}, NEUTRAL)
            self.assert_same_as_archive(old, new_cond, old_conditions[name], name)

    def test_25_stage3_config_differs_only_where_declared(self):
        for k in ('root', 'five_task_src', 'parent_src', 'dataset', 'candidates', 'context', 'shuffle_seed', 'evaluation', 'decision', 'training',
                  'learning_rate', 'execution', 'scorers'):
            self.assertEqual(C2[k], C3[k], k)
        for k in C2['paths']:
            if k in ('runs', 'reports'):
                self.assertNotIn(C3['paths'][k], (C['paths'][k], C2['paths'][k]), k)
            else:
                self.assertEqual(C2['paths'][k], C3['paths'][k], k)
        self.assertEqual(set(C2['paths']), set(C3['paths']))
        self.assertEqual({k: v for k, v in C3['candidate_features'].items() if k not in ('match_scope', 'pairs')}, C2['candidate_features'])
        self.assertFalse(set(C3['conditions']) & (set(C['conditions']) | set(C2['conditions'])))
        for name, cond in C3['conditions'].items():
            self.assertEqual((cond['scope'], cond['context_identity'], cond['scorer'], cond['summary'], cond['static']), ('match', False, 'mlp', True, True), name)
        self.assertEqual(C3['report']['reference_runs'], {'M2_pooled': {'runs': C2['paths']['runs'], 'version': C2['version']}})
        named = set(C3['conditions']) | set(C3['report']['reference_runs'])
        for a, b, _ in C3['report']['contrasts']:
            self.assertIn(a, named)
            self.assertIn(b, named)
        self.assertFalse(C3['test_evaluated'])

    def test_26_stage3b_config_differs_only_where_declared(self):
        for k in ('root', 'five_task_src', 'parent_src', 'dataset', 'candidates', 'context', 'shuffle_seed', 'evaluation', 'decision', 'training',
                  'learning_rate', 'scorers'):
            self.assertEqual(C3[k], C3B[k], k)
        # Execution differs only in runs per GPU, which does not change results.
        self.assertEqual({k: v for k, v in C3B['execution'].items() if k != 'jobs_per_gpu'}, {k: v for k, v in C3['execution'].items() if k != 'jobs_per_gpu'})
        for k in C3['paths']:
            if k in ('runs', 'reports'):
                self.assertNotIn(C3B['paths'][k], (C['paths'][k], C2['paths'][k], C3['paths'][k]), k)
            else:
                self.assertEqual(C3['paths'][k], C3B['paths'][k], k)
        self.assertEqual(set(C3['paths']), set(C3B['paths']))
        self.assertEqual({k: v for k, v in C3B['candidate_features'].items() if k != 'pairs'}, C3['candidate_features'])
        self.assertEqual(list(C3B['conditions']), ['P_anchor', 'P_window', 'P_window_shuffled'])
        for name, cond in C3B['conditions'].items():
            self.assertEqual(dict(cond, pairs='none'), C3['conditions']['H_match'], name)
        self.assertEqual(C3B['report']['reference_runs'], {'H_match': {'runs': C3['paths']['runs'], 'version': C3['version']}})
        named = set(C3B['conditions']) | set(C3B['report']['reference_runs'])
        for a, b, _ in C3B['report']['contrasts']:
            self.assertIn(a, named)
            self.assertIn(b, named)
        self.assertFalse(C3B['test_evaluated'])

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
