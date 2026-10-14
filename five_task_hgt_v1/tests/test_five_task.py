"""Acceptance tests; the run writes reports/acceptance.json and reports/acceptance.log."""
import copy
import io
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from units import read_json, write_json, match_units, roster_of, parent
from batches import SplitData, gather, hgt_batch, seq_batch, to_torch, roster_slots
from models import build, objective_parts, five_task, backward, FastHGTConv
from scoring import sample_statistics, match_tables, metric_array, paired_bootstrap, bootstrap_weights, VIEWS

C = read_json(ROOT / 'configs/experiment.json')
V = read_json(ROOT / 'data/vocabulary.json')
P = parent(C)
sys.path.insert(0, C['parent_src'])
from train import deterministic


def synthetic(event, sub, team, player, x, y, sec, period='1H'):
    return dict(eventId=event, subEventId=sub, matchPeriod=period, eventSec=sec, positions=[dict(x=x, y=y), dict(x=99, y=1)],
                teamId=team, playerId=player, tags=[dict(id=1801)], id=0, matchId=0)


class FiveTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deterministic(20260715)
        cls.val = SplitData(C, 'validation')
        cls.train = SplitData(C, 'train')

    def test_01_whitelist_and_roster_fields(self):
        e = synthetic(8, 85, 1, 5, 30, 40, 12.0)
        changed = dict(e, tags=[dict(id=301)], positions=[dict(x=30, y=40), dict(x=1, y=99)])
        self.assertEqual(P.clean_event(e), P.clean_event(changed))
        match = json.loads(json.dumps(next(m for m in read_json(C['paths']['raw_matches']) if m['wyId'] == int(self.val.matches['match_id'][0]))))
        base = roster_of(match)
        for td in match['teamsData'].values():
            for group in ('lineup', 'bench'):
                for entry in td['formation'][group]:
                    entry.update(goals='9', ownGoals='9', redCards='1', yellowCards='1')
            td['formation']['substitutions'] = []
        self.assertEqual(base, roster_of(match))

    def test_02_pair_merge_is_order_free(self):
        a = P.clean_event(synthetic(1, 11, 7, 70, 30, 40, 10.0))
        b = P.clean_event(synthetic(1, 12, 9, 90, 70, 60, 10.2))
        pre = P.clean_event(synthetic(8, 85, 9, 91, 20, 50, 9.0))
        u1 = match_units([pre, a, b], home=7)
        u2 = match_units([pre, dict(b, seconds=10.0), dict(a, seconds=10.2)], home=7)
        self.assertEqual(len(u1), 2)
        for k in ('event', 'pair', 'parts', 'ref', 'merged', 'period'):
            self.assertEqual(u1[1][k], u2[1][k])
        np.testing.assert_allclose(u1[1]['xy'], u2[1]['xy'], atol=1e-12)
        self.assertEqual(u1[1]['parts'][0][1], 7)
        self.assertEqual(u1[1]['ref'], 9)
        np.testing.assert_allclose(u1[1]['xy'], [0.3, 0.4], atol=1e-12)

    def test_03_units_and_targets(self):
        for d in (self.train, self.val):
            u, t = d.units, d.targets
            merged = u['merged']
            self.assertTrue((u['part_team'][merged, 0] != u['part_team'][merged, 1]).all())
            self.assertTrue((u['part_team'][merged, 0] == d.matches['home_team'][u['match_row'][merged]]).all())
            self.assertTrue((u['event'][merged] == 1).all() and (u['loc_valid'][merged]).all())
            self.assertTrue((u['part_team'][~merged, 1] == 0).all())
            nxt = t['anchor'] + 1
            np.testing.assert_array_equal(t['target_duel'], merged[nxt])
            np.testing.assert_array_equal(t['event'], u['event'][nxt] - 1)
            self.assertTrue(np.allclose(t['team_same'][t['target_duel']], 0.5))
            w = t['player_weight'][t['player_mask']]
            np.testing.assert_allclose(w.sum(1), 1.0)
            duel_known = t['target_duel'] & t['player_mask']
            np.testing.assert_allclose(t['player_weight'][duel_known], 0.5)
            single = ~t['target_duel']
            self.assertTrue(np.isin(t['team_same'][single], [0.0, 1.0]).all())
            home = d.matches['home_team'][t['match_row']]
            ref = u['ref_team'][t['anchor']]
            expect = np.where((ref == home)[:, None], u['xy_home'][nxt], 1 - u['xy_home'][nxt])
            np.testing.assert_allclose(t['xy'][t['pos_mask']], expect[t['pos_mask']], atol=1e-6)
            same = u['period'][nxt] == u['period'][t['anchor']]
            np.testing.assert_array_equal(t['time_mask'], same)
            # Reference team of merged duels: the latest preceding single-actor unit of the match.
            for i in np.flatnonzero(merged)[:2000]:
                j = i - 1
                while j >= 0 and u['match_row'][j] == u['match_row'][i] and u['merged'][j]:
                    j -= 1
                if j >= 0 and u['match_row'][j] == u['match_row'][i]:
                    self.assertEqual(u['ref_team'][i], u['part_team'][j, 0])

    def test_04_future_and_target_invariance(self):
        for row in (3000, 5000, 70000):
            d = copy.copy(self.val)
            d.units = {k: v.copy() for k, v in self.val.units.items()}
            d.targets = {k: v.copy() for k, v in self.val.targets.items()}
            cut = int(d.targets['anchor'][row]) + 1
            for k in ('event', 'sub_code', 'raw_sub', 'part_player', 'part_team', 'xy_home', 'seconds', 'merged', 'loc_valid', 'ref_team'):
                d.units[k][cut:] = d.units[k][cut:][::-1]
            for k in ('event', 'time', 'xy', 'team_same', 'player_slot', 'player_weight', 'player_mask', 'target_duel'):
                d.targets[k] = d.targets[k][::-1].copy()
            rows = np.array([row])
            for name in ('hgt_full', 'hgt_shuffled_player', 'seq_mean'):
                cond = C['conditions'][name]
                make = (lambda data: hgt_batch(data, rows, cond, C['relations'])[0]) if cond['family'] == 'hgt' else (lambda data: seq_batch(data, rows)[0])
                self.assertTrue(equal_nested(make(self.val), make(d)), (row, name))

    def test_05_relation_firewall(self):
        rows = np.arange(2000, 2064)
        for name, cond in C['conditions'].items():
            if cond['family'] != 'hgt':
                continue
            x, _ = hgt_batch(self.val, rows, cond, C['relations'])
            allowed = {tuple(e) for f in cond['relations'] for e in C['relations'][f]}
            self.assertEqual(set(x['edges']), allowed, name)
            model = build(C, cond, V)
            self.assertEqual(set(model.edges), allowed)
            self.assertTrue(set(x['x']) == set(model.types), name)

    def test_06_shuffled_control(self):
        rows = np.arange(4000, 4256)
        full = hgt_batch(self.val, rows, C['conditions']['hgt_full'], C['relations'])[0]
        shuf = hgt_batch(self.val, rows, C['conditions']['hgt_shuffled_player'], C['relations'])[0]
        key = ('player', 'performs', 'event')
        a, b = full['edges'][key], shuf['edges'][key]
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_array_equal(np.sort(a[1]), np.sort(b[1]))
        g = gather(self.val, rows)
        team_of_node = self.val.roster['team'][g['match_row']][full['player_id'] >= 0]
        np.testing.assert_array_equal(team_of_node[a[0]], team_of_node[b[0]])
        self.assertGreater((a[0] != b[0]).mean(), 0.8)
        for k in full['edges']:
            if k[1] not in ('performs', 'performed_by') or k[0] == 'team' or k[2] == 'team':
                np.testing.assert_array_equal(full['edges'][k], shuf['edges'][k])

    def test_07_coordinate_frame_probe(self):
        rows = np.arange(0, len(self.val.targets['anchor']), 7)
        g = gather(self.val, rows)
        single = g['valid'] & ~g['merged'] & g['loc_valid']
        pair = single[:, :-1] & single[:, 1:] & g['dt_valid'][:, 1:]
        a, b = g['xy'][:, :-1][pair] * [105, 68], g['xy'][:, 1:][pair] * [105, 68]
        change = (g['part_team'][:, :-1, 0] != g['part_team'][:, 1:, 0])[pair]
        guess = np.hypot(*(b - ([105, 68] - a)).T) < np.hypot(*(b - a).T)
        accuracy = (guess == change).mean()
        majority = max(change.mean(), 1 - change.mean())
        self.assertLessEqual(accuracy, majority)

    def test_08_padding_and_nodes(self):
        rows = np.flatnonzero(self.val.targets['anchor'] == self.val.targets['match_start'])[:4]
        x, _ = hgt_batch(self.val, rows, C['conditions']['hgt_full'], C['relations'])
        self.assertEqual(len(x['event_batch']), len(rows))
        np.testing.assert_array_equal(x['anchor_event'], np.arange(len(rows)))
        rows = np.array([10, 100, 1000])
        g = gather(self.val, rows)
        x, _ = hgt_batch(self.val, rows, C['conditions']['hgt_full'], C['relations'])
        self.assertEqual(len(x['x']['event']['kind']), int(g['valid'].sum()))
        slots = roster_slots(self.val, g)
        self.assertTrue(((slots >= 0) == ((g['part_player'] > 0) & g['valid'][..., None])).all())

    def test_09_fast_hgt_equals_pyg(self):
        rows = np.arange(6000, 6128)
        for name in ('hgt_full', 'hgt_events_only', 'hgt_no_member'):
            cond = C['conditions'][name]
            torch.manual_seed(1)
            model = build(C, cond, V).double()
            x, _ = to_torch(hgt_batch(self.val, rows, cond, C['relations']))
            x['x'] = {t: {k: (v.double() if v.is_floating_point() else v) for k, v in d.items()} for t, d in x['x'].items()}
            states = model.encode(x['x'])
            edges = {k: v for k, v in x['edges'].items() if v.shape[1] > 0}
            for layer in (model.convolutions[0], model.player_convolution):
                a, b = layer(states, edges), layer.conv(states, edges)
                self.assertEqual(sorted(a), sorted(k for k, v in b.items() if v is not None))
                for t in a:
                    torch.testing.assert_close(a[t], b[t], rtol=0, atol=1e-12)
                w = {t: torch.randn_like(a[t]) for t in a}
                ga = torch.autograd.grad(sum((a[t] * w[t]).sum() for t in a), list(layer.parameters()), allow_unused=True)
                b = layer.conv(states, edges)
                gb = torch.autograd.grad(sum((b[t] * w[t]).sum() for t in a), list(layer.parameters()), allow_unused=True)
                for p, q in zip(ga, gb):
                    if p is not None or q is not None:
                        torch.testing.assert_close(p, q, rtol=0, atol=1e-9)

    def test_10_every_condition_trains_and_reloads(self):
        rows = np.arange(256)
        for name, cond in C['conditions'].items():
            torch.manual_seed(2)
            model = build(C, cond, V)
            batch = hgt_batch(self.val, rows, cond, C['relations']) if cond['family'] == 'hgt' else seq_batch(self.val, rows)
            x, y = to_torch(batch)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            parts = objective_parts(model(x), y)
            self.assertTrue(torch.isfinite(five_task(parts)), name)
            backward(model, parts)
            for p in model.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
            optimizer.step()
            model.eval()
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            other = build(C, cond, V)
            other.load_state_dict(torch.load(buffer, weights_only=True))
            other.eval()
            with torch.no_grad():
                a, b = model(x), other(x)
            for k in a:
                torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)

    def test_11_player_gradient_rule(self):
        cond = C['conditions']['hgt_full']
        x, y = to_torch(hgt_batch(self.val, np.arange(64), cond, C['relations']))
        torch.manual_seed(3)
        model = build(C, cond, V)
        model.eval()
        parts = objective_parts(model(x), y)
        backward(model, parts)
        rule = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None and n.startswith(('convolutions.0.', 'norms.0.'))}
        model.zero_grad()
        parts = objective_parts(model(x), y)
        player = torch.autograd.grad(0.4 * parts['player'] / 3, [p for n, p in model.named_parameters() if n in rule], retain_graph=True, allow_unused=True)
        rest = torch.autograd.grad((0.2 * parts['event'] + parts['time'] + parts['position'] + 0.05 * parts['team']) / 3,
                                   [p for n, p in model.named_parameters() if n in rule], allow_unused=True)
        for (n, g), gp, gr in zip(rule.items(), player, rest):
            expect = (0 if gp is None else 0.25 * gp) + (0 if gr is None else gr)
            torch.testing.assert_close(g, expect, rtol=1e-5, atol=1e-8)

    def test_12_scoring(self):
        y = dict(event=torch.tensor([1, 2]), time=torch.tensor([1.0, 2.0]), time_mask=torch.tensor([True, True]), xy=torch.rand(2, 2),
                 pos_mask=torch.tensor([True, False]), team_same=torch.tensor([0.5, 1.0]), team_mask=torch.tensor([True, True]),
                 player_slot=torch.tensor([[0, 2], [1, -1]]), player_weight=torch.tensor([[0.5, 0.5], [1.0, 0.0]]),
                 player_mask=torch.tensor([True, True]))
        scores = torch.tensor([[3.0, 2.0, 1.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
        pred = dict(event_logits=torch.zeros(2, 10), time_seconds=torch.tensor([1.0, 1.0]), position_xy=torch.rand(2, 2),
                    team_logits=torch.tensor([[0.0, 1.0], [0.0, 1.0]]), player_scores=scores)
        s = sample_statistics(pred, y)
        np.testing.assert_allclose(s['player_top1'], [0.5, 0.0])
        np.testing.assert_allclose(s['player_top3'], [1.0, 1.0])
        np.testing.assert_allclose(s['team_credit'], [0.5, 1.0])
        logp = torch.log_softmax(scores, -1)
        np.testing.assert_allclose(s['player_ce'], [-(0.5 * logp[0, 0] + 0.5 * logp[0, 2]).item(), -logp[1, 1].item()], rtol=1e-6)
        yy = {k: v.numpy() for k, v in y.items()}
        yy.update(match_id=np.array([1, 2]), target_duel=np.array([True, False]), anchor_duel=np.array([False, False]))
        matches, tables = match_tables(s, yy)
        self.assertEqual(set(tables), set(VIEWS))
        m = metric_array(tables['all'].sum(0))
        self.assertAlmostEqual(m['player_top1'], 0.25)
        self.assertAlmostEqual(metric_array(tables['duel_target'].sum(0))['team_accuracy'], 0.5)
        stacked = np.stack([tables['all']] * 2)
        result = paired_bootstrap(stacked, stacked, bootstrap_weights(2, 100, 1))
        self.assertEqual(result['five_task_loss']['mean'], 0.0)

    def test_13_batches_are_deterministic(self):
        rows = np.arange(9000, 9100)
        for name in ('hgt_full', 'hgt_shuffled_player', 'seq_mean'):
            cond = C['conditions'][name]
            make = (lambda: hgt_batch(self.val, rows, cond, C['relations'])) if cond['family'] == 'hgt' else (lambda: seq_batch(self.val, rows))
            self.assertTrue(equal_nested(make(), make()), name)


def equal_nested(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_nested(a[k], b[k]) for k in a)
    if isinstance(a, tuple):
        return len(a) == len(b) and all(equal_nested(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(a, b)
    return a == b


if __name__ == '__main__':
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FiveTaskTests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (ROOT / 'reports/acceptance.log').write_text(stream.getvalue())
    print(stream.getvalue())
    write_json(ROOT / 'reports/acceptance.json', dict(status='passed' if result.wasSuccessful() else 'failed', tests_run=result.testsRun,
                                                     failures=len(result.failures), errors=len(result.errors), test_evaluated=False))
    sys.exit(0 if result.wasSuccessful() else 1)
