"""Five-task HGT (port of the CoHiG representation stage) and its sequence counterpart."""
import copy
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import HGTConv
from torch_geometric.utils import softmax


class FastHGTConv(nn.Module):
    """PyG 2.8 HGTConv mathematics on its own parameters, with one batched transform per relation.

    PyG's HeteroLinear falls back to one slice per (head, relation) pair; this forward computes
    the same keys, values, priors, softmax over all incoming edges, GELU output layer and
    sigmoid skip gate without those slices.
    """

    def __init__(self, hidden, metadata, heads):
        super().__init__()
        self.conv = HGTConv(hidden, hidden, metadata, heads=heads)
        self.heads = heads
        self.dim = hidden // heads
        self.relation_index = {tuple(e): i for i, e in enumerate(metadata[1])}

    def forward(self, x_dict, edge_index_dict):
        c, H, D = self.conv, self.heads, self.dim
        R = len(self.relation_index)
        k, q, v, offset = {}, {}, {}, {}
        total = 0
        for t, x in x_dict.items():
            lin = c.kqv_lin.lins[t]
            k[t], q[t], v[t] = F.linear(x, lin.weight, lin.bias).view(-1, 3, H, D).unbind(1)
            offset[t] = total
            total += x.shape[0]
        device = next(iter(x_dict.values())).device
        heads = torch.arange(H, device=device) * R
        by_source = {}
        for edge_type, index in edge_index_dict.items():
            by_source.setdefault(edge_type[0], []).append((edge_type, index))
        keys, values, targets, priors = [], [], [], []
        for s, items in by_source.items():
            # All relations leaving node type s in one (H, D, R_s*D) transform for keys and one for values.
            n, m = k[s].shape[0], len(items)
            chosen = (heads[:, None] + torch.tensor([self.relation_index[e] for e, _ in items], device=device)[None]).flatten()
            wk = c.k_rel.weight[chosen].view(H, m, D, D).permute(0, 2, 1, 3).reshape(H, D, m * D)
            wv = c.v_rel.weight[chosen].view(H, m, D, D).permute(0, 2, 1, 3).reshape(H, D, m * D)
            ks = torch.bmm(k[s].transpose(0, 1), wk).view(H, n, m, D).permute(1, 2, 0, 3).reshape(n * m, H, D)
            vs = torch.bmm(v[s].transpose(0, 1), wv).view(H, n, m, D).permute(1, 2, 0, 3).reshape(n * m, H, D)
            flat = torch.cat([index[0] * m + j for j, (_, index) in enumerate(items)])
            keys.append(ks[flat])
            values.append(vs[flat])
            targets.append(torch.cat([index[1] + offset[e[2]] for e, index in items]))
            priors.append(torch.cat([c.p_rel['__'.join(e)].expand(index.shape[1], H) for e, index in items]))
        result = {}
        if keys:
            target = torch.cat(targets)
            query = torch.cat([q[t] for t in x_dict])[target]
            score = (query * torch.cat(keys)).sum(-1) * torch.cat(priors) / math.sqrt(D)
            message = torch.cat(values)
            weight = softmax(score, target, num_nodes=total)
            out = torch.zeros(total, H, D, device=score.device, dtype=score.dtype).index_add_(0, target, message * weight[..., None])
            out = out.view(total, H * D)
        else:
            out = torch.zeros(total, H * D, device=next(iter(x_dict.values())).device)
        for (t, x), o in zip(x_dict.items(), torch.split(out, [x.shape[0] for x in x_dict.values()])):
            if t not in c.dst_node_types:
                continue
            lin = c.out_lin.lins[t]
            alpha = c.skip[t].sigmoid()
            result[t] = alpha * F.linear(F.gelu(o), lin.weight, lin.bias) + (1 - alpha) * x
        return result


class Projection(nn.Module):
    def __init__(self, inputs, outputs, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, outputs), nn.LayerNorm(outputs), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)


def edge_types(condition, relations):
    return [tuple(e) for family in condition['relations'] for e in relations[family]]


def node_types(condition):
    families = set(condition['relations'])
    types = ['event', 'player']
    if families & {'team', 'member'}:
        types.append('team')
    if 'type' in families:
        types.append('event_type')
    if 'zone' in families:
        types.append('zone')
    if 'possession' in families:
        types.append('possession')
    return types


def segment_mean(values, index, count):
    total = torch.zeros(count, values.shape[1], device=values.device, dtype=values.dtype).index_add_(0, index, values)
    n = torch.zeros(count, device=values.device, dtype=values.dtype).index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return total / n.clamp_min(1)[:, None]


class Heads(nn.Module):
    """Event, Time, Position and Team from the main context; Player scores from the private context."""

    def __init__(self, hidden, dropout):
        super().__init__()
        self.event = nn.Linear(hidden, 10)
        self.time = nn.Linear(hidden, 1)
        self.position = nn.Linear(hidden, 2)
        self.team = nn.Linear(hidden, 2)
        self.player = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, context, player_context, player_states, player_batch, player_slot_index, batch_size, slots):
        log_delta = F.softplus(self.time(context).squeeze(-1))
        scores = self.player(torch.cat((player_context[player_batch], player_states), -1)).squeeze(-1)
        table = torch.full((batch_size * slots,), float('-inf'), device=scores.device, dtype=scores.dtype)
        table = table.index_copy(0, player_slot_index, scores).view(batch_size, slots)
        return dict(event_logits=self.event(context), time_seconds=torch.expm1(log_delta).clamp(max=60.0),
                    position_xy=self.position(context).sigmoid(), team_logits=self.team(context), player_scores=table)


class HGT5(nn.Module):
    family = 'hgt'

    def __init__(self, config, condition, vocabulary):
        super().__init__()
        m, h = config['model'], config['model']['hidden']
        d = m['dropout']
        self.condition = condition
        self.types = node_types(condition)
        self.edges = edge_types(condition, config['relations'])
        self.event_kind = nn.Embedding(10, 16)
        self.event_sub = nn.Embedding(vocabulary['subtype_codes'], 16, padding_idx=0)
        self.event_period = nn.Embedding(6, 4, padding_idx=0)
        self.event_projection = Projection(16 + 16 + 4 + 7, h, d)
        self.player_vocab = nn.Embedding(vocabulary['players'], 24)
        self.player_role = nn.Embedding(5, 4)
        self.player_foot = nn.Embedding(4, 4)
        self.player_projection = Projection(24 + 4 + 4 + 4, h, d)
        if 'team' in self.types:
            self.team_vocab = nn.Embedding(vocabulary['teams'], 16)
            self.team_side = nn.Embedding(3, 4, padding_idx=0)
            self.team_kind = nn.Embedding(3, 4)
            self.team_projection = Projection(16 + 4 + 4 + 1, h, d)
        if 'event_type' in self.types:
            self.type_node = nn.Embedding(10, 24)
            self.type_projection = Projection(24, h, d)
        if 'zone' in self.types:
            self.zone_node = nn.Embedding(20, 24)
            self.zone_projection = Projection(26, h, d)
        if 'possession' in self.types:
            self.possession_base = nn.Parameter(torch.empty(h))
            nn.init.normal_(self.possession_base, std=0.02)
            self.possession_projection = Projection(7, h, d)
        metadata = (self.types, self.edges)
        self.convolutions = nn.ModuleList([FastHGTConv(h, metadata, m['heads']) for _ in range(2)])
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(h) for t in self.types}) for _ in range(2)])
        self.dropout = nn.Dropout(d)
        self.context_projection = Projection(2 * h, h, d)
        self.heads = Heads(h, d)
        # Task division of the research representation: a private Player layer 2 copied from the main one.
        self.player_convolution = copy.deepcopy(self.convolutions[1])
        self.player_norms = copy.deepcopy(self.norms[1])
        self.player_context_projection = copy.deepcopy(self.context_projection)

    def shared_parameters(self):
        return [p for name, p in self.named_parameters() if name.startswith(('convolutions.0.', 'norms.0.'))]

    def encode(self, x):
        e = x['event']
        states = dict(event=self.event_projection(torch.cat((self.event_kind(e['kind']), self.event_sub(e['sub']),
                                                            self.event_period(e['period']), e['numeric']), -1)))
        p = x['player']
        states['player'] = self.player_projection(torch.cat((self.player_vocab(p['vocab']), self.player_role(p['role']),
                                                             self.player_foot(p['foot']), p['numeric']), -1))
        if 'team' in self.types:
            t = x['team']
            states['team'] = self.team_projection(torch.cat((self.team_vocab(t['vocab']), self.team_side(t['side']),
                                                             self.team_kind(t['kind']), t['is_ref']), -1))
        if 'event_type' in self.types:
            states['event_type'] = self.type_projection(self.type_node(x['event_type']['kind']))
        if 'zone' in self.types:
            z = x['zone']
            states['zone'] = self.zone_projection(torch.cat((self.zone_node(z['kind']), z['center']), -1))
        if 'possession' in self.types:
            numeric = x['possession']['numeric']
            states['possession'] = self.possession_base[None].expand(len(numeric), -1) + self.possession_projection(numeric)
        return states

    def apply(self, states, edges, convolution, norms):
        updated = convolution(states, edges) if edges else {}
        return {t: norms[t](s + self.dropout(updated[t])) if updated.get(t) is not None else s for t, s in states.items()}

    def readout(self, events, batch, projection):
        anchor = events[batch['anchor_event']]
        pooled = segment_mean(events, batch['event_batch'], batch['batch_size'])
        if self.condition['readout'] == 'anchor_only':
            pooled = torch.zeros_like(pooled)
        return projection(torch.cat((anchor, pooled), -1))

    def forward(self, batch):
        edges = {k: v for k, v in batch['edges'].items() if v.shape[1] > 0}
        shared = self.apply(self.encode(batch['x']), edges, self.convolutions[0], self.norms[0])
        main = self.apply(shared, edges, self.convolutions[1], self.norms[1])
        private = self.apply(shared, edges, self.player_convolution, self.player_norms)
        context = self.readout(main['event'], batch, self.context_projection)
        player_context = self.readout(private['event'], batch, self.player_context_projection)
        slots = batch['player_id'].shape[1]
        valid = batch['player_id'] >= 0
        slot_index = torch.nonzero(valid.flatten()).flatten()
        return self.heads(context, player_context, private['player'], batch['player_batch'], slot_index, batch['batch_size'], slots)


class Seq5(nn.Module):
    """Dense causal attention over the same units, participants and phase relations; static roster candidates."""
    family = 'seq'

    def __init__(self, config, condition, vocabulary):
        super().__init__()
        if config['parent_src'] not in sys.path:
            sys.path.insert(0, config['parent_src'])
        from model import Layer
        m, h = config['model'], config['model']['hidden']
        s = m['sequence_counterpart']
        d = m['dropout']
        self.condition = condition
        self.event_kind = nn.Embedding(10, 16)
        self.event_sub = nn.Embedding(vocabulary['subtype_codes'], 16, padding_idx=0)
        self.event_period = nn.Embedding(6, 4, padding_idx=0)
        self.event_projection = Projection(16 + 16 + 4 + 7, h, d)
        self.player_vocab = nn.Embedding(vocabulary['players'], 24)
        self.player_role = nn.Embedding(5, 4)
        self.player_foot = nn.Embedding(4, 4)
        self.player_projection = Projection(24 + 4 + 4 + 4, h, d)
        self.team_relation = nn.Embedding(3, h, padding_idx=0)
        self.phase = nn.Linear(3, h, bias=False)
        self.order = nn.Embedding(81, h, padding_idx=0)
        layer = dict(hidden=h, heads=m['heads'], ffn=s['ffn'], dropout=d, time_bias=dict(mlp_hidden=s['time_bias_mlp_hidden']))
        self.layers = nn.ModuleList([Layer(layer) for _ in range(2)])
        self.player_layer = copy.deepcopy(self.layers[1])
        self.norm = nn.LayerNorm(h)
        self.player_norm = copy.deepcopy(self.norm)
        self.drop = nn.Dropout(d)
        self.context_projection = Projection(2 * h, h, d)
        self.heads = Heads(h, d)
        self.player_context_projection = copy.deepcopy(self.context_projection)

    def shared_parameters(self):
        return list(self.layers[0].parameters())

    def player_state(self, p):
        return self.player_projection(torch.cat((self.player_vocab(p['vocab']), self.player_role(p['role']), self.player_foot(p['foot']),
                                                 p['numeric']), -1))

    def readout(self, states, valid, norm, projection):
        anchor = norm(states[:, -1])
        pooled = (states * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        if self.condition['readout'] == 'anchor_only':
            pooled = torch.zeros_like(pooled)
        return projection(torch.cat((anchor, norm(pooled)), -1))

    def forward(self, batch):
        x = batch['x']
        valid = x['valid']
        tokens = self.event_projection(torch.cat((self.event_kind(x['kind']), self.event_sub(x['sub']), self.event_period(x['period']),
                                                  x['numeric']), -1))
        participant = x['participant']
        people = self.player_state(participant) * participant['has'][..., None]
        tokens = tokens + people.sum(2) + self.team_relation(x['team_relation']).sum(2) + self.phase(x['phase_numeric']) + self.order(x['order'])
        n = valid.shape[1]
        index = torch.arange(n, device=valid.device)
        allowed = (index[:, None] >= index[None, :])[None] & valid[:, :, None] & valid[:, None, :]
        allowed = allowed | (~valid[:, :, None] & torch.eye(n, dtype=torch.bool, device=valid.device)[None])
        same_period = x['period_raw'][:, :, None] == x['period_raw'][:, None, :]
        dt_valid = same_period & allowed & valid[:, :, None]
        dn = (x['distance'][:, :, None] - x['distance'][:, None, :]).float() / 80
        dt = torch.where(dt_valid, (x['seconds'][:, :, None] - x['seconds'][:, None, :]) / 60, 0.0)
        zero = torch.zeros_like(dn)
        channels = torch.stack([dn, dt, dt_valid.float(), zero, zero, zero], -1)
        h = self.drop(tokens) * valid[..., None]
        shared = self.layers[0](h, channels, allowed, valid)
        main = self.layers[1](shared, channels, allowed, valid)
        private = self.player_layer(shared, channels, allowed, valid)
        context = self.readout(main, valid, self.norm, self.context_projection)
        player_context = self.readout(private, valid, self.player_norm, self.player_context_projection)
        candidates = x['candidates']
        cvalid = candidates['valid']
        states = self.player_state(candidates)
        B, slots = cvalid.shape
        batch_index = torch.arange(B, device=valid.device)[:, None].expand(B, slots)[cvalid]
        slot_index = torch.nonzero(cvalid.flatten()).flatten()
        return self.heads(context, player_context, states[cvalid], batch_index, slot_index, B, slots)


def build(config, condition, vocabulary):
    return (HGT5 if condition['family'] == 'hgt' else Seq5)(config, condition, vocabulary)


def task_parts(pred, y):
    """Per-sample losses; Team and Player use the soft targets of merged duels."""
    event = F.cross_entropy(pred['event_logits'], y['event'].long(), reduction='none')
    time = F.smooth_l1_loss(pred['time_seconds'] / 60, y['time'].float() / 60, beta=1 / 60, reduction='none')
    position = F.smooth_l1_loss(pred['position_xy'], y['xy'].float(), reduction='none').mean(-1)
    team_logp = F.log_softmax(pred['team_logits'], -1)
    q = torch.stack((1 - y['team_same'], y['team_same']), -1)
    team = -(q * team_logp).sum(-1)
    player_logp = F.log_softmax(pred['player_scores'], -1)
    slot = y['player_slot'].long()
    picked = torch.gather(player_logp, 1, slot.clamp_min(0))
    player = -(torch.where(slot >= 0, picked, 0.0) * y['player_weight']).sum(-1)
    return dict(event=event, time=time, position=position, team=team, player=player)


def masked_mean(value, mask):
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def objective_parts(pred, y):
    p = task_parts(pred, y)
    return dict(event=p['event'].mean() / math.log(10), time=masked_mean(p['time'], y['time_mask']),
                position=masked_mean(p['position'], y['pos_mask']), team=masked_mean(p['team'], y['team_mask']),
                player=masked_mean(p['player'], y['player_mask']))


def five_task(parts):
    return (0.2 * parts['event'] + parts['time'] + parts['position'] + 0.05 * parts['team'] + 0.4 * parts['player']) / 3


def backward(model, parts):
    """Player gradients reach shared layer 1 scaled by 0.25; everything else at full weight."""
    hooks = [p.register_hook(lambda grad: grad * 0.25) for p in model.shared_parameters()]
    try:
        (0.4 * parts['player'] / 3).backward(retain_graph=True)
    finally:
        for hook in hooks:
            hook.remove()
    ((0.2 * parts['event'] + parts['time'] + parts['position'] + 0.05 * parts['team']) / 3).backward()
