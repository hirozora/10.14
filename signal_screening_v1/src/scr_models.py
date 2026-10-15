"""Context encoder plus a candidate encoder whose inputs grow by condition (M0-M3).

The context is identity-free except under context_identity (M0_ctx_identity), where each window
unit token adds the static candidate states of its participants: one shared player embedding.
Scorer 'mlp' is MLP([context, state]); 'mlp_plus_dot' adds (W context) . state / sqrt(hidden),
with W initialised at zero (stage 2, M0_ctx_identity_dot).
Stage 3: scope 'match' widens the summary (8 inputs) and the history event features (2 more inputs);
pairs 'anchor' adds a linear map of the candidate's pair values with the anchor actor, and pairs 'window'
also mean-pools one relation token per other window actor. Their output layers start at zero.
"""
import copy
import math
import sys

import torch
from torch import nn
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, inputs, outputs, dropout):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, outputs), nn.LayerNorm(outputs), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)


class ContextEncoder(nn.Module):
    """Two Pre-LN attention layers over the window; anchor and window-mean readout.

    `people` (B, W, hidden) is added to the unit tokens only under context_identity.
    """

    def __init__(self, config, vocabulary):
        super().__init__()
        if config['parent_src'] not in sys.path:
            sys.path.insert(0, config['parent_src'])
        from model import Layer
        b = config['context']['backbone']
        h, d = b['hidden'], b['dropout']
        self.kind = nn.Embedding(10, 16)
        self.sub = nn.Embedding(vocabulary['subtype_codes'], 16, padding_idx=0)
        self.period = nn.Embedding(6, 4, padding_idx=0)
        self.project = Projection(16 + 16 + 4 + 7, h, d)
        self.team_relation = nn.Embedding(3, h, padding_idx=0)
        self.phase = nn.Linear(3, h, bias=False)
        self.order = nn.Embedding(81, h, padding_idx=0)
        spec = dict(hidden=h, heads=b['heads'], ffn=b['ffn'], dropout=d, time_bias=dict(mlp_hidden=b['time_bias_mlp_hidden']))
        self.layers = nn.ModuleList([Layer(spec) for _ in range(b['layers'])])
        self.norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(d)
        self.readout = Projection(2 * h, h, d)

    def forward(self, x, people=None):
        valid = x['valid']
        tokens = self.project(torch.cat((self.kind(x['kind']), self.sub(x['sub']), self.period(x['period']), x['numeric']), -1))
        tokens = tokens + self.team_relation(x['team_relation']).sum(2) + self.phase(x['phase']) + self.order(x['order'])
        if people is not None:
            tokens = tokens + people
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
        for layer in self.layers:
            h = layer(h, channels, allowed, valid)
        anchor = self.norm(h[:, -1])
        pooled = self.norm((h * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1))
        return self.readout(torch.cat((anchor, pooled), -1))


class Screen(nn.Module):
    def __init__(self, config, condition, vocabulary):
        super().__init__()
        b = config['context']['backbone']
        h, d = b['hidden'], b['dropout']
        self.condition = condition
        self.context = ContextEncoder(config, vocabulary)
        self.player_vocab = nn.Embedding(vocabulary['players'], 24)
        self.role = nn.Embedding(5, 4)
        self.foot = nn.Embedding(4, 4)
        self.static = Projection(24 + 4 + 4 + 4 + 1, h, d)
        if condition['scope'] not in ('window', 'match'):
            raise ValueError(f"Unknown scope {condition['scope']}")
        match_scope = condition['scope'] == 'match'
        if condition['summary']:
            self.summary = nn.Linear(5 + 3 * match_scope, h)
        if condition['history'] != 'none':
            self.h_kind = nn.Embedding(10, 16)
            self.h_sub = nn.Embedding(vocabulary['subtype_codes'], 16, padding_idx=0)
            self.h_team = nn.Embedding(4, 4, padding_idx=0)
            self.h_event = Projection(16 + 16 + 4 + 6 + 2 * match_scope, h, d)
            if condition['history'] == 'candidate_attention':
                self.heads = b['heads']
                self.query = nn.Linear(2 * h, h)
                self.key = nn.Linear(h, h)
                self.value = nn.Linear(h, h)
                self.attend_out = nn.Linear(h, h)
            else:
                self.pool_out = nn.Linear(2 * h, h)
        self.scorer = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(d), nn.Linear(h, 1))
        if condition['scorer'] == 'mlp_plus_dot':
            # Created last and zeroed: the other modules initialise as without it, and so do the scores.
            self.dot = nn.Linear(h, h, bias=False)
            nn.init.zeros_(self.dot.weight)
        elif condition['scorer'] != 'mlp':
            raise ValueError(f"Unknown scorer {condition['scorer']}")
        if condition['pairs'] not in ('none', 'anchor', 'window', 'window_shuffled_within_team'):
            raise ValueError(f"Unknown pairs {condition['pairs']}")
        if condition['pairs'] != 'none':
            # Created last with zeroed outputs: the other modules initialise as without them, and so do the scores.
            self.pair_anchor = nn.Linear(4, h)
            nn.init.zeros_(self.pair_anchor.weight)
            nn.init.zeros_(self.pair_anchor.bias)
            if condition['pairs'] != 'anchor':
                self.pair_token = Projection(8, h, d)
                self.pair_out = nn.Linear(h, h)
                nn.init.zeros_(self.pair_out.weight)
                nn.init.zeros_(self.pair_out.bias)

    def static_states(self, c):
        return self.static(torch.cat((self.player_vocab(c['vocab']), self.role(c['role']), self.foot(c['foot']), c['numeric'],
                                      c['same_team'][..., None]), -1))

    def candidate_states(self, c, context, static):
        state = static
        if self.condition['summary']:
            state = state + self.summary(c['summary'])
        history = self.condition['history']
        if history != 'none':
            events = self.h_event(torch.cat((self.h_kind(c['h_kind']), self.h_sub(c['h_sub']), self.h_team(c['h_team']), c['h_numeric']), -1))
            mask = c['h_mask']
            any_event = mask.any(-1, keepdim=True)
            if history == 'candidate_attention':
                B, S, K, H = events.shape
                heads, width = self.heads, H // self.heads
                q = self.query(torch.cat((context[:, None, :].expand(-1, S, -1), static), -1)).view(B, S, heads, width)
                k = self.key(events).view(B, S, K, heads, width)
                v = self.value(events).view(B, S, K, heads, width)
                score = torch.einsum('bshw,bskhw->bshk', q, k) / math.sqrt(width)
                # A finite fill keeps rows without events NaN-free; their reads are zeroed below.
                score = score.masked_fill(~mask[:, :, None, :], -1e9)
                weight = score.softmax(-1) * mask[:, :, None, :].float()
                read = torch.einsum('bshk,bskhw->bshw', weight, v).reshape(B, S, H)
                state = state + self.attend_out(read) * any_event
            else:
                m = mask[..., None].float()
                pooled = (events * m).sum(2) / m.sum(2).clamp_min(1)
                last = events[:, :, 0] * mask[:, :, :1].float()
                state = state + self.pool_out(torch.cat((pooled, last), -1)) * any_event
        if self.condition['pairs'] != 'none':
            state = state + self.pair_anchor(c['pair_anchor'])
            if self.condition['pairs'] != 'anchor':
                tokens = self.pair_token(c['pair_tokens'])
                m = c['pair_mask'][..., None].float()
                pooled = (tokens * m).sum(2) / m.sum(2).clamp_min(1)
                state = state + self.pair_out(pooled) * c['pair_mask'].any(-1, keepdim=True)
        return state

    def forward(self, batch):
        c = batch['candidates']
        if self.condition['context_identity']:
            # Every roster slot's static state, added to the tokens of the units it took part in.
            static = self.static_states(c)
            context = self.context(batch['context'], torch.einsum('bws,bsh->bwh', batch['context']['member'].float(), static))
        else:
            context = self.context(batch['context'])
            static = self.static_states(c)
        state = self.candidate_states(c, context, static)
        S = state.shape[1]
        scores = self.scorer(torch.cat((context[:, None, :].expand(-1, S, -1), state), -1)).squeeze(-1)
        if self.condition['scorer'] == 'mlp_plus_dot':
            scores = scores + torch.einsum('bh,bsh->bs', self.dot(context), state) / math.sqrt(state.shape[-1])
        return scores.masked_fill(~c['mask'], float('-inf'))


def player_loss(scores, y):
    logp = F.log_softmax(scores, -1)
    slot = y['slot'].long()
    picked = torch.gather(logp, 1, slot.clamp_min(0))
    use = (slot >= 0) & (y['loss_weight'] > 0)
    per = -(torch.where(use, picked, 0.0) * y['loss_weight']).sum(-1)
    mask = y['known'] & y['reachable']
    return per, mask
