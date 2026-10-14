"""One fresh run: explicit condition, seed, learning rate and mode; no test split is loaded."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from units import read_json, write_json
from batches import SplitData, hgt_batch, seq_batch, to_torch
from models import build, objective_parts, five_task, backward
from scoring import sample_statistics, match_tables, metric_array, json_metrics


def parent_tools(config):
    if config['parent_src'] not in sys.path:
        sys.path.insert(0, config['parent_src'])
    from train import deterministic, rotate_orders, Stopper
    return deterministic, rotate_orders, Stopper


class Batches(Dataset):
    """Batch j of a fixed row order; built in loader workers from read-only arrays."""

    def __init__(self, data, order, condition, relations):
        self.data, self.order, self.condition, self.relations = data, order, condition, relations

    def __len__(self):
        return len(self.order)

    def __getitem__(self, j):
        rows = self.order[j]
        if self.condition['family'] == 'hgt':
            return to_torch(hgt_batch(self.data, rows, self.condition, self.relations))
        return to_torch(seq_batch(self.data, rows))


ALLOWED_CPUS = os.sched_getaffinity(0)


def restore_affinity(_):
    # Loader workers otherwise inherit an OpenMP binding to a single core.
    os.sched_setaffinity(0, ALLOWED_CPUS)


def loader(data, order, condition, config):
    t = config['training']
    return DataLoader(Batches(data, order, condition, config['relations']), batch_size=None, shuffle=False,
                      num_workers=t['loader_workers'], prefetch_factor=t['prefetch_factor'], persistent_workers=False,
                      worker_init_fn=restore_affinity)


def move(value, device):
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


def evaluate(model, data, condition, config, device):
    model.eval()
    n = len(data.targets['anchor'])
    order = [np.arange(lo, min(lo + 256, n)) for lo in range(0, n, 256)]
    stats = []
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in loader(data, order, condition, config):
            x, y = move(x, device), move(y, device)
            pred = model(x)
            for k, v in pred.items():
                if not torch.isfinite(v[v != float('-inf')]).all():
                    raise FloatingPointError('Nonfinite validation prediction: ' + k)
            stats.append(sample_statistics(pred, y))
    model.train()
    stats = {k: np.concatenate([s[k] for s in stats]) for k in stats[0]}
    y = {k: v for k, v in data.targets.items()}
    matches, tables = match_tables(stats, y)
    metrics = {view: json_metrics(metric_array(table.sum(0))) for view, table in tables.items()}
    return stats, matches, tables, metrics, time.perf_counter() - start


def fit(config, condition_id, seed, lr, device, mode, run_id):
    root = Path(config['paths']['runs']) / run_id
    if read_json(Path(config['paths']['reports']) / 'acceptance.json')['status'] != 'passed':
        raise ValueError('Acceptance has not passed')
    deterministic, rotate_orders, Stopper = parent_tools(config)
    root.mkdir(exist_ok=False)
    deterministic(seed)
    torch.cuda.set_device(device)
    condition = config['conditions'][condition_id]
    vocabulary = read_json(Path(config['paths']['data']) / 'vocabulary.json')
    train = SplitData(config, 'train')
    validation = SplitData(config, 'validation')
    model = build(config, condition, vocabulary).to(device)
    t = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t['weight_decay'])
    epochs = config['lr_search']['epochs'] if mode == 'lr_search' else t['max_epochs']
    orders = rotate_orders(train.targets, seed, epochs)
    if orders.shape != (epochs, t['targets_per_epoch']):
        raise ValueError('Rotate count differs')
    np.save(root / 'sample_order.npy', orders)
    write_json(root / 'config.json', dict(protocol=config, condition_id=condition_id, condition=condition, seed=seed, lr=lr, mode=mode,
                                         device=device, epochs=epochs))
    write_json(root / 'status.json', dict(status='running', pid=os.getpid()))
    stopper = Stopper(t)
    best, best_epoch, steps, start = float('inf'), 0, 0, time.perf_counter()
    total_parameters = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)
    stop_reason = 'calibration_budget' if mode == 'lr_search' else 'budget_exhausted'
    for epoch in range(1, epochs + 1):
        model.train()
        values, parts_sum = [], {}
        epoch_start, epoch_lr = time.perf_counter(), optimizer.param_groups[0]['lr']
        batches = orders[epoch - 1].reshape(-1, t['batch_size'])
        for x, y in loader(train, batches, condition, config):
            x, y = move(x, device), move(y, device)
            optimizer.zero_grad(set_to_none=True)
            parts = objective_parts(model(x), y)
            total = five_task(parts)
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite training loss')
            backward(model, parts)
            torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            values.append(float(total.detach()))
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + float(v.detach())
            steps += 1
        row = dict(epoch=epoch, optimizer_steps=steps, cumulative_targets=epoch * t['targets_per_epoch'],
                   full_train_equivalents=epoch * t['targets_per_epoch'] / len(train.targets['anchor']), train_loss=float(np.mean(values)),
                   train_parts={k: v / len(values) for k, v in parts_sum.items()}, lr=epoch_lr, train_seconds=time.perf_counter() - epoch_start)
        action = 'continue'
        scheduled = epoch % t['validation_every'] == 0
        if scheduled or epoch in t['curve_only_validation_epochs']:
            stats, matches, tables, metrics, seconds = evaluate(model, validation, condition, config, device)
            row['validation'] = metrics
            row['validation_seconds'] = seconds
            value = metrics['all'][t['monitor']]
            if scheduled and value < best:
                best, best_epoch = value, epoch
                torch.save(dict(model=model.state_dict(), epoch=epoch, monitor=best), root / 'best.pt')
                np.savez(root / 'best_validation_statistics.npz', **stats)
                np.savez(root / 'best_match_statistics.npz', match_id=matches, **tables)
                write_json(root / 'best_metrics.json', dict(epoch=epoch, metrics=metrics))
            if scheduled and mode != 'lr_search':
                action = stopper.step(epoch, value)
                if action == 'decay':
                    for group in optimizer.param_groups:
                        group['lr'] *= t['lr_decay_factor']
                elif action == 'stop':
                    stop_reason = 'plateau_after_two_decays'
        row.update(next_lr=optimizer.param_groups[0]['lr'], scheduler_action=action, elapsed_seconds=time.perf_counter() - start)
        with (root / 'curve.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch, seed=seed), root / 'last.pt')
        write_json(root / 'status.json', dict(status='running', pid=os.getpid(), epoch=epoch, best_epoch=best_epoch,
                                              best_monitor=None if best_epoch == 0 else best))
        print(run_id, 'epoch', epoch, 'train', round(row['train_loss'], 6), 'best', best, 'seconds', round(row['elapsed_seconds'], 1), flush=True)
        if action == 'stop':
            break
    write_json(root / 'cost.json', dict(total_parameters=total_parameters, peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
                                        training_wall_seconds=time.perf_counter() - start))
    write_json(root / 'status.json', dict(status='complete', stop_reason=stop_reason, epochs=epoch, best_epoch=best_epoch, best_monitor=best,
                                          optimizer_steps=steps, cumulative_targets=epoch * t['targets_per_epoch']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for field in ('config', 'condition', 'mode', 'run-id', 'device'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--lr', type=float, required=True)
    a = p.parse_args()
    c = read_json(a.config)
    if a.condition not in c['conditions'] or a.mode not in ('lr_search', 'formal'):
        raise ValueError('Unknown condition or mode')
    fit(c, a.condition, a.seed, a.lr, a.device, a.mode, a.run_id)
