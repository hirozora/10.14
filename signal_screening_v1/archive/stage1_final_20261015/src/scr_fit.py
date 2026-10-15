"""One fresh screening run (Player only): explicit condition, seed, learning rate and mode; no test split."""
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
from scr_candidates import setup
from scr_batches import ScreenData, screen_batch
from scr_models import Screen, player_loss
from scr_scoring import sample_statistics, match_table, metrics, json_metrics

ALLOWED_CPUS = os.sched_getaffinity(0)


def restore_affinity(_):
    # Loader workers otherwise inherit an OpenMP binding to a single core.
    os.sched_setaffinity(0, ALLOWED_CPUS)


def to_torch(value):
    if isinstance(value, dict):
        return {k: to_torch(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(to_torch(v) for v in value)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(value))
    return value


def move(value, device):
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    return value


class Batches(Dataset):
    def __init__(self, data, order, condition):
        self.data, self.order, self.condition = data, order, condition

    def __len__(self):
        return len(self.order)

    def __getitem__(self, j):
        return to_torch(screen_batch(self.data, self.order[j], self.condition))


def loader(data, order, condition, config):
    t = config['training']
    return DataLoader(Batches(data, order, condition), batch_size=None, shuffle=False, num_workers=t['loader_workers'],
                      prefetch_factor=t['prefetch_factor'], persistent_workers=False, worker_init_fn=restore_affinity)


@torch.no_grad()
def evaluate(model, data, condition, config, device):
    model.eval()
    n = len(data.base.targets['anchor'])
    order = [np.arange(lo, min(lo + 256, n)) for lo in range(0, n, 256)]
    stats, targets = [], []
    start = time.perf_counter()
    for x, y in loader(data, order, condition, config):
        scores = model(move(x, device))
        stats.append(sample_statistics(scores, move(y, device)))
        targets.append({k: v.numpy() for k, v in y.items()})
    model.train()
    stats = {k: np.concatenate([s[k] for s in stats]) for k in stats[0]}
    y = {k: np.concatenate([t[k] for t in targets]) for k in targets[0]}
    matches, table = match_table(stats, y)
    return stats, y, matches, table, json_metrics(metrics(table.sum(0))), time.perf_counter() - start


def fit(config, condition_id, seed, lr, device, mode, run_id):
    setup(config)
    from units import read_json, write_json
    from train import deterministic, rotate_orders, Stopper
    root = Path(config['paths']['runs']) / run_id
    if read_json(Path(config['paths']['reports']) / 'acceptance.json')['status'] != 'passed':
        raise ValueError('Acceptance has not passed')
    root.mkdir(exist_ok=False)
    deterministic(seed)
    torch.cuda.set_device(device)
    condition = config['conditions'][condition_id]
    vocabulary = read_json(Path(config['paths']['data']) / 'vocabulary.json')
    train, validation = ScreenData(config, 'train'), ScreenData(config, 'validation')
    model = Screen(config, condition, vocabulary).to(device)
    t = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=t['weight_decay'])
    epochs = config['lr_search']['epochs'] if mode == 'lr_search' else t['max_epochs']
    orders = rotate_orders(train.base.targets, seed, epochs)
    if orders.shape != (epochs, t['targets_per_epoch']):
        raise ValueError('Rotate count differs')
    np.save(root / 'sample_order.npy', orders)
    write_json(root / 'config.json', dict(protocol=config, condition_id=condition_id, condition=condition, seed=seed, lr=lr, mode=mode, device=device, epochs=epochs))
    write_json(root / 'status.json', dict(status='running', pid=os.getpid()))
    stopper = Stopper(t)
    best, best_epoch, steps, start = float('inf'), 0, 0, time.perf_counter()
    stop_reason = 'calibration_budget' if mode == 'lr_search' else 'budget_exhausted'
    for epoch in range(1, epochs + 1):
        model.train()
        values = []
        epoch_start, epoch_lr = time.perf_counter(), optimizer.param_groups[0]['lr']
        for x, y in loader(train, orders[epoch - 1].reshape(-1, t['batch_size']), condition, config):
            x, y = move(x, device), move(y, device)
            optimizer.zero_grad(set_to_none=True)
            per, mask = player_loss(model(x), y)
            loss = (per * mask).sum() / mask.sum().clamp_min(1)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            values.append(float(loss.detach()))
            steps += 1
        row = dict(epoch=epoch, optimizer_steps=steps, train_loss=float(np.mean(values)), lr=epoch_lr, train_seconds=time.perf_counter() - epoch_start)
        action = 'continue'
        scheduled = epoch % t['validation_every'] == 0
        if scheduled or epoch in t['curve_only_validation_epochs']:
            stats, y, matches, table, m, seconds = evaluate(model, validation, condition, config, device)
            row.update(validation=m, validation_seconds=seconds)
            value = m[t['monitor']]
            if scheduled and value < best:
                best, best_epoch = value, epoch
                torch.save(dict(model=model.state_dict(), epoch=epoch, monitor=best), root / 'best.pt')
                np.savez(root / 'best_validation_statistics.npz', **stats, **{f'target_{k}': v for k, v in y.items()})
                np.savez(root / 'best_match_statistics.npz', match_id=matches, table=table)
                write_json(root / 'best_metrics.json', dict(epoch=epoch, metrics=m))
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
        write_json(root / 'status.json', dict(status='running', pid=os.getpid(), epoch=epoch, best_epoch=best_epoch, best_monitor=None if best_epoch == 0 else best))
        print(run_id, 'epoch', epoch, 'train', round(row['train_loss'], 5), 'best', best, 'seconds', round(row['elapsed_seconds'], 1), flush=True)
        if action == 'stop':
            break
    write_json(root / 'cost.json', dict(total_parameters=sum(p.numel() for p in model.parameters()), peak_gpu_bytes=torch.cuda.max_memory_allocated(device),
                                        training_wall_seconds=time.perf_counter() - start))
    write_json(root / 'status.json', dict(status='complete', stop_reason=stop_reason, epochs=epoch, best_epoch=best_epoch, best_monitor=best, optimizer_steps=steps))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for field in ('config', 'condition', 'mode', 'run-id', 'device'):
        p.add_argument('--' + field, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--lr', type=float, required=True)
    a = p.parse_args()
    c = json.load(open(a.config))
    if a.condition not in c['conditions'] or a.mode not in ('lr_search', 'formal'):
        raise ValueError('Unknown condition or mode')
    fit(c, a.condition, a.seed, a.lr, a.device, a.mode, a.run_id)
