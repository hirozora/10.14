"""LR search, LR lock, formal runs (all conditions x 3 seeds) and the report; stops on the first failed job."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from units import read_json, write_json


def log(config, message):
    with (Path(config['paths']['reports']) / 'queue.log').open('a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + message + '\n')


def status(config, **value):
    write_json(Path(config['paths']['reports']) / 'queue_status.json', dict(value, pid=os.getpid(), test_evaluated=False))


def run_jobs(config, jobs, stage):
    """jobs: list of (run_id, condition, seed, lr, mode). Two processes per GPU; no retries."""
    root = Path(config['root'])
    slots = [g for g in config['execution']['gpus'] for _ in range(config['execution']['jobs_per_gpu'])]
    pending, active, failed = list(jobs), [], []
    while pending or active:
        while pending and len(active) < len(slots) and not failed:
            busy = [a[1] for a in active]
            gpu = next(g for g in slots if busy.count(g) < slots.count(g))
            run_id, condition, seed, lr, mode = pending.pop(0)
            if (root / 'runs' / run_id).exists():
                raise FileExistsError(run_id)
            command = [config['execution']['python'], '-B', str(root / 'src/fit.py'), '--config', str(root / 'configs/experiment.json'),
                       '--condition', condition, '--seed', str(seed), '--lr', str(lr), '--mode', mode, '--run-id', run_id, '--device', f'cuda:{gpu}']
            handle = (root / 'reports' / f'{run_id}.log').open('x')
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=root)
            active.append((run_id, gpu, process, handle))
            log(config, f'start {stage} {run_id} gpu{gpu} pid{process.pid}')
        status(config, status='running', stage=stage, active=[a[0] for a in active], pending=len(pending), failed=failed)
        time.sleep(10)
        for item in list(active):
            run_id, gpu, process, handle = item
            code = process.poll()
            if code is not None:
                handle.close()
                active.remove(item)
                log(config, f'finish {run_id} exit {code}')
                if code:
                    failed.append(dict(run_id=run_id, exit_code=code))
        if failed and not active:
            break
    if failed:
        status(config, status='failed', stage=stage, failed=failed)
        raise RuntimeError(f'Failed runs: {failed}')


def best_monitor(run):
    rows = [json.loads(line) for line in (run / 'curve.jsonl').open()]
    values = [(r['validation']['all']['five_task_loss'], r['epoch']) for r in rows if 'validation' in r and r['epoch'] % 4 == 0]
    return min(values)


def select_rates(config):
    runs = Path(config['paths']['runs'])
    grid = config['lr_search']['grid']
    selected, scores = {}, {}
    for family, condition in config['lr_search']['references'].items():
        scores[family] = [dict(lr=lr, best=best_monitor(runs / f'lr_{condition}_lr{lr}')) for lr in grid]
        value = min((s['best'][0], s['lr']) for s in scores[family])
        selected[family] = value[1]
    edge = {f: lr in (min(grid), max(grid)) for f, lr in selected.items()}
    write_json(Path(config['paths']['reports']) / 'selected_learning_rates.json',
               dict(status='locked', selected=selected, at_grid_edge=edge, scores=scores, rule=config['lr_search']['rule']))
    return selected


def main(config):
    reports = Path(config['paths']['reports'])
    if read_json(reports / 'acceptance.json')['status'] != 'passed':
        raise ValueError('Acceptance has not passed')
    if (reports / 'queue_status.json').exists():
        raise FileExistsError('queue_status.json')
    log(config, 'queue started')
    seed = config['lr_search']['seed']
    search = [(f'lr_{c}_lr{lr}', c, seed, lr, 'lr_search') for c in config['lr_search']['references'].values() for lr in config['lr_search']['grid']]
    run_jobs(config, search, 'lr_search')
    rates = select_rates(config)
    log(config, f'learning rates locked {rates}')
    seeds = config['training']['seeds']
    formal = [(f'formal_{c}_s{s}', c, s, rates[cond['family']], 'formal') for s in seeds for c, cond in config['conditions'].items()]
    run_jobs(config, formal, 'formal')
    status(config, status='reporting')
    subprocess.run([config['execution']['python'], '-B', str(Path(config['root']) / 'src/report.py'), '--config',
                    str(Path(config['root']) / 'configs/experiment.json')], check=True, cwd=config['root'])
    status(config, status='complete', stage='report')
    log(config, 'queue complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    config = read_json(parser.parse_args().config)
    try:
        main(config)
    except Exception as error:
        write_json(Path(config['paths']['reports']) / 'queue_error.json', dict(status='failed', error=str(error), pid=os.getpid()))
        raise
