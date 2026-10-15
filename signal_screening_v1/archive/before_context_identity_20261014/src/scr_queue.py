"""Wait for the five-task queue, then LR search, LR lock, formal runs and the report; stops on the first failure."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def log(config, message):
    with (Path(config['paths']['reports']) / 'queue.log').open('a') as f:
        f.write(time.strftime('%Y-%m-%d %H:%M:%S ') + message + '\n')


def status(config, **value):
    write_json(Path(config['paths']['reports']) / 'queue_status.json', dict(value, pid=os.getpid(), test_evaluated=False))


def run_jobs(config, jobs, stage):
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
            command = [config['execution']['python'], '-B', str(root / 'src/scr_fit.py'), '--config', str(root / 'configs/experiment.json'),
                       '--condition', condition, '--seed', str(seed), '--lr', str(lr), '--mode', mode, '--run-id', run_id, '--device', f'cuda:{gpu}']
            handle = (root / 'reports' / f'{run_id}.log').open('x')
            active.append((run_id, gpu, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=root), handle))
            log(config, f'start {stage} {run_id} gpu{gpu}')
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


def best_monitor(run, monitor):
    rows = [json.loads(line) for line in (run / 'curve.jsonl').open()]
    return min((r['validation'][monitor], r['epoch']) for r in rows if 'validation' in r and r['epoch'] % 4 == 0)


def main(config):
    reports = Path(config['paths']['reports'])
    if json.load(open(reports / 'acceptance.json'))['status'] != 'passed':
        raise ValueError('Acceptance has not passed')
    if (reports / 'queue_status.json').exists():
        raise FileExistsError('queue_status.json')
    status(config, status='waiting_for_five_task_queue')
    log(config, 'waiting for the five-task queue')
    while json.load(open(config['execution']['wait_for']))['status'] != 'complete':
        if json.load(open(config['execution']['wait_for']))['status'] == 'failed':
            raise RuntimeError('Five-task queue failed; screening not started')
        time.sleep(60)
    log(config, 'five-task queue complete; starting')
    s = config['lr_search']
    run_jobs(config, [(f'lr_{c}_lr{lr}', c, s['seed'], lr, 'lr_search') for c in s['references'] for lr in s['grid']], 'lr_search')
    runs, monitor = Path(config['paths']['runs']), config['training']['monitor']
    scores = {lr: [best_monitor(runs / f'lr_{c}_lr{lr}', monitor)[0] for c in s['references']] for lr in s['grid']}
    lr = min(s['grid'], key=lambda x: (sum(scores[x]) / len(scores[x]), x))
    write_json(reports / 'selected_learning_rate.json', dict(status='locked', selected=lr, at_grid_edge=lr in (min(s['grid']), max(s['grid'])),
                                                             scores={str(k): v for k, v in scores.items()}, rule=s['rule']))
    log(config, f'learning rate locked {lr}')
    formal = [(f'formal_{c}_s{seed}', c, seed, lr, 'formal') for seed in config['training']['seeds'] for c in config['conditions']]
    run_jobs(config, formal, 'formal')
    status(config, status='reporting')
    subprocess.run([config['execution']['python'], '-B', str(Path(config['root']) / 'src/scr_report.py'), '--config',
                    str(Path(config['root']) / 'configs/experiment.json')], check=True, cwd=config['root'])
    status(config, status='complete')
    log(config, 'queue complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    config = json.load(open(parser.parse_args().config))
    try:
        main(config)
    except Exception as error:
        write_json(Path(config['paths']['reports']) / 'queue_error.json', dict(status='failed', error=str(error), pid=os.getpid()))
        raise
