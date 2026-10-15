"""Wait for the five-task queue, then LR search, LR lock, formal runs and the report; stops on the first failure.

--resume-formal FILE continues an interrupted formal stage: FILE lists the formal runs an earlier
queue process launched and that are still running (run_id, pid, gpu). They hold their GPU slots
until their processes exit and must end with status complete; the remaining formal runs are
launched with the locked learning rate, then the report is written.
"""
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


def alive(pid):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
    except FileNotFoundError:
        return False
    return state != 'Z'


def run_jobs(config, jobs, stage, adopted=()):
    root = Path(config['root'])
    slots = [g for g in config['execution']['gpus'] for _ in range(config['execution']['jobs_per_gpu'])]
    pending, active, failed, adopted = list(jobs), [], [], [dict(a) for a in adopted]
    while pending or active or adopted:
        while pending and len(active) + len(adopted) < len(slots) and not failed:
            busy = [a[1] for a in active] + [a['gpu'] for a in adopted]
            gpu = next(g for g in slots if busy.count(g) < slots.count(g))
            run_id, condition, seed, lr, mode = pending.pop(0)
            if (root / 'runs' / run_id).exists():
                raise FileExistsError(run_id)
            command = [config['execution']['python'], '-B', str(root / 'src/scr_fit.py'), '--config', str(root / 'configs/experiment.json'),
                       '--condition', condition, '--seed', str(seed), '--lr', str(lr), '--mode', mode, '--run-id', run_id, '--device', f'cuda:{gpu}']
            handle = (root / 'reports' / f'{run_id}.log').open('x')
            active.append((run_id, gpu, subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=root), handle))
            log(config, f'start {stage} {run_id} gpu{gpu}')
        status(config, status='running', stage=stage, active=[a[0] for a in active] + [a['run_id'] for a in adopted], pending=len(pending), failed=failed)
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
        for item in list(adopted):
            if not alive(item['pid']):
                adopted.remove(item)
                final = json.load(open(root / 'runs' / item['run_id'] / 'status.json'))['status']
                log(config, f"finish {item['run_id']} (adopted) status {final}")
                if final != 'complete':
                    failed.append(dict(run_id=item['run_id'], status=final))
        if failed and not active and not adopted:
            break
    if failed:
        status(config, status='failed', stage=stage, failed=failed)
        raise RuntimeError(f'Failed runs: {failed}')


def best_monitor(run, monitor):
    rows = [json.loads(line) for line in (run / 'curve.jsonl').open()]
    return min((r['validation'][monitor], r['epoch']) for r in rows if 'validation' in r and r['epoch'] % 4 == 0)


def resume_formal(config, plan):
    """Adopt the formal runs listed in `plan`, launch the remaining ones with the locked learning rate, then report."""
    reports, runs = Path(config['paths']['reports']), Path(config['paths']['runs'])
    if json.load(open(reports / 'acceptance.json'))['status'] != 'passed':
        raise ValueError('Acceptance has not passed')
    if (reports / 'queue_status.json').exists():
        raise FileExistsError('queue_status.json')
    if alive(plan['interrupted_queue_pid']):
        raise RuntimeError('The interrupted queue process is still alive')
    locked = json.load(open(reports / 'selected_learning_rate.json'))
    if locked['status'] != 'locked' or locked['selected'] != plan['learning_rate']:
        raise ValueError('Learning rate is not the locked one')
    lr = locked['selected']
    for item in plan['adopted']:
        if not alive(item['pid']):
            raise ValueError(f"Process {item['pid']} of {item['run_id']} is not running")
        args = Path(f"/proc/{item['pid']}/cmdline").read_bytes().decode().split('\0')
        if args[args.index('--run-id') + 1] != item['run_id'] or args[args.index('--device') + 1] != f"cuda:{item['gpu']}":
            raise ValueError(f"Process {item['pid']} is not {item['run_id']} on gpu {item['gpu']}")
        if json.load(open(runs / item['run_id'] / 'status.json'))['status'] != 'running':
            raise ValueError(f"{item['run_id']} is not running")
    adopted_ids = {item['run_id'] for item in plan['adopted']}
    formal = [(f'formal_{c}_s{seed}', c, seed, lr, 'formal') for seed in config['training']['seeds'] for c in config['conditions']]
    remaining = [job for job in formal if job[0] not in adopted_ids]
    if len(remaining) + len(adopted_ids) != len(formal):
        raise ValueError('Adopted runs are not formal runs of this configuration')
    status(config, status='resuming', stage='formal')
    log(config, f"resume formal stage: adopt {len(adopted_ids)} runs of queue pid {plan['interrupted_queue_pid']}, launch {len(remaining)} with lr {lr}")
    run_jobs(config, remaining, 'formal', plan['adopted'])
    status(config, status='reporting')
    subprocess.run([config['execution']['python'], '-B', str(Path(config['root']) / 'src/scr_report.py'), '--config',
                    str(Path(config['root']) / 'configs/experiment.json')], check=True, cwd=config['root'])
    status(config, status='complete')
    log(config, 'queue complete')


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
    parser.add_argument('--resume-formal')
    args = parser.parse_args()
    config = json.load(open(args.config))
    try:
        if args.resume_formal:
            resume_formal(config, json.load(open(args.resume_formal)))
        else:
            main(config)
    except Exception as error:
        write_json(Path(config['paths']['reports']) / 'queue_error.json', dict(status='failed', error=str(error), pid=os.getpid()))
        raise
