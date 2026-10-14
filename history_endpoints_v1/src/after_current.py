"""Wait for the explicit parent queue, then run exactly three endpoint fits."""
import argparse
import os
import subprocess
import time
from pathlib import Path
from endpoints import read_config,write_json


def process_running(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():return False
    # Zombie means the process has exited; no parent job is still being dispatched.
    return path.read_text().rsplit(')',1)[1].split()[0]!='Z'


def main(config):
    root=Path(config['root']);reports=root/'reports';ext=config['extension'];status=reports/'queue_status.json'
    if status.exists():raise FileExistsError(status)
    if read_config(reports/'acceptance.json')['status']!='passed':raise ValueError('Endpoint acceptance failed')
    write_json(status,dict(status='waiting_for_parent',pid=os.getpid(),parent_pid=ext['base_queue_pid'],condition=ext['base_condition'],seeds=ext['seeds']))
    print('waiting for parent',ext['base_queue_pid'],flush=True)
    while process_running(ext['base_queue_pid']):time.sleep(ext['poll_seconds'])
    parent=read_config(ext['wait_for'])
    if parent['status']!='complete':raise RuntimeError('Parent queue did not complete successfully; endpoint jobs not started')
    jobs=[]
    for gpu,seed in zip(ext['gpus'],ext['seeds'],strict=True):
        run_id=f'endpoints_etp_s{seed}'
        if (root/'runs'/run_id).exists():raise FileExistsError(run_id)
        command=[config['execution']['python'],str(root/'src/train_endpoints.py'),'--config',str(root/'configs/experiments.json'),'--seed',str(seed),'--device',f'cuda:{gpu}']
        handle=(reports/f'{run_id}.log').open('x');p=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,cwd=root)
        jobs.append((seed,gpu,p,handle));print('start',seed,gpu,p.pid,flush=True)
    failures=[]
    while jobs:
        write_json(status,dict(status='running',pid=os.getpid(),active=[dict(seed=s,gpu=g,pid=p.pid) for s,g,p,f in jobs],failed=failures))
        for seed,gpu,p,handle in list(jobs):
            code=p.poll()
            if code is not None:
                handle.close();jobs.remove((seed,gpu,p,handle))
                if code:failures.append(dict(seed=seed,exit_code=code))
                print('finish',seed,code,flush=True)
        if jobs:time.sleep(5)
    if failures:raise RuntimeError(f'Endpoint runs failed: {failures}')
    write_json(status,dict(status='reporting',pid=os.getpid()))
    subprocess.run([config['execution']['python'],str(root/'src/compare.py'),'--config',str(root/'configs/experiments.json')],check=True,cwd=root)
    write_json(status,dict(status='complete',pid=os.getpid(),seeds=ext['seeds'],test_evaluated=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();config=read_config(args.config)
    try:main(config)
    except Exception as e:
        write_json(Path(config['paths']['reports'])/'queue_error.json',dict(status='failed',pid=os.getpid(),error=str(e)))
        raise
