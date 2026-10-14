"""Explicit four-GPU queue: LR calibration, baseline SD, then B/C/D runs."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from data import read_config,write_json


def jobs_run(config,jobs,stage):
    root=Path(config['root']);pending=list(jobs);active={};completed=[];failed=[]
    report=root/'reports/queue_status.json'
    def status():write_json(report,dict(status='failed' if failed else 'running',stage=stage,pid=os.getpid(),pending=[j['run_id'] for j in pending],active={str(g):dict(run_id=j['run_id'],pid=p.pid) for g,(p,f,j) in active.items()},completed=completed,failed=failed))
    while pending or active:
        for gpu in config['execution']['gpus']:
            if gpu not in active and pending and not failed:
                job=pending.pop(0)
                if (root/'runs'/job['run_id']).exists():raise FileExistsError(job['run_id'])
                log=(root/'reports'/f"{job['run_id']}.log").open('x')
                command=[config['execution']['python'],str(root/'src/train.py'),'--config',str(root/'configs/experiments.json'),'--condition',job['condition'],'--task',job['task'],'--seed',str(job['seed']),'--lr',str(job['lr']),'--mode',job['mode'],'--device',f'cuda:{gpu}','--run-id',job['run_id']]
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=root)
                active[gpu]=(process,log,job)
                print('start',stage,gpu,job['run_id'],process.pid,flush=True)
        for gpu,(process,log,job) in list(active.items()):
            code=process.poll()
            if code is not None:
                log.close();del active[gpu]
                if code:failed.append(dict(run_id=job['run_id'],exit_code=code))
                else:completed.append(job['run_id'])
                print('finish',job['run_id'],code,flush=True)
        status()
        if failed and not active:raise RuntimeError(f'Queue stopped after failure: {failed}')
        if pending or active:time.sleep(5)
    return jobs


def make_job(condition,task,seed,lr,mode):
    return dict(condition=condition,task=task,seed=seed,lr=lr,mode=mode,run_id=f'{mode}_{task}_{condition}_s{seed}_lr{lr:g}')


def main(config):
    root=Path(config['root']);report=root/'reports'
    if read_config(report/'acceptance.json')['status']!='passed':raise ValueError('Acceptance has not passed')
    if (report/'queue_status.json').exists():raise FileExistsError('Explicit queue already exists; inspect its status')
    grid=[make_job(cond,task,config['lr_search']['seed'],lr,'lr_search') for lr in config['lr_search']['candidates'] for task in config['tasks'] for cond in ['D00','D11']]
    write_json(report/'lr_search_plan.json',grid);jobs_run(config,grid,'learning_rate_search')
    selected={};scores={}
    for task in config['tasks']:
        scores[task]=[]
        for lr in config['lr_search']['candidates']:
            runs=[j for j in grid if j['task']==task and j['lr']==lr]
            values=[read_config(root/'runs'/j['run_id']/'status.json')['best_monitor'] for j in runs]
            scores[task].append(dict(lr=lr,reference_scores=values,mean=float(np.mean(values))))
        selected[task]=min(scores[task],key=lambda row:(row['mean'],row['lr']))['lr']
    write_json(report/'selected_learning_rates.json',dict(status='locked',selected=selected,scores=scores))
    baselines=[make_job('D00',task,seed,selected[task],'baseline_calibration') for task in config['tasks'] for seed in config['training']['seeds']]
    jobs_run(config,baselines,'D00_three_seed_calibration')
    thresholds={}
    for task in config['tasks']:
        runs=[j for j in baselines if j['task']==task];metric=config['tasks'][task]['monitor']
        values=[read_config(root/'runs'/j['run_id']/'best_metrics.json')['metrics']['duel_corrected'][metric] for j in runs]
        sd=float(np.std(values,ddof=1));thresholds[task]=dict(metric=metric,per_seed=values,sd=sd,minimum_absolute_improvement=3*sd,mean=float(np.mean(values)),etp_relative_equivalent=3*sd/np.mean(values) if task=='etp' else None)
    write_json(report/'thresholds.json',dict(status='locked',formula='3 * D00 sample SD (ddof=1)',view='duel_corrected_validation',tasks=thresholds,runs=baselines))
    confirmation=set()
    for stage in ['B','C1','C2','D']:confirmation.update(config['stages'][stage]['confirm'])
    formal=[]
    for task in config['tasks']:
        for condition in config['conditions']:
            if condition=='D00':continue
            seeds=config['training']['seeds'] if condition in confirmation else [config['training']['exploration_seed']]
            for seed in seeds:formal.append(make_job(condition,task,seed,selected[task],'formal'))
    # First complete the exploratory seed across stages, then remaining confirmations.
    formal.sort(key=lambda j:j['seed'])
    write_json(report/'formal_plan.json',formal)
    jobs_run(config,formal,'B_C_D_training')
    write_json(report/'completed_runs.json',baselines+formal)
    subprocess.run([config['execution']['python'],str(root/'src/report.py'),'--config',str(root/'configs/experiments.json')],check=True,cwd=root)
    write_json(report/'queue_status.json',dict(status='complete',stage='B_C_D_analysis',pid=os.getpid(),test_evaluated=False,next='Review adoption results before implementing or launching eligible external backbones'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    try:main(read_config(a.config))
    except Exception:
        import traceback
        traceback.print_exc();sys.exit(1)
