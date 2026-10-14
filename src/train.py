"""One fresh run, explicit condition/task/LR; no test loader or checkpoint refit."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from data import read_config,write_json,windows,features
from model import Predictor,loss,pair_channels
from evaluation import VIEWS,match_table,metric_array,json_metrics


def deterministic(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1);torch.backends.mkldnn.enabled=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def stable_seed(*parts):
    # Sampling seed only, matching the existing Rotate algorithm; not artifact hashing.
    return int.from_bytes(hashlib.sha256(':'.join(map(str,parts)).encode()).digest()[:8],'little')


def rotate_orders(targets,seed,epochs):
    tables=[]
    for mid in np.unique(targets['match_id']):
        rows=np.flatnonzero(targets['match_id']==mid);n=len(rows);count=128
        if n<count:raise ValueError('Match has fewer than 128 targets')
        first_generator=torch.Generator().manual_seed(stable_seed(seed,'train',mid))
        slots=[]
        for slot in range(count):
            lo,hi=slot*n//count,(slot+1)*n//count
            first=lo+int(torch.randint(hi-lo,(1,),generator=first_generator))
            rest=[x for x in range(lo,hi) if x!=first]
            order=[first,*np.random.default_rng(stable_seed(seed,mid,slot)).permutation(rest).tolist()]
            slots.append(rows[np.array([order[e%len(order)] for e in range(epochs)])])
        tables.append(np.stack(slots,1))
    generator=torch.Generator().manual_seed(seed);result=[]
    for e in range(epochs):
        result.append(np.concatenate([tables[m][e][torch.randperm(128,generator=generator).numpy()] for m in torch.randperm(len(tables),generator=generator).tolist()]))
    return np.stack(result)


class Split:
    def __init__(self,config,name,condition,seed):
        root=Path(config['paths']['data']);self.name=name;self.condition=condition
        self.inputs=dict(np.load(root/f'{name}_inputs.npz'));self.targets=dict(np.load(root/f'{name}_targets.npz'))
        self.offsets=None;self.q=None
        if condition['selection']!='recent':
            selection_seed=config['selection']['seed_map'][str(seed)]
            with np.load(root/f"{name}_{condition['selection']}_{selection_seed}.npz") as selected:
                self.offsets=selected[condition['arm']];self.q=selected['q']
    def batch(self,rows,device):
        offsets=None if self.offsets is None else self.offsets[rows]
        w=windows(self.inputs,self.targets,rows,self.condition['window'],offsets)
        f=features(w)
        inputs={k:torch.as_tensor(v,device=device) for k,v in f.items()}
        targets={k:torch.as_tensor(v[rows],device=device) for k,v in self.targets.items()}
        return inputs,targets


def evaluate(model,dataset,device,batch_size):
    model.eval();preds={};start=time.perf_counter()
    reach=[];spans=[]
    with torch.no_grad():
        for lo in range(0,len(dataset.targets['anchor']),batch_size):
            rows=np.arange(lo,min(lo+batch_size,len(dataset.targets['anchor'])))
            x,y=dataset.batch(rows,device);p=model(x)
            if not all(torch.isfinite(v).all() for v in p.values()):raise FloatingPointError('Nonfinite validation prediction')
            for k,v in p.items():preds.setdefault(k,[]).append(v.cpu().numpy())
            _,_,edges=pair_channels(x,model.condition)
            one=edges[:,-1];two=(one[:,:,None]&edges).any(1)|one
            reach.append(torch.stack([one.sum(1),two.sum(1)],1).cpu().numpy())
            distance=x['seconds'][:,-1:]-x['seconds'];valid_time=x['period']==x['period'][:,-1:]
            spans.append(torch.stack([torch.where(one&valid_time,distance,0).max(1).values,torch.where(two&valid_time,distance,0).max(1).values],1).cpu().numpy())
    preds={k:np.concatenate(v) for k,v in preds.items()};tables={};metrics={}
    for view in VIEWS:
        matches,table=match_table(preds,dataset.targets,model.task,view);tables[view]=table
        metrics[view]=json_metrics(metric_array(table.sum(0),model.task))
    if dataset.q is not None:
        _,table=match_table(preds,dataset.targets,model.task,'duel_corrected',subset=dataset.q>0)
        tables['q_positive']=table;metrics['q_positive']=json_metrics(metric_array(table.sum(0),model.task))
    elapsed=time.perf_counter()-start
    return preds,matches,tables,metrics,dict(seconds=elapsed,samples_per_second=len(dataset.targets['anchor'])/elapsed,reachability=np.concatenate(reach),same_period_time_span=np.concatenate(spans))


class Stopper:
    def __init__(self,c):self.c=c;self.reference=float('inf');self.bad=0;self.decays=0
    def step(self,epoch,value):
        if epoch<=self.c['warmup_no_decay_epochs']:
            self.reference=min(self.reference,value);return 'continue'
        if value<=self.reference*(1-self.c['relative_improvement']):
            self.reference=value;self.bad=0;return 'continue'
        self.bad+=1
        if self.bad<self.c['patience_validations']:return 'continue'
        if self.decays<self.c['max_lr_decays']:
            self.decays+=1;self.bad=0;return 'decay'
        return 'stop' if epoch>=self.c['min_stop_epoch'] else 'continue'


def fit(config,condition_id,task,seed,lr,device,mode,run_id):
    root=Path(config['paths']['runs'])/run_id
    acceptance=read_config(Path(config['paths']['reports'])/'acceptance.json')
    if acceptance['status']!='passed':raise ValueError('Stage A has not passed')
    if mode=='formal':
        thresholds=read_config(Path(config['paths']['reports'])/'thresholds.json')
        if thresholds['status']!='locked':raise ValueError('Thresholds are not locked')
    root.mkdir(exist_ok=False)
    deterministic(seed);condition=config['conditions'][condition_id]
    train=Split(config,'train',condition,seed);val=Split(config,'validation',condition,seed)
    model=Predictor(config,condition,task).to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=config['training']['weight_decay'])
    epochs=config['lr_search']['epochs'] if mode=='lr_search' else config['training']['max_epochs']
    orders=rotate_orders(train.targets,seed,epochs)
    if orders.shape!=(epochs,34048):raise ValueError('Rotate count differs')
    np.save(root/'sample_order.npy',orders)
    fields={'core':config['information']['core'],'groups':{g:config['information']['groups'][g] for g in condition['groups']},'possession':condition['possession'],'coordinate_frame':config['coordinates']}
    write_json(root/'config.json',dict(protocol=config,condition_id=condition_id,condition=condition,task=task,seed=seed,lr=lr,mode=mode,device=device,epochs=epochs,input_fields=fields))
    write_json(root/'status.json',dict(status='running',pid=os.getpid()))
    stopper=Stopper(config['training']);best=float('inf');best_epoch=0;start=time.perf_counter();steps=0
    gradient_seen={name:torch.zeros_like(p,dtype=torch.bool) for name,p in model.named_parameters()}
    total_parameters=sum(p.numel() for p in model.parameters());gradient_tensors=set()
    torch.cuda.reset_peak_memory_stats(device)
    stop_reason='calibration_budget' if mode=='lr_search' else 'budget_exhausted'
    monitor=config['tasks'][task]['monitor']
    for epoch in range(1,epochs+1):
        model.train();values=[];epoch_start=time.perf_counter();epoch_lr=optimizer.param_groups[0]['lr']
        for lo in range(0,len(orders[epoch-1]),256):
            rows=orders[epoch-1,lo:lo+256];x,y=train.batch(rows,device)
            optimizer.zero_grad(set_to_none=True);pred=model(x);objective=loss(pred,y,task)
            if not torch.isfinite(objective):raise FloatingPointError('Nonfinite training loss')
            objective.backward()
            for name,p in model.named_parameters():
                if p.grad is not None:
                    if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite gradient: '+name)
                    gradient_tensors.add(name);gradient_seen[name] |= p.grad!=0
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['training']['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();values.append(float(objective.detach()));steps+=1
        row=dict(epoch=epoch,optimizer_steps=steps,cumulative_targets=epoch*34048,full_train_equivalents=epoch*34048/449025,train_loss=float(np.mean(values)),lr=epoch_lr,train_seconds=time.perf_counter()-epoch_start)
        action='continue'
        if epoch%config['training']['validation_every']==0:
            preds,matches,tables,metrics,cost=evaluate(model,val,device,256)
            row['validation']=metrics;row['validation_seconds']=cost['seconds'];value=metrics['per_record'][monitor]
            if value<best:
                best=value;best_epoch=epoch
                torch.save(dict(model=model.state_dict(),epoch=epoch,monitor=best),root/'best.pt')
                np.savez(root/'best_validation_predictions.npz',**{'pred_'+k:v for k,v in preds.items()},**{'target_'+k:v for k,v in val.targets.items()})
                np.savez(root/'best_match_statistics.npz',match_id=matches,**tables)
                np.savez(root/'reachability.npz',anchor=val.targets['anchor'],match_id=val.targets['match_id'],count=cost['reachability'],same_period_seconds=cost['same_period_time_span'])
                write_json(root/'best_metrics.json',dict(epoch=epoch,metrics=metrics,inference_samples_per_second=cost['samples_per_second']))
            if mode!='lr_search':
                action=stopper.step(epoch,value)
                if action=='decay':
                    for group in optimizer.param_groups:group['lr']*=config['training']['lr_decay_factor']
                elif action=='stop':stop_reason='plateau_after_two_decays'
        row['next_lr']=optimizer.param_groups[0]['lr'];row['scheduler_action']=action;row['elapsed_seconds']=time.perf_counter()-start
        with (root/'curve.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,seed=seed),root/'last.pt')
        write_json(root/'status.json',dict(status='running',pid=os.getpid(),epoch=epoch,best_epoch=best_epoch,best_monitor=None if best_epoch==0 else best))
        print(run_id,'epoch',epoch,'train',row['train_loss'],'best',best,'seconds',round(row['elapsed_seconds'],1),flush=True)
        if action=='stop':break
    cost=dict(total_parameters=total_parameters,parameters_in_gradient_tensors=sum(p.numel() for name,p in model.named_parameters() if name in gradient_tensors),observed_nonzero_gradient_elements=sum(int(v.sum()) for v in gradient_seen.values()),peak_gpu_bytes=torch.cuda.max_memory_allocated(device),training_wall_seconds=time.perf_counter()-start)
    write_json(root/'cost.json',cost)
    if val.q is not None:write_json(root/'selection.json',dict(validation_coverage=float((val.q>0).mean()),validation_q_counts=np.bincount(val.q,minlength=9).tolist(),training_coverage=float((train.q>0).mean()),training_q_counts=np.bincount(train.q,minlength=9).tolist()))
    write_json(root/'status.json',dict(status='complete',stop_reason=stop_reason,epochs=epoch,best_epoch=best_epoch,best_monitor=best,optimizer_steps=steps,cumulative_targets=epoch*34048))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for field in ['config','condition','task','mode','run-id']:p.add_argument('--'+field,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--device',required=True)
    a=p.parse_args();c=read_config(a.config)
    if a.task not in c['tasks'] or a.mode not in ['lr_search','baseline_calibration','formal']:raise ValueError('Unknown task/mode')
    fit(c,a.condition,a.task,a.seed,a.lr,a.device,a.mode,a.run_id)
