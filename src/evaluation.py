"""Per-match sufficient statistics and paired match-cluster bootstrap."""
import math
import numpy as np

VIEWS=('per_record','duel_corrected','nonpaired')
# Scalar sums followed by the 10x10 Event and 2x2 Team confusion matrices.
SCALARS=('event_n','event_ce_sum','time_n','time_loss_sum','time_abs_sum','position_n','position_loss_sum','position_distance_sum','team_n','team_nll_sum')
WIDTH=len(SCALARS)+100+4


def match_table(pred,target,task,view,subset=None):
    n=len(target['anchor']);keep=np.ones(n,bool) if subset is None else subset.copy()
    if view=='duel_corrected': keep &= ~target['duel_second']
    elif view=='nonpaired': keep &= ~(target['duel_first']|target['duel_second'])
    elif view!='per_record': raise ValueError(view)
    matches,inverse=np.unique(target['match_id'],return_inverse=True)
    table=np.zeros((len(matches),WIDTH),np.float64)
    def add(col,values,mask):
        np.add.at(table[:,col],inverse,np.where(mask,values,0))
    if task=='etp':
        logits=pred['event'].astype(np.float64);logp=logits-logits.max(1,keepdims=True);logp-=np.log(np.exp(logp).sum(1,keepdims=True))
        ce=-logp[np.arange(n),target['event']]
        tm=keep&target['time_mask'];pm=keep&target['position_mask']
        td=np.abs(pred['time']-target['time'])
        d=np.abs(pred['position']-target['xy'])
        tl=np.where(td<1,0.5*td**2,td-0.5)/60
        pl=np.where(d<1,0.5*d**2,d-0.5).mean(-1)
        distance=np.linalg.norm((pred['position']-target['xy'])*np.array([105,68]),axis=-1)
        for col,values,mask in [(0,1,keep),(1,ce,keep),(2,1,tm),(3,tl,tm),(4,td,tm),(5,1,pm),(6,pl,pm),(7,distance,pm)]: add(col,values,mask)
        indices=np.flatnonzero(keep);code=target['event'][indices]*10+logits.argmax(1)[indices]
        np.add.at(table,(inverse[indices],len(SCALARS)+code),1)
    else:
        mask=keep&target['team_mask']
        if view=='duel_corrected': mask &= ~target['duel_first']
        logits=pred['team'].astype(np.float64);logp=logits-logits.max(1,keepdims=True);logp-=np.log(np.exp(logp).sum(1,keepdims=True))
        ce=-logp[np.arange(n),target['team']]
        add(8,1,mask);add(9,ce,mask)
        indices=np.flatnonzero(mask);code=target['team'][indices]*2+logits.argmax(1)[indices]
        np.add.at(table,(inverse[indices],len(SCALARS)+100+code),1)
    return matches,table


def metric_array(table,task):
    def ratio(a,b):return table[...,a]/np.maximum(table[...,b],1)
    def classification(start,k):
        c=table[...,start:start+k*k].reshape(*table.shape[:-1],k,k)
        diag=np.diagonal(c,axis1=-2,axis2=-1)
        f1=2*diag/np.maximum(c.sum(-1)+c.sum(-2),1)
        return diag.sum(-1)/np.maximum(c.sum((-1,-2)),1),f1
    if task=='etp':
        acc,f1=classification(len(SCALARS),10)
        ce,tl,pl=ratio(1,0),ratio(3,2),ratio(6,5)
        return dict(etp_loss=(0.2*ce/math.log(10)+tl+pl)/3,event_ce=ce,time_loss=tl,position_loss=pl,event_accuracy=acc,event_macro_f1=f1.mean(-1),event_per_class_f1=f1,time_mae_seconds=ratio(4,2),position_error_metres=ratio(7,5))
    acc,f1=classification(len(SCALARS)+100,2)
    return dict(team_nll=ratio(9,8),team_accuracy=acc,team_macro_f1=f1.mean(-1))


def json_metrics(metrics):
    return {k:v.tolist() if isinstance(v,np.ndarray) else float(v) for k,v in metrics.items()}


def aligned(a,b):
    for key in ['anchor','match_id','event','time','xy','team','time_mask','position_mask','team_mask','duel_first','duel_second']:
        if not np.array_equal(a[key],b[key]):raise ValueError('Unaligned samples/targets: '+key)


def bootstrap_weights(matches,replicates,seed):
    return np.random.default_rng(seed).multinomial(matches,np.full(matches,1/matches),replicates)


def paired_bootstrap(a,b,weights,task,family_size=24):
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1]!=weights.shape[1]:raise ValueError('Bootstrap dimensions differ')
    am,bm=metric_array(a.sum(1),task),metric_array(b.sum(1),task)
    draws={k:[] for k in am if k!='event_per_class_f1'}
    for w in np.array_split(weights,40):
        aa=metric_array(np.einsum('bm,smk->bsk',w,a),task);bb=metric_array(np.einsum('bm,smk->bsk',w,b),task)
        for k in draws:draws[k].append((aa[k]-bb[k]).mean(1))
    result={};tail=0.05/(2*family_size)
    for k,values in draws.items():
        delta=am[k]-bm[k];draw=np.concatenate(values)
        result[k]=dict(per_seed=delta.tolist(),mean=float(delta.mean()),sd=float(delta.std(ddof=1)),ci95=np.quantile(draw,[.025,.975]).tolist(),corrected_ci=np.quantile(draw,[tail,1-tail]).tolist())
    return result
