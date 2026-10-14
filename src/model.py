"""Common Pre-LN backbone; all conditions share parameter construction order."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def pair_channels(batch, condition):
    valid=batch['valid']; n=valid.shape[1]; device=valid.device
    index=torch.arange(n,device=device)
    causal=index[:,None]>=index[None,:]
    edge=causal[None]&valid[:,:,None]&valid[:,None,:]
    period_same=batch['period'][:,:,None]==batch['period'][:,None,:]
    dn=(batch['order'][:,:,None]-batch['order'][:,None,:]).float()/80
    dt_valid=period_same&edge
    dt=(batch['seconds'][:,:,None]-batch['seconds'][:,None,:])/60
    if 'C' not in condition['groups']:
        dt=dt*0;dt_valid=dt_valid&False
    dt=torch.where(dt_valid,dt,0)
    relation=[]
    for name,enabled in [('player','P' in condition['groups']),('team','T' in condition['groups']),('phase',condition['possession'])]:
        v=batch[name]
        r=(v[:,:,None]==v[:,None,:])&(v[:,:,None]>0)&(v[:,None,:]>0)&period_same&edge
        relation.append(r if enabled else r&False)
    # Sparse phase links need phases even when not provided as a node input.
    sparse=edge
    if condition['connection']=='sparse':
        strict=edge&~torch.eye(n,dtype=torch.bool,device=device)[None]
        recent=strict&(index[None,:,None]-index[None,None,:]<=8)
        same_player,same_team,same_phase=relation
        different_known_player=(batch['player'][:,:,None]>0)&(batch['player'][:,None,:]>0)&(batch['player'][:,:,None]!=batch['player'][:,None,:])
        sparse=recent|torch.eye(n,dtype=torch.bool,device=device)[None]&edge
        for rel in [same_player,same_team&different_known_player,same_phase]:
            eligible=rel&strict
            rank=eligible.flip(-1).long().cumsum(-1).flip(-1)
            sparse |= eligible&(rank<=4)
    rb=relation if condition['relation_bias'] else [r&False for r in relation]
    channels=torch.stack([dn,dt,dt_valid.float(),*(r.float() for r in rb)],dim=-1)
    # Invalid queries have a harmless diagonal to avoid an all-masked softmax.
    safe=sparse|(~valid[:,:,None]&torch.eye(n,dtype=torch.bool,device=device)[None])
    return channels,safe,sparse


class Layer(nn.Module):
    def __init__(self,c):
        super().__init__();d=c['hidden'];self.heads=c['heads'];self.width=d//self.heads
        self.ln1=nn.LayerNorm(d);self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.bias=nn.Sequential(nn.Linear(6,c['time_bias']['mlp_hidden']),nn.GELU(),nn.Linear(c['time_bias']['mlp_hidden'],self.heads))
        self.ln2=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,c['ffn']),nn.GELU(),nn.Dropout(c['dropout']),nn.Linear(c['ffn'],d))
        self.drop=nn.Dropout(c['dropout'])
    def forward(self,x,channels,allowed,valid):
        b,n,d=x.shape
        q,k,v=self.qkv(self.ln1(x)).reshape(b,n,3,self.heads,self.width).permute(2,0,3,1,4).unbind(0)
        score=(q@k.transpose(-1,-2))/math.sqrt(self.width)+self.bias(channels).permute(0,3,1,2)
        attention=score.masked_fill(~allowed[:,None],float('-inf')).softmax(-1)
        message=(self.drop(attention)@v).transpose(1,2).reshape(b,n,d)
        x=x+self.drop(self.out(message));x=x+self.drop(self.ff(self.ln2(x)))
        return x*valid[:,:,None]


class Predictor(nn.Module):
    def __init__(self,config,condition,task):
        super().__init__();self.condition=condition;self.task=task;c=config['backbone'];d=c['hidden']
        self.event=nn.Embedding(11,d,padding_idx=0);self.subevent=nn.Embedding(102,d,padding_idx=0)
        self.period=nn.Embedding(6,d,padding_idx=0);self.order=nn.Embedding(81,d,padding_idx=0)
        self.C=nn.Linear(3,d,bias=False);self.X=nn.Linear(3,d,bias=False)
        self.P=nn.Embedding(81,d,padding_idx=0);self.T=nn.Embedding(3,d,padding_idx=0);self.Z=nn.Embedding(81,d,padding_idx=0)
        self.drop=nn.Dropout(c['dropout']);self.layers=nn.ModuleList([Layer(c) for _ in range(c['layers'])]);self.norm=nn.LayerNorm(d)
        # Only the chosen task's heads exist; runs never share trained parameters.
        self.heads=nn.ModuleDict({'event':nn.Linear(d,10),'time':nn.Linear(d,1),'position':nn.Linear(d,2)} if task=='etp' else {'team':nn.Linear(d,2)})
    def forward(self,batch):
        c=self.condition;valid=batch['valid']
        x=self.event(batch['event'])+self.subevent(torch.where(valid,batch['subevent']+1,0))+self.period(batch['period'])
        positions=torch.where(valid,valid.long().cumsum(1),0)
        x=x+self.order(positions)
        if 'C' in c['groups']:
            prior=torch.roll(batch['seconds'],1,1);same=batch['period']==torch.roll(batch['period'],1,1)
            same[:,0]=False;same &= valid&torch.roll(valid,1,1)
            temporal=torch.stack([batch['seconds']/3600,torch.where(same,(batch['seconds']-prior)/60,0),same.float()],-1)
            x=x+self.C(temporal)
        if 'X' in c['groups']: x=x+self.X(torch.cat([batch['xy'],batch['xy_valid'][:,:,None].float()],-1))
        if 'P' in c['groups']: x=x+self.P(batch['player'])
        if 'T' in c['groups']: x=x+self.T(batch['team'])
        if c['possession']: x=x+self.Z(batch['phase'])
        channels,allowed,_=pair_channels(batch,c)
        x=self.drop(x)*valid[:,:,None]
        for layer in self.layers: x=layer(x,channels,allowed,valid)
        anchor=self.norm(x[:,-1])
        out={k:head(anchor) for k,head in self.heads.items()}
        if self.task=='etp': out['time']=out['time'].squeeze(-1)
        return out


def per_record(pred,target,task):
    if task=='team_only':
        return {'team_nll':F.cross_entropy(pred['team'],target['team'].long(),reduction='none')}
    return dict(event_ce=F.cross_entropy(pred['event'],target['event'].long(),reduction='none'),
        time_loss=F.smooth_l1_loss(pred['time']/60,target['time'].float()/60,beta=1/60,reduction='none'),
        position_loss=F.smooth_l1_loss(pred['position'],target['xy'].float(),beta=1,reduction='none').mean(-1))


def loss(pred,target,task):
    p=per_record(pred,target,task)
    def masked(value,mask):
        return (value*mask).sum()/mask.sum().clamp_min(1)
    if task=='team_only':return masked(p['team_nll'],target['team_mask'])
    return (0.2*p['event_ce'].mean()/math.log(10)+masked(p['time_loss'],target['time_mask'])+masked(p['position_loss'],target['position_mask']))/3
