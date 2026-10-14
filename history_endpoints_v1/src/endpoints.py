"""Historical endpoints only; the original experiment modules remain unchanged."""
import sys
from pathlib import Path
BASE=Path('/home/li/football/graph_prediction_research')
sys.path.insert(0,str(BASE/'src'))
import numpy as np
import torch
from torch import nn
from data import read_config,write_json,anchor_frame,clean_event
from train import Split
from model import Predictor,pair_channels


def clean_endpoint(event,extension):
    positions=event['positions']
    if len(positions) not in [0,1,2]:raise ValueError('Unexpected coordinate record length')
    sub=-1 if event['subEventId']=='' else int(event['subEventId'])
    placeholder=int(event['eventId']) in extension['endpoint_placeholder_event_ids'] or sub in extension['endpoint_placeholder_subevents']
    if len(positions)<2:return np.zeros(2,np.float32),False
    xy=np.array([positions[1]['x'],positions[1]['y']],np.float32)/100
    if not np.isfinite(xy).all() or (xy<0).any() or (xy>1).any():raise ValueError('Invalid raw endpoint')
    if placeholder:
        if not ((xy==0).all() or (xy==1).all()):raise ValueError('Audited placeholder endpoint is not a corner placeholder')
        return np.zeros(2,np.float32),False
    return xy,True


def build(config):
    output=Path(config['paths']['endpoint_data']);original=Path(config['paths']['data']);ext=config['extension']
    for split in ['train','validation']:
        if (output/f'{split}_endpoints.npz').exists():raise FileExistsError(split)
    raw=read_config(config['paths']['raw_events']);report={}
    for split in ['train','validation']:
        index=dict(np.load(original/f'{split}_index.npz'));inputs=dict(np.load(original/f'{split}_inputs.npz'))
        n=len(index['event_id']);xy=np.zeros((n,2),np.float32);valid=np.zeros(n,bool);pass_tags={301:0,302:0};kept_corners=0
        for i,original_order in enumerate(index['original_order']):
            event=raw[int(original_order)]
            if event['id']!=index['event_id'][i] or event['matchId']!=index['match_id'][i]:raise ValueError('Raw record alignment differs')
            clean=clean_event(event)
            for field in ['event','subevent','period','player','team']:
                if clean[field]!=inputs[field][i]:raise ValueError('Whitelist record differs: '+field)
            xy[i],valid[i]=clean_endpoint(event,ext)
            kept_corners+=int(valid[i] and ((xy[i]==0).all() or (xy[i]==1).all()))
            # Audit only; tags are not stored in the model endpoint table.
            if clean['event']==8:
                tags={int(t['id']) for t in event['tags']}
                for tag in pass_tags:pass_tags[tag]+=int(tag in tags)
        np.savez(output/f'{split}_endpoints.npz',end_xy=xy,end_valid=valid)
        report[split]=dict(records=n,valid_endpoints=int(valid.sum()),masked_missing_or_placeholder=int((~valid).sum()),retained_corner_endpoints_in_mixed_types=kept_corners,pass_records_with_assist=pass_tags[301],pass_records_with_key_pass=pass_tags[302])
    write_json(Path(config['paths']['reports'])/'endpoint_data_audit.json',report)
    print(report,flush=True)


def endpoint_batch(inputs,targets,endpoints,rows,valid,offsets):
    """Use observed selected historical rows only; mask before gathering values."""
    anchor=targets['anchor'][rows]
    indices=anchor[:,None]-offsets
    allowed=valid & (offsets>0) & (indices>=targets['match_start'][rows,None])
    xy=np.zeros((*indices.shape,2),np.float32);mask=np.zeros(indices.shape,bool)
    b,j=np.nonzero(allowed)
    if len(b):
        ids=indices[b,j]
        values,known=anchor_frame(endpoints['end_xy'][ids],inputs['team'][ids],inputs['team'][anchor[b]],endpoints['end_valid'][ids])
        xy[b,j]=values;mask[b,j]=known
    return xy,mask


class EndpointSplit(Split):
    def __init__(self,config,name,condition,seed):
        super().__init__(config,name,condition,seed)
        self.endpoints=dict(np.load(Path(config['paths']['endpoint_data'])/f'{name}_endpoints.npz'))
        if len(self.endpoints['end_valid'])!=len(self.inputs['event']):raise ValueError('Endpoint population differs')
    def batch(self,rows,device):
        x,target=super().batch(rows,'cpu')
        offsets=self.offsets[rows] if self.offsets is not None else np.broadcast_to(np.arange(self.condition['window']-1,-1,-1),(len(rows),self.condition['window']))
        xy,valid=endpoint_batch(self.inputs,self.targets,self.endpoints,rows,x['valid'].numpy(),offsets)
        x['end_xy']=torch.as_tensor(xy);x['end_valid']=torch.as_tensor(valid)
        return {k:v.to(device) for k,v in x.items()},{k:v.to(device) for k,v in target.items()}


class EndpointPredictor(Predictor):
    def __init__(self,config,condition,task):
        super().__init__(config,condition,task)
        if task!='etp' or 'X' not in condition['groups']:raise ValueError('This follow-up is ETP with X enabled')
        # Appended last: construction of every common parameter remains identical.
        self.endpoint=nn.Linear(3,config['backbone']['hidden'],bias=False)
    def forward(self,batch):
        c=self.condition;valid=batch['valid']
        x=self.event(batch['event'])+self.subevent(torch.where(valid,batch['subevent']+1,0))+self.period(batch['period'])
        positions=torch.where(valid,valid.long().cumsum(1),0);x=x+self.order(positions)
        if 'C' in c['groups']:
            prior=torch.roll(batch['seconds'],1,1);same=batch['period']==torch.roll(batch['period'],1,1)
            same[:,0]=False;same &= valid&torch.roll(valid,1,1)
            temporal=torch.stack([batch['seconds']/3600,torch.where(same,(batch['seconds']-prior)/60,0),same.float()],-1)
            x=x+self.C(temporal)
        if 'X' in c['groups']:x=x+self.X(torch.cat([batch['xy'],batch['xy_valid'][:,:,None].float()],-1))
        end_valid=batch['end_valid']&valid&(batch['order']<0)
        end_xy=torch.where(end_valid[:,:,None],batch['end_xy'],0)
        x=x+self.endpoint(torch.cat([end_xy,end_valid[:,:,None].float()],-1))
        if 'P' in c['groups']:x=x+self.P(batch['player'])
        if 'T' in c['groups']:x=x+self.T(batch['team'])
        if c['possession']:x=x+self.Z(batch['phase'])
        channels,allowed,_=pair_channels(batch,c);x=self.drop(x)*valid[:,:,None]
        for layer in self.layers:x=layer(x,channels,allowed,valid)
        anchor=self.norm(x[:,-1]);out={k:head(anchor) for k,head in self.heads.items()};out['time']=out['time'].squeeze(-1)
        return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();build(read_config(a.config))
