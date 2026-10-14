import copy
import io
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from data import clean_event,anchor_frame,windows,features,phases,paired_select,read_config
from model import Predictor,pair_channels,loss
from train import deterministic,Split,Stopper,rotate_orders
from evaluation import match_table,metric_array,aligned,bootstrap_weights,paired_bootstrap

CONFIG=read_config(ROOT/'configs/experiments.json')


def tensor_features(w):return {k:torch.as_tensor(v) for k,v in features(w).items()}


def synthetic(n=100):
    x=dict(event=np.full(n,8),subevent=np.full(n,85),period=np.ones(n,dtype=np.int64),seconds=np.arange(n,dtype=np.float32)*2,
        player=np.arange(n)%6+1,team=np.arange(n)%2+1,xy=np.tile([.2,.4],(n,1)).astype(np.float32),xy_valid=np.ones(n,bool))
    t=dict(anchor=np.arange(n-1),match_start=np.zeros(n-1,np.int64))
    return x,t


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deterministic(20260715)
        cls.split=Split(CONFIG,'validation',CONFIG['conditions']['D00'],20260715)
    def test_01_whitelist_ignores_forbidden_fields(self):
        e=dict(eventId=8,subEventId=85,matchPeriod='1H',eventSec=10,positions=[dict(x=20,y=40),dict(x=99,y=99)],playerId=1,teamId=2)
        other=copy.deepcopy(e);other['positions'][1]=dict(x=0,y=0);other.update(tags=[dict(id=101)],score=12,old_possession=999,outcome='failure')
        self.assertEqual(clean_event(e),clean_event(other))
        other['subEventId']='';self.assertEqual(clean_event(other)['subevent'],-1)
    def test_02_future_invariance(self):
        x,t=synthetic();rows=np.array([50]);before=features(windows(x,t,rows))
        changed=copy.deepcopy(x)
        for key in ['event','subevent','player','team']:changed[key][51:]=1
        changed['xy'][51:]=.9
        after=features(windows(changed,t,rows))
        for k in before:np.testing.assert_array_equal(before[k],after[k])
        rng=np.random.default_rng(1);priority=rng.random((1,64))
        for rule in CONFIG['selection']['rules']:
            a=paired_select(windows(x,t,rows),rule,priority);b=paired_select(windows(changed,t,rows),rule,priority)
            for av,bv in zip(a,b):np.testing.assert_array_equal(av,bv)
    def test_03_rename_invariance(self):
        x,t=synthetic();rows=np.array([60,90]);w=windows(x,t,rows);before=features(w)
        x['player']=1000-x['player'];x['team']=500-x['team'];v=windows(x,t,rows);after=features(v)
        for k in before:np.testing.assert_array_equal(before[k],after[k])
        for rule in CONFIG['selection']['rules']:
            for a,b in zip(paired_select(w,rule,np.ones((2,64))),paired_select(v,rule,np.ones((2,64)))):np.testing.assert_array_equal(a,b)
        model=Predictor(CONFIG,CONFIG['conditions']['D11'],'etp').eval()
        with torch.no_grad():
            a=model(tensor_features(w));b=model(tensor_features(v))
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    def test_04_group_firewall(self):
        w=windows(*synthetic(),np.array([80,90]));f=tensor_features(w)
        for group,keys in [('C',['seconds']),('X',['xy','xy_valid']),('P',['player']),('T',['team'])]:
            c=dict(CONFIG['conditions']['B_CXPT'],groups=[g for g in 'CXPT' if g!=group])
            model=Predictor(CONFIG,c,'etp').eval();changed={k:v.clone() for k,v in f.items()}
            for k in keys:changed[k]=torch.zeros_like(changed[k])
            with torch.no_grad():a=model(f);b=model(changed)
            for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    def test_05_masks_reachability(self):
        rows=np.array([0,1,60]);f,_=self.split.batch(rows,'cpu')
        for name in ['D00','D01','D10','D11']:
            channels,safe,edges=pair_channels(f,CONFIG['conditions'][name]);valid=f['valid']
            self.assertFalse(edges.triu(1).any())
            self.assertFalse((edges&~valid[:,None,:]).any())
            torch.testing.assert_close(edges.diagonal(dim1=1,dim2=2),valid)
            self.assertTrue(safe.any(-1).all())
            if name in ['D00','D01']:torch.testing.assert_close(edges[:,-1],valid)
            self.assertTrue((edges.sum(-1)<=21).all() if name in ['D10','D11'] else True)
        a=pair_channels(f,CONFIG['conditions']['D00'])[0];b=pair_channels(f,CONFIG['conditions']['D01'])[0]
        torch.testing.assert_close(a[...,:3],b[...,:3]);self.assertFalse(a[...,3:].any());self.assertTrue(b[...,3:].any())
    def test_06_possession_boundaries(self):
        x,t=synthetic(9);x['event']=np.array([1,8,1,2,1,8,1,8,8]);x['team']=np.array([2,1,2,1,2,2,1,1,1]);x['period'][7:]=2
        w=windows(x,t,np.array([7]),8)
        np.testing.assert_array_equal(w['phase'][0],[0,1,1,1,0,2,2,3])
        # A truncated window beginning with a contest never inherits an older owner.
        short=windows(x,t,np.array([6]),1);self.assertEqual(short['phase'][0,0],0)
    def test_07_selection_matching(self):
        x,t=synthetic();rows=np.arange(70,99);w=windows(x,t,rows)
        # Create input-only phase patterns exercising every selector (including long phases).
        w['phase'][:,:30]=1;w['phase'][:,30:53]=2;w['phase'][:,53:]=3
        w['team'][:,53:]=w['team'][:,-1:]
        w['player'][:,53:]=np.tile([1,3,5],9)
        priority=np.random.default_rng(9).random((len(rows),64))
        for rule in CONFIG['selection']['rules']:
            r,c,q,dn,dt=paired_select(w,rule,priority)
            self.assertTrue((q>0).any(),rule)
            for i in range(len(rows)):
                for selected in [r[i],c[i]]:
                    self.assertEqual(selected[-1],0);self.assertEqual(len(np.unique(selected[selected>=0])),np.count_nonzero(selected>=0))
                np.testing.assert_array_equal(r[i][r[i]<16],c[i][c[i]<16])
                self.assertEqual(np.count_nonzero(r[i]>=16),q[i]);self.assertEqual(np.count_nonzero(c[i]>=16),q[i])
                def buckets(offset):return sorted((int((n-16)//16),int(np.searchsorted([5,15,30,60],w['seconds'][i,-1]-w['seconds'][i,79-n],side='right'))) for n in offset if n>=16)
                self.assertEqual(buckets(r[i]),buckets(c[i]))
        x['player'][:]=0;w=windows(x,t,np.array([0,1,30]))
        a,b,q,_,_=paired_select(w,'player',np.ones((3,64)))
        self.assertFalse(q.any());np.testing.assert_array_equal(a,b);self.assertTrue((a[:,-1]==0).all())
    def test_08_unselected_firewall(self):
        x,t=synthetic();rows=np.array([90]);offsets=np.arange(15,-1,-1)[None]
        before=features(windows(x,t,rows,16,offsets));x['event'][:75]=2;x['team'][:75]=9;x['player'][:75]=888;x['seconds'][:75]=999;x['xy'][:75]=.8
        after=features(windows(x,t,rows,16,offsets))
        for k in before:np.testing.assert_array_equal(before[k],after[k])
    def test_09_target_alignment(self):
        reference=self.split.targets
        for name,c in CONFIG['conditions'].items():
            # Every condition uses the same independent target table.
            other=Split(CONFIG,'validation',c,20260715)
            aligned(reference,other.targets)
            if other.offsets is not None:self.assertEqual(len(other.offsets),len(reference['anchor']))
    def test_10_serialization_finite_full_batch(self):
        device='cuda:0';x,y=self.split.batch(np.arange(256),'cuda:0')
        for task in ['etp','team_only']:
            deterministic(20260715);model=Predictor(CONFIG,CONFIG['conditions']['D11'],task).to(device)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
            out=model(x);value=loss(out,y,task);self.assertTrue(torch.isfinite(value));value.backward()
            for param in model.parameters():
                if param.grad is not None:self.assertTrue(torch.isfinite(param.grad).all())
            optimizer.step();model.eval()
            with torch.no_grad():before=model(x)
            stream=io.BytesIO();torch.save(model.state_dict(),stream);stream.seek(0)
            new=Predictor(CONFIG,CONFIG['conditions']['D11'],task).to(device);new.load_state_dict(torch.load(stream,weights_only=True));new.eval()
            with torch.no_grad():after=new(x)
            for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
    def test_11_duel_coordinates(self):
        x,t=self.split.inputs,self.split.targets;first=t['anchor'][t['duel_first']]+1;second=first+1
        xy,valid=anchor_frame(x['xy'][second],x['team'][second],x['team'][first],x['xy_valid'][second])
        self.assertTrue(valid.all());self.assertLess(float(np.abs(xy-x['xy'][first]).max()),.011)
    def test_12_coordinate_mirror_probe(self):
        x,t=self.split.inputs,self.split.targets;a=t['anchor'];mask=t['position_mask']&x['xy_valid'][a]&t['time_mask']
        here=x['xy'][a];there=t['xy'];predict_change=np.linalg.norm(1-here-there,axis=1)<np.linalg.norm(here-there,axis=1)
        truth=t['team']==0;acc=float((predict_change[mask]==truth[mask]).mean());majority=max(float(truth[mask].mean()),float(1-truth[mask].mean()))
        self.assertLessEqual(acc,majority)
        report=dict(population='same-period adjacent valid nonplaceholder starts with known teams',n=int(mask.sum()),mirror_rule_accuracy=acc,majority_accuracy=majority,passes=acc<=majority,anchor_copy_error_metres=float(np.linalg.norm((here-there)*[105,68],axis=1)[mask].mean()))
        from data import write_json
        write_json(ROOT/'reports/coordinate_audit.json',report)
    def test_13_bootstrap_and_statistics(self):
        t={k:v[:600] for k,v in self.split.targets.items()};t['match_id']=np.repeat([1,2,3],[100,200,300]);p=dict(event=np.zeros((600,10)),time=np.zeros(600),position=np.zeros((600,2)))
        ids,table=match_table(p,t,'etp','per_record');m=metric_array(table.sum(0),'etp')
        self.assertAlmostEqual(float(m['event_ce']),np.log(10));self.assertEqual(int(table[:,0].sum()),600)
        selected=np.concatenate([np.arange(100),np.arange(100),np.arange(300,600)])
        repeated={k:v[selected] for k,v in t.items()};pp={k:v[selected] for k,v in p.items()}
        _,direct=match_table(pp,repeated,'etp','per_record')
        np.testing.assert_allclose(direct.sum(0),np.array([2,0,1])@table)
        weights=bootstrap_weights(3,10000,20261014);a=np.stack([table]*3)
        result=paired_bootstrap(a,a,weights,'etp')
        self.assertEqual(result['etp_loss']['corrected_ci'],[0.,0.])
        bad=copy.deepcopy(t);bad['event'][0]=(bad['event'][0]+1)%10
        with self.assertRaises(ValueError):aligned(t,bad)
    def test_14_stopping_and_paired_initialization(self):
        stop=Stopper(CONFIG['training']);actions={epoch:stop.step(epoch,1.) for epoch in range(4,53,4)}
        self.assertEqual(actions[24],'continue');self.assertEqual(actions[36],'decay');self.assertEqual(actions[48],'decay');self.assertNotIn('stop',actions.values())
        self.assertEqual(stop.step(56,1.),'continue');self.assertEqual(stop.step(60,1.),'stop')
        deterministic(5);a=Predictor(CONFIG,CONFIG['conditions']['D00'],'etp')
        deterministic(5);b=Predictor(CONFIG,CONFIG['conditions']['D11'],'etp')
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    def test_15_masked_padding_and_position(self):
        x,t=synthetic();w=windows(x,t,np.array([0,1]),80);f=tensor_features(w);model=Predictor(CONFIG,CONFIG['conditions']['D00'],'etp').eval()
        altered={k:v.clone() for k,v in f.items()};altered['xy'][~altered['valid']]=900
        with torch.no_grad():a=model(f);b=model(altered)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        e=dict(eventId=3,subEventId=30,matchPeriod='1H',eventSec=0,positions=[dict(x=0,y=0)],playerId=1,teamId=1)
        self.assertTrue(clean_event(e)['xy_valid']);e['subEventId']=34;self.assertFalse(clean_event(e)['xy_valid'])


if __name__=='__main__':unittest.main(verbosity=2)
