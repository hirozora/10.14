import copy
import io
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from endpoints import EndpointSplit,EndpointPredictor,endpoint_batch,clean_endpoint
from data import read_config,clean_event,write_json
from train import deterministic,Split,rotate_orders
from model import Predictor,loss
C=read_config(ROOT/'configs/experiments.json');CONDITION=C['conditions'][C['extension']['condition']]


class EndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        deterministic(20260715);cls.dataset=EndpointSplit(C,'validation',CONDITION,20260715)
    def test_anchor_and_future_never_enter(self):
        d=self.dataset;rows=np.array([150]);x,_=d.batch(rows,'cpu');anchor=d.targets['anchor'][rows[0]]
        original=d.endpoints['end_xy'][anchor:].copy();original_mask=d.endpoints['end_valid'][anchor:].copy()
        try:
            d.endpoints['end_xy'][anchor:]=.987;d.endpoints['end_valid'][anchor:]=~original_mask
            y,_=d.batch(rows,'cpu')
        finally:d.endpoints['end_xy'][anchor:]=original;d.endpoints['end_valid'][anchor:]=original_mask
        for k in x:torch.testing.assert_close(x[k],y[k],rtol=0,atol=0)
        self.assertFalse(x['end_valid'][:,-1].any());self.assertFalse(x['end_xy'][:,-1].any())
    def test_real_history_available_and_unselected_excluded(self):
        d=self.dataset;rows=np.array([150]);x,_=d.batch(rows,'cpu');anchor=d.targets['anchor'][150]
        selected=set((anchor-d.offsets[150][d.offsets[150]>=0]).tolist());unselected=[i for i in range(max(0,anchor-79),anchor) if i not in selected]
        original=d.endpoints['end_xy'][unselected].copy()
        try:
            d.endpoints['end_xy'][unselected]=.987;y,_=d.batch(rows,'cpu')
        finally:d.endpoints['end_xy'][unselected]=original
        torch.testing.assert_close(x['end_xy'],y['end_xy'],rtol=0,atol=0)
        j=int(torch.nonzero(x['end_valid'][0])[0]);event=anchor-int(d.offsets[150,j]);old=d.endpoints['end_xy'][event].copy()
        try:
            d.endpoints['end_xy'][event]=[.123,.456];z,_=d.batch(rows,'cpu')
        finally:d.endpoints['end_xy'][event]=old
        self.assertFalse(torch.equal(x['end_xy'],z['end_xy']))
    def test_direction_padding_and_unknown(self):
        inputs={'team':np.array([1,2,1,1])};targets={'anchor':np.array([3]),'match_start':np.array([0])}
        end={'end_xy':np.array([[.1,.2],[.3,.4],[.5,.6],[.8,.9]],np.float32),'end_valid':np.ones(4,bool)}
        offsets=np.array([[-1,3,2,1,0]]);valid=offsets>=0
        xy,mask=endpoint_batch(inputs,targets,end,np.array([0]),valid,offsets)
        np.testing.assert_allclose(xy[0],[[0,0],[.1,.2],[.7,.6],[.5,.6],[0,0]],atol=1e-7);np.testing.assert_array_equal(mask,[[False,True,True,True,False]])
        inputs['team'][1]=0;xy,mask=endpoint_batch(inputs,targets,end,np.array([0]),valid,offsets);self.assertFalse(mask[0,2]);self.assertFalse(xy[0,2].any())
    def test_assist_keypass_keep_pass(self):
        event=dict(eventId=8,subEventId=85,matchPeriod='1H',eventSec=12,positions=[dict(x=25,y=40),dict(x=75,y=60)],teamId=1,playerId=5,tags=[])
        base=clean_event(event);end,known=clean_endpoint(event,C['extension'])
        for tags in [[{'id':301}],[{'id':302}],[{'id':301},{'id':302}]]:
            changed=dict(event,tags=tags);self.assertEqual(clean_event(changed),base);self.assertEqual(clean_event(changed)['event'],8);self.assertEqual(clean_event(changed)['event']-1,7)
            xy,valid=clean_endpoint(changed,C['extension']);np.testing.assert_array_equal(xy,end);self.assertEqual(valid,known)
    def test_missing_and_placeholder(self):
        e=dict(eventId=8,subEventId=85,positions=[dict(x=25,y=40)])
        xy,valid=clean_endpoint(e,C['extension']);self.assertFalse(valid);self.assertFalse(xy.any())
        e=dict(eventId=10,subEventId=100,positions=[dict(x=80,y=40),dict(x=100,y=100)])
        xy,valid=clean_endpoint(e,C['extension']);self.assertFalse(valid);self.assertFalse(xy.any())
    def test_same_targets_selection_and_order(self):
        for split in ['train','validation']:
            for seed in C['extension']['seeds']:
                d=EndpointSplit(C,split,CONDITION,seed);b=Split(C,split,CONDITION,seed)
                for k in b.targets:np.testing.assert_array_equal(b.targets[k],d.targets[k])
                np.testing.assert_array_equal(b.offsets,d.offsets);np.testing.assert_array_equal(b.q,d.q)
                if split=='train':
                    order=rotate_orders(d.targets,seed,120);base_order=np.load(Path(C['extension']['base_runs'][str(seed)])/'sample_order.npy')
                    np.testing.assert_array_equal(order,base_order)
    def test_initialization_roundtrip_full_batch(self):
        deterministic(20260715);base=Predictor(C,CONDITION,'etp')
        deterministic(20260715);model=EndpointPredictor(C,CONDITION,'etp')
        for name,p in base.state_dict().items():torch.testing.assert_close(p,model.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in model.parameters())-sum(p.numel() for p in base.parameters()),192)
        x,y=self.dataset.batch(np.arange(256),'cpu');base.eval();model.eval();zero=copy.deepcopy(x);zero['end_xy'].zero_();zero['end_valid'].zero_()
        with torch.no_grad():a=base(zero);b=model(zero)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        model.train();opt=torch.optim.AdamW(model.parameters(),lr=.0027);value=loss(model(x),y,'etp');self.assertTrue(torch.isfinite(value));value.backward()
        for p in model.parameters():
            if p.grad is not None:self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue((model.endpoint.weight.grad!=0).any());opt.step();model.eval()
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0);other=EndpointPredictor(C,CONDITION,'etp');other.load_state_dict(torch.load(buffer,weights_only=True));other.eval()
        with torch.no_grad():a=model(x);b=other(x)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    def test_model_defense_anchor_and_padding(self):
        x,_=self.dataset.batch(np.array([0,1,150]),'cpu');model=EndpointPredictor(C,CONDITION,'etp').eval();altered=copy.deepcopy(x)
        forbid=(~x['valid'])|(x['order']==0);altered['end_xy'][forbid]=999;altered['end_valid'][forbid]=True
        with torch.no_grad():a=model(x);b=model(altered)
        for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)


if __name__=='__main__':
    import io
    stream=io.StringIO();suite=unittest.defaultTestLoader.loadTestsFromTestCase(EndpointTests);result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    (ROOT/'reports/acceptance.log').write_text(stream.getvalue());print(stream.getvalue())
    write_json(ROOT/'reports/acceptance.json',dict(status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,errors=len(result.errors),failures=len(result.failures),full_batch_device='cpu',test_evaluated=False))
    sys.exit(0 if result.wasSuccessful() else 1)
