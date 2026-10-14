"""Post-selection paired description; excluded from the original primary test family."""
import argparse
from pathlib import Path
import numpy as np
from endpoints import read_config,write_json
from evaluation import aligned,bootstrap_weights,paired_bootstrap


def compare(config):
    root=Path(config['root']);ext=config['extension'];new=[];old=[];matches=None;seeds=[]
    for seed in ext['seeds']:
        a=root/'runs'/f'endpoints_etp_s{seed}';b=Path(ext['base_runs'][str(seed)])
        states=[read_config(p/'status.json') for p in [a,b]]
        if any(s['status']!='complete' for s in states):raise ValueError('Incomplete paired run')
        with np.load(a/'best_validation_predictions.npz') as aa,np.load(b/'best_validation_predictions.npz') as bb:
            aligned({k[7:]:aa[k] for k in aa.files if k.startswith('target_')},{k[7:]:bb[k] for k in bb.files if k.startswith('target_')})
        with np.load(a/'best_match_statistics.npz') as aa,np.load(b/'best_match_statistics.npz') as bb:
            np.testing.assert_array_equal(aa['match_id'],bb['match_id'])
            if matches is None:matches=aa['match_id'].copy()
            np.testing.assert_array_equal(matches,aa['match_id'])
            new.append({k:aa[k].copy() for k in ['per_record','duel_corrected','nonpaired','q_positive']})
            old.append({k:bb[k].copy() for k in ['per_record','duel_corrected','nonpaired','q_positive']})
        seeds.append(dict(seed=seed,new_path=str(a),base_path=str(b),new=read_config(a/'best_metrics.json'),base=read_config(b/'best_metrics.json'),new_status=states[0],base_status=states[1],new_cost=read_config(a/'cost.json'),base_cost=read_config(b/'cost.json')))
    weights=bootstrap_weights(len(matches),10000,20261014);np.savez(root/'reports/bootstrap_weights.npz',match_id=matches,weights=weights)
    differences={}
    for view in new[0]:
        result=paired_bootstrap(np.stack([a[view] for a in new]),np.stack([a[view] for a in old]),weights,'etp',family_size=1)
        for item in result.values():del item['corrected_ci']
        differences[view]=result
    write_json(root/'reports/comparison.json',dict(base_condition=ext['base_condition'],additional_condition=ext['condition'],interpretation='exploratory follow-up selected using existing validation results; conditional intervals, not part of original 24 primary tests',difference='historical_endpoints_minus_base',primary_view='duel_corrected',parameter_increase=192,seeds=seeds,differences=differences,test_evaluated=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();compare(read_config(a.config))
