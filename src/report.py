"""Stage reports; no test-set access or automatic choice of a winning C2 rule."""
import argparse
import itertools
import math
from pathlib import Path
import numpy as np
from data import read_config,write_json
from evaluation import aligned,bootstrap_weights,paired_bootstrap,match_table


def information_decomposition(values):
    groups='CXPT';full=frozenset(groups);out={}
    for g in groups:
        others=[x for x in groups if x!=g];phi=0.
        for size in range(4):
            for subset in itertools.combinations(others,size):
                s=frozenset(subset);weight=math.factorial(size)*math.factorial(3-size)/math.factorial(4)
                phi+=weight*(values[s|{g}]-values[s])
        out[g]=dict(add_to_core=values[frozenset([g])]-values[frozenset()],remove_from_full=values[full]-values[full-{g}],shapley=phi)
    interactions={}
    for a,b in itertools.combinations(groups,2):
        rest=[g for g in groups if g not in [a,b]];effects=[]
        for flags in itertools.product([False,True],repeat=2):
            s=frozenset(g for g,keep in zip(rest,flags) if keep)
            effects.append(values[s|{a,b}]-values[s|{a}]-values[s|{b}]+values[s])
        interactions[a+b]=dict(by_background=effects,uniform_background_mean=float(np.mean(effects)))
    return dict(groups=out,pair_interactions=interactions,utility_gain=values[full]-values[frozenset()],shapley_sum=sum(v['shapley'] for v in out.values()))


def main(c):
    root=Path(c['root']);report=root/'reports';jobs=read_config(report/'completed_runs.json')
    index={(j['task'],j['condition'],j['seed']):root/'runs'/j['run_id'] for j in jobs}
    phase_b={}
    for task in c['tasks']:
        utilities=['team_nll'] if task=='team_only' else ['etp_loss','event_ce','time_loss','position_loss']
        phase_b[task]={}
        for utility in utilities:
            values={}
            for name,condition in c['conditions'].items():
                if not name.startswith('B_'):continue
                metrics=read_config(index[task,name,20260715]/'best_metrics.json')['metrics']['per_record']
                values[frozenset(condition['groups'])]=-metrics[utility]
            phase_b[task][utility]=information_decomposition(values)
    write_json(report/'stage_B_exploratory.json',dict(seed=20260715,view='per_record_validation',training_randomness_inference=False,results=phase_b))
    thresholds=read_config(report/'thresholds.json');primary=[];weights=None;match_ids=None
    for task in c['tasks']:
        for contrast in c['statistics']['comparisons']:
            a_tables=[];b_tables=[];budget=[]
            for seed in c['training']['seeds']:
                ar=index[task,contrast['candidate'],seed];br=index[task,contrast['reference'],seed]
                with np.load(ar/'best_validation_predictions.npz') as az,np.load(br/'best_validation_predictions.npz') as bz:
                    at={k[7:]:az[k] for k in az.files if k.startswith('target_')};bt={k[7:]:bz[k] for k in bz.files if k.startswith('target_')};aligned(at,bt)
                with np.load(ar/'best_match_statistics.npz') as az,np.load(br/'best_match_statistics.npz') as bz:
                    np.testing.assert_array_equal(az['match_id'],bz['match_id'])
                    if match_ids is None:
                        match_ids=az['match_id'];weights=bootstrap_weights(len(match_ids),10000,c['statistics']['bootstrap']['seed']);np.savez(report/'bootstrap_weights.npz',match_id=match_ids,weights=weights)
                    np.testing.assert_array_equal(match_ids,az['match_id'])
                    a_tables.append(az['duel_corrected']);b_tables.append(bz['duel_corrected'])
                budget.append(any(read_config(d/'status.json')['stop_reason']=='budget_exhausted' for d in [ar,br]))
            result=paired_bootstrap(np.stack(a_tables),np.stack(b_tables),weights,task,24)
            metric=c['statistics']['primary_metrics'][task];p=result[metric];minimum=thresholds['tasks'][task]['minimum_absolute_improvement'];labels=[]
            if -p['mean']<minimum:labels.append('效应小于关注阈值')
            if p['corrected_ci'][1]>=0 or sum(x<0 for x in p['per_seed'])<2:labels.append('证据不足')
            if task=='etp' and (result['event_macro_f1']['mean']<-.005 or result['time_mae_seconds']['mean']>.02 or result['position_error_metres']['mean']>.25):labels.append('存在退步')
            if any(budget):labels.append('预算耗尽')
            primary.append(dict(task=task,contrast=contrast,view='duel_corrected_validation',metrics=result,minimum_primary_effect=minimum,labels=labels,eligible_for_external_validation=not labels))
    write_json(report/'primary_confirmatory.json',dict(family_size=24,bootstrap_replicates=10000,conditional_on_trained_models=True,results=primary))
    secondary=[]
    for task in c['tasks']:
        for rule in c['selection']['rules']:
            for reference in ['C1_16',f'C2_{rule}_control']:
                a=[];b=[];coverage=[]
                for seed in c['training']['seeds']:
                    selection_seed=c['selection']['seed_map'][str(seed)]
                    with np.load(root/'data'/f'validation_{rule}_{selection_seed}.npz') as f:q=f['q']>0
                    coverage.append(float(q.mean()))
                    for name,dest in [(f'C2_{rule}_relation',a),(reference,b)]:
                        with np.load(index[task,name,seed]/'best_validation_predictions.npz') as f:
                            target={k[7:]:f[k] for k in f.files if k.startswith('target_')};pred={k[5:]:f[k] for k in f.files if k.startswith('pred_')}
                            ids,table=match_table(pred,target,task,'duel_corrected',q);np.testing.assert_array_equal(ids,match_ids);dest.append(table)
                secondary.append(dict(task=task,rule=rule,reference=reference,coverage_per_seed=coverage,metrics=paired_bootstrap(np.stack(a),np.stack(b),weights,task,24),interpretation='prespecified secondary q>0 paired analysis; does not replace primary population'))
    write_json(report/'C2_q_positive_secondary.json',secondary)
    summaries={}
    for stage in ['B','C1','C2','D']:
        summaries[stage]={}
        for name in c['stages'][stage]['conditions']:
            summaries[stage][name]={}
            for task in c['tasks']:
                entries=[]
                for seed in c['training']['seeds']:
                    key=(task,name,seed)
                    if key in index:
                        path=index[key];entries.append(dict(seed=seed,path=str(path),best=read_config(path/'best_metrics.json'),status=read_config(path/'status.json'),cost=read_config(path/'cost.json')))
                summaries[stage][name][task]=dict(exploratory=entries[0],confirmatory=entries if len(entries)==3 else None)
    write_json(report/'stage_results.json',summaries)
    write_json(report/'external_gate.json',dict(status='review_eligible_conditions',eligible=[dict(task=p['task'],contrast=p['contrast']) for p in primary if p['eligible_for_external_validation'] and p['contrast']['id'] not in ['player_information','team_information','possession_representation']],test_locked=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();main(read_config(a.config))
