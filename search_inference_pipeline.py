#!/usr/bin/env python3
"""Finite-domain HBF inference search, including PP and microbatch scheduling.

Missing exact full-layer profiles remain UNKNOWN and are exported as sampling
requests. A best completed candidate is never described as global optimum when
coverage is incomplete. Default min-q placement is per pipeline stage.
"""
import argparse
import hashlib
import json
from pathlib import Path
from capacity_strategy_search import enumerate_candidates,csv_ints,write_csv
from build_strategy_body_workload import build_pipeline
from evaluate_inference_pipeline import evaluate,effective_bandwidth_costs


def find_profile(model,w,q,latency,profiles):
    expected=dict(TP=w.tp,M=w.tokens_per_expert,E=w.experts,rank_batch=w.rank_batch,
                  seq=w.seq,q=q,hbf_latency=latency)
    model_hash=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
    matches=[p for p in profiles if p.get('shape_placement')==expected and
             p.get('model_sha256')==model_hash and p.get('status')=='SUPPORTED' and
             p.get('mode','reference')=='reference']
    if not matches:return None,dict(shape_placement=expected,model_sha256=model_hash,
                                    reason='missing exact full-layer profile')
    timings={json.dumps(p.get('stage_times_us'),sort_keys=True) for p in matches}
    if len(timings)!=1:raise ValueError('conflicting full-layer labels for '+str(expected))
    return matches[0],None


def search(model,candidates,profiles,*,output,simai_root,hbf_latency=500,
           pp_bandwidth=None,pp_latency=None,dry_run=False,diagnostic=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    results=[];missing=[]
    for candidate in candidates:
        w=candidate['workload'];cap=candidate['capacity']
        row=dict(strategy=w.tag(),TP=w.tp,EP=w.ep,PP=w.pp,DP=w.dp,
            microbatch_size=w.microbatch_size,num_microbatches=w.num_microbatches,
            q_by_stage=cap['q_min_by_stage'],status='CAPACITY_INFEASIBLE',total_us=None,reason='')
        if cap['feasible']:
            selected=[]
            for rank,q in enumerate(cap['q_min_by_stage']):
                try:
                    profile,gap=find_profile(model,w,q,hbf_latency,profiles)
                except ValueError as exc:
                    profile,gap=None,dict(reason=str(exc))
                if gap:missing.append(dict(strategy=w.tag(),pp_rank=rank,**gap))
                selected.append(profile)
            if any(p is None for p in selected):
                row.update(status='UNKNOWN',reason='missing exact stage profiles; see missing_profiles.json')
            else:
                try:
                    bundle=build_pipeline(model,w,selected,q_by_stage=cap['q_min_by_stage'],
                        hbf_latency=hbf_latency,**candidate['capacity_options'])
                    costs=None
                    if w.pp>1:
                        if pp_bandwidth is None or pp_latency is None:
                            raise ValueError('UNKNOWN: explicit PP network parameters required')
                        costs=effective_bandwidth_costs(bundle,pp_bandwidth,pp_latency)
                    if dry_run:
                        path=output/(w.tag()+'.bundle.json')
                        path.write_text(json.dumps(bundle,indent=2)+'\n')
                        row.update(status='READY',reason='dry run; no timing result')
                    else:
                        result=evaluate(bundle,simai_root=simai_root,output=output/w.tag(),
                                        pp_costs=costs,diagnostic=diagnostic)
                        row.update(status='COMPLETE',total_us=result['total_us'],reason='')
                except (ValueError,FileNotFoundError) as exc:
                    row.update(status='UNKNOWN',reason=str(exc))
        results.append(row)
    complete=[r for r in results if r['status']=='COMPLETE']
    unknown=[r for r in results if r['status'] in ('UNKNOWN','READY')]
    summary=dict(schema='aicb_pp_search_v1',results=results,
        best_completed=min(complete,key=lambda r:r['total_us']) if complete else None,
        complete_candidates=len(complete),unresolved_candidates=len(unknown),
        search_domain_fully_evaluated=not unknown and bool(complete),
        optimum_scope='covered finite candidates; min-q per-stage placement; explicit network model',
        diagnostic=diagnostic,physical_pp_validation=False)
    (output/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'missing_profiles.json').write_text(json.dumps(missing,indent=2)+'\n')
    write_csv(output/'results.csv',results,['strategy','status','total_us'])
    return summary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--profile-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--simai-root',type=Path,default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--world-size',type=int,default=32);ap.add_argument('--global-batch',type=int,default=32)
    ap.add_argument('--seq',type=int,default=256)
    ap.add_argument('--candidate-tp',type=csv_ints,default=(1,2,4))
    ap.add_argument('--candidate-ep',type=csv_ints,default=(1,2,4,8,16,32))
    ap.add_argument('--candidate-pp',type=csv_ints,default=(1,2,4))
    ap.add_argument('--microbatch-sizes',type=csv_ints)
    ap.add_argument('--hbm-gib',type=float,default=40);ap.add_argument('--hbf-gib',type=float,default=128)
    ap.add_argument('--reserve-gib',type=float,default=2);ap.add_argument('--utilization',type=float,default=.9)
    ap.add_argument('--hbf-latency',type=int,default=500)
    ap.add_argument('--pp-effective-bandwidth-gbytes-per-sec',type=float)
    ap.add_argument('--pp-boundary-latency-us',type=float)
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--diagnostic',action='store_true')
    a=ap.parse_args();model=json.loads(a.config.read_text());profiles=[]
    if not a.profile_dir.is_dir():ap.error('--profile-dir must exist (an empty directory is allowed)')
    for path in sorted(a.profile_dir.glob('*.json')):
        obj=json.loads(path.read_text());profile=obj.get('profile',obj)
        if 'shape_placement' in profile:profiles.append(profile)
    options=dict(hbm_gib=a.hbm_gib,hbf_gib=a.hbf_gib,reserve_gib=a.reserve_gib,utilization=a.utilization)
    candidates,rejected=enumerate_candidates(model,world_size=a.world_size,global_batch=a.global_batch,
        seq=a.seq,tps=a.candidate_tp,eps=a.candidate_ep,pps=a.candidate_pp,
        microbatch_sizes=a.microbatch_sizes or (None,),**options)
    for c in candidates:c['capacity_options']=options
    result=search(model,candidates,profiles,output=a.output,simai_root=a.simai_root,
        hbf_latency=a.hbf_latency,pp_bandwidth=a.pp_effective_bandwidth_gbytes_per_sec,
        pp_latency=a.pp_boundary_latency_us,dry_run=a.dry_run,diagnostic=a.diagnostic)
    (a.output/'rejected_candidates.json').write_text(json.dumps(rejected,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('best_completed','complete_candidates','unresolved_candidates','search_domain_fully_evaluated')}))


if __name__=='__main__':main()
