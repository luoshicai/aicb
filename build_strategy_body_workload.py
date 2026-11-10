#!/usr/bin/env python3
"""Audited prefill body bridge; PP uses per-stage workloads plus explicit schedule."""
import argparse,hashlib,json,math
from dataclasses import asdict
from pathlib import Path
from inference_strategy_contract import RankWorkload,capacity_ledger,communication_contract
STAGES=('inputLayerNorm','selfAttn','postAttentionLayerNorm','moeRouter','moeExperts')
NAMES=('attention_norm','attention_layer','moe_norm','moe_route','moe_expert')

def build(model,w,profile,*,q,hbf_latency,hbm_gib=40,hbf_gib=128,reserve_gib=2,utilization=.9,_pp_rank=None):
    if _pp_rank is None and (w.pp!=1 or w.num_microbatches!=1):
        raise ValueError('PP/multiple microbatches require build_pipeline; a single workload would lose scheduling')
    pp_rank=0 if _pp_rank is None else _pp_rank
    layer_start,layer_end=w.stage_range(pp_rank)
    if w.tp>1:
        audit=profile.get('tp_reduction_audit',{})
        evidence=Path(audit.get('file','/nonexistent'))
        if not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest()!=audit.get('sha256'):
            raise ValueError('TP reduction audit missing or changed')
        if json.loads(evidence.read_text()).get('status')!='PASS':raise ValueError('TP reduction audit did not pass')
        if profile.get('execution_boundary_version')!='tp_reduction_v2':raise ValueError('residual/TP boundary not supported')
    expected=dict(TP=w.tp,M=w.tokens_per_expert,E=w.experts,rank_batch=w.rank_batch,seq=w.seq,q=q,hbf_latency=hbf_latency)
    if profile.get('shape_placement')!=expected:raise ValueError('profile shape/placement mismatch')
    model_hash=hashlib.sha256(json.dumps(model,sort_keys=True).encode()).hexdigest()
    if profile.get('model_sha256')!=model_hash:raise ValueError('model signature mismatch')
    if profile.get('status')!='SUPPORTED' or not profile.get('sources'):
        raise ValueError('missing audited stage sources')
    times=profile.get('stage_times_us',{})
    if set(times)!=set(STAGES) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in times.values()):
        raise ValueError('all five finite positive compute stages required')
    ledger=capacity_ledger(model,w,hbm_gib=hbm_gib,hbf_gib=hbf_gib,reserve_gib=reserve_gib,utilization=utilization)['stages'][pp_rank]
    if q not in ledger['feasible_q']:raise ValueError('placement fails capacity checks')
    stage_world=w.world_size//w.pp
    header=f'HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: {w.tp} ep: {w.ep} pp: 1 all_gpus: {stage_world} mode: 1 vpp: 1 ga: 1 checkpoints: 0 checkpoint_initiates: 0 pp_comm: 0'
    rows=[]
    for layer in range(layer_start,layer_end):
        for stage,name in zip(STAGES,NAMES):
            comm='ALLTOALL_EP' if stage in ('moeRouter','moeExperts') and w.ep>1 else 'NONE'
            if w.tp>1 and stage in ('selfAttn','moeExperts'):comm='ALLREDUCE'
            payload=w.rank_tokens*model['num_experts_per_tok']*model['hidden_size']*2 if comm!='NONE' else 0
            if comm=='ALLREDUCE':
                payload=(w.rank_tokens if stage=='selfAttn' else w.replica_tokens)*model['hidden_size']*2
            # Input workload compute is nanoseconds; analytical CSV output is us.
            ns=max(1,round(times[stage]*1000))
            rows.append('\t'.join(map(str,[name,-1,ns,comm,payload,0,'NONE',0,0,'NONE',0,100])))
            if w.tp>1 and stage=='moeExperts' and w.ep>1:
                rows.append('\t'.join(map(str,['moe_combine',-1,0,'ALLTOALL_EP',w.rank_tokens*model['num_experts_per_tok']*model['hidden_size']*2,0,'NONE',0,0,'NONE',0,100])))
    text=header+'\n'+str(len(rows))+'\n'+'\n'.join(rows)+'\n'
    return text,dict(strategy=w.tag(),scope='controlled_94_layer_body_prefill',global_tokens=w.global_batch*w.seq,
        compute_input_unit='ns',stage_source_unit='us',layers=layer_end-layer_start,rows=len(rows),
        compute_sum_us=(layer_end-layer_start)*sum(times.values()),placement_q=q,
        pp_rank=pp_rank,layer_range=[layer_start,layer_end],stage_world_size=stage_world,
        workload_sha256=hashlib.sha256(text.encode()).hexdigest(),profile=profile,
        capacity_q_feasible=True,full_service_TTFT_supported=False)


def build_pipeline(model,w,profiles,*,q_by_stage,hbf_latency,**capacity_options):
    """Emit one audited microbatch workload per PP stage, never training PP math.

    TP/EP service times come from actual SimAI runs. Inter-stage transfers and
    forward scheduling are explicit in evaluate_inference_pipeline.py.
    """
    if len(profiles)!=w.pp or len(q_by_stage)!=w.pp:
        raise ValueError('one profile and placement per pipeline stage required')
    stages=[]
    for rank,(profile,q) in enumerate(zip(profiles,q_by_stage)):
        text,manifest=build(model,w,profile,q=q,hbf_latency=hbf_latency,
                            _pp_rank=rank,**capacity_options)
        stages.append(dict(pp_rank=rank,workload=text,manifest=manifest))
    return dict(schema='aicb_forward_pipeline_v1',workload=asdict(w),stages=stages,
        communication=communication_contract(model,w),
        capacity=capacity_ledger(model,w,**capacity_options),
        global_tokens=w.global_batch*w.seq,scope='controlled_transformer_body_prefill',
        schedule='forward_store_and_forward_no_pp_compute_overlap',
        tp_ep_service='SimAI analytical isolated per-stage workload',
        pp_transfer='required explicit boundary costs; no default zero',
        assumptions=['balanced identical DP replicas; no cross-replica contention',
                     'TP/EP intra-stage network contention not coupled across PP stages',
                     'body only; embedding/final head excluded from timing but budgeted in capacity'],
        full_service_TTFT_supported=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--profile',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--ep',type=int,required=True)
    ap.add_argument('--q',type=int,nargs='+',required=True);ap.add_argument('--hbf-latency',type=int,required=True)
    ap.add_argument('--tp',type=int,default=1);ap.add_argument('--pp',type=int,default=1)
    ap.add_argument('--microbatch-size',type=int)
    ap.add_argument('--global-batch',type=int,default=32);ap.add_argument('--world-size',type=int,default=32)
    ap.add_argument('--seq',type=int,default=256);ap.add_argument('--hbm-gib',type=float,default=40)
    ap.add_argument('--hbf-gib',type=float,default=128);ap.add_argument('--reserve-gib',type=float,default=2)
    a=ap.parse_args();model=json.loads(a.config.read_text());profile=json.loads(a.profile.read_text())
    w=RankWorkload.build(model,world_size=a.world_size,global_batch=a.global_batch,seq=a.seq,tp=a.tp,ep=a.ep,pp=a.pp,microbatch_size=a.microbatch_size)
    q=a.q*w.pp if len(a.q)==1 else a.q
    if len(q)!=w.pp:ap.error('--q must have one value or PP values')
    opts=dict(hbm_gib=a.hbm_gib,hbf_gib=a.hbf_gib,reserve_gib=a.reserve_gib)
    if w.pp>1 or w.num_microbatches>1:
        profiles=profile.get('stage_profiles',[profile]*w.pp)
        bundle=build_pipeline(model,w,profiles,q_by_stage=q,hbf_latency=a.hbf_latency,**opts)
        if a.output.exists():raise FileExistsError(str(a.output))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(bundle,indent=2)+'\n')
        return
    text,manifest=build(model,w,profile,q=q[0],hbf_latency=a.hbf_latency,**opts)
    if a.output.exists():raise FileExistsError('preserve existing workload: '+str(a.output))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    a.output.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__':main()
