#!/usr/bin/env python3
"""AICB fixed-global-batch TP/EP/PP/DP two-pool capacity and profile coverage."""
import argparse
import csv
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path
from inference_strategy_contract import RankWorkload, capacity_ledger, communication_contract
from strategy_compute_profiles import ComputeProfiles


def enumerate_candidates(model, *, world_size, global_batch, seq, tps=(1,2,4),
                         eps=(1,2,4,8,16,32), pps=(1,2,4), microbatch_sizes=(None,),
                         **capacity_options):
    accepted=[]; rejected=[]
    for tp,ep,pp,mb in itertools.product(tps,eps,pps,microbatch_sizes):
        params=dict(world_size=world_size,global_batch=global_batch,seq=seq,
                    tp=tp,ep=ep,pp=pp,microbatch_size=mb)
        try:
            w=RankWorkload.build(model,**params)
        except ValueError as exc:
            rejected.append(dict(candidate=params,reason=str(exc)))
            continue
        ledger=capacity_ledger(model,w,**capacity_options)
        accepted.append(dict(workload=w,capacity=ledger,capacity_options=dict(capacity_options),
                             communication=communication_contract(model,w)))
    return accepted,rejected


def write_csv(path,rows,empty_fields):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else empty_fields)
        writer.writeheader();writer.writerows(rows)


def csv_ints(value):
    try:
        result=tuple(int(x) for x in value.split(','))
        if not result or any(x<=0 for x in result):raise ValueError()
        return tuple(dict.fromkeys(result))
    except ValueError:
        raise argparse.ArgumentTypeError('comma-separated positive integers required')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--profiles',type=Path,help='optional V2 expert microbenchmark CSV, not full-layer timings')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--global-batch',type=int,default=32)
    ap.add_argument('--world-size',type=int,default=32)
    ap.add_argument('--seq',type=int,default=256)
    ap.add_argument('--candidate-tp',type=csv_ints,default=(1,2,4))
    ap.add_argument('--candidate-ep',type=csv_ints,default=(1,2,4,8,16,32))
    ap.add_argument('--candidate-pp',type=csv_ints,default=(1,2,4))
    ap.add_argument('--microbatch-sizes',type=csv_ints,help='default: one complete replica batch')
    ap.add_argument('--hbm-gib',type=float,default=40)
    ap.add_argument('--hbf-gib',type=float,default=128)
    ap.add_argument('--utilization',type=float,default=.9)
    ap.add_argument('--reserve-gib',type=float,default=2)
    args=ap.parse_args();model=json.loads(args.config.read_text())
    profiles=None
    if args.profiles:
        supported=dict(hidden_size=4096,moe_intermediate_size=1536,num_experts=128,
                       num_experts_per_tok=8,num_hidden_layers=94,num_attention_heads=64,
                       num_key_value_heads=4,head_dim=128,torch_dtype='bfloat16',model_type='qwen3_moe')
        if any(model.get(k)!=v for k,v in supported.items()):
            ap.error('V2 expert profiles only cover the pinned Qwen3 BF16 structure')
        profiles=ComputeProfiles(args.profiles)
    options=dict(hbm_gib=args.hbm_gib,hbf_gib=args.hbf_gib,
                 utilization=args.utilization,reserve_gib=args.reserve_gib)
    candidates,rejected=enumerate_candidates(model,world_size=args.world_size,
        global_batch=args.global_batch,seq=args.seq,tps=args.candidate_tp,
        eps=args.candidate_ep,pps=args.candidate_pp,
        microbatch_sizes=args.microbatch_sizes or (None,),**options)
    ledgers=[];shapes=[];coverage=[];comms=[]
    for candidate in candidates:
        w=candidate['workload'];ledger=candidate['capacity']
        ledgers.append(ledger);comms.append(candidate['communication'])
        shapes.append(dict(strategy=w.tag(),**asdict(w),
            feasible=ledger['feasible'],q_min_by_stage=json.dumps(ledger['q_min_by_stage']),
            max_all_hbm_gib=ledger['all_hbm_peak_bytes']/1024**3))
        for stage in ledger['stages']:
            for latency in (100,300,500,1000):
                q=stage['q_min'];status='UNKNOWN';estimate=None;method=''
                if q is None:status='CAPACITY_INFEASIBLE'
                elif profiles:
                    placement='full' if q in (0,w.experts) else ('quarter' if q*4==w.experts else None)
                    if placement:
                        result=profiles.query(tp=w.tp,m=w.tokens_per_expert,e=w.experts,
                            latency=100 if q==0 else latency,placement=placement)
                        if result['status']=='SUPPORTED':
                            status='COVERED_MICROBENCH_ONLY';estimate=result['time_us'];method=result.get('method','')
                coverage.append(dict(strategy=w.tag(),pp_rank=stage['pp_rank'],
                    local_layers=stage['local_layers'],TP=w.tp,PP=w.pp,M=w.tokens_per_expert,E=w.experts,
                    microbatch_size=w.microbatch_size,num_microbatches=w.num_microbatches,
                    latency=latency,q_min=q,status=status,microbench_us=estimate,method=method,
                    full_layer_status='UNKNOWN',ranking_eligible=False))
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('output must be new or empty; preserve earlier results')
    args.output.mkdir(parents=True,exist_ok=True)
    metadata=dict(schema='capacity_strategy_pp_v2',global_batch=args.global_batch,seq=args.seq,
        world_size=args.world_size,capacity_options=options,
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        profiles_sha256=hashlib.sha256(args.profiles.read_bytes()).hexdigest() if args.profiles else None,
        code_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                    for name in ('capacity_strategy_search.py','inference_strategy_contract.py','estimate_prefill_memory.py')},
        capacity_ledger=ledgers,rejected_candidates=rejected,ranking_enabled=False,
        reason='capacity/coverage only; audited full-layer profiles and explicit PP timing required')
    (args.output/'capacity_ledger.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.output/'communication_contracts.json').write_text(json.dumps(comms,indent=2)+'\n')
    write_csv(args.output/'candidate_shapes.csv',shapes,['strategy','feasible'])
    write_csv(args.output/'coverage_matrix.csv',coverage,['strategy','status'])
    print(json.dumps(dict(candidates=len(candidates),feasible=sum(x['feasible'] for x in ledgers),
        rejected=len(rejected),pp_values=sorted(set(x['workload'].pp for x in candidates)),
        output=str(args.output),ranking_enabled=False)))


if __name__=='__main__':main()
