#!/usr/bin/env python3
"""Evaluate AICB forward PP bundles with SimAI stage service and explicit PP costs.

This adapter deliberately does not use SimAI's training PP summary formula.
It runs each distinct stage workload in SimAI analytical, then schedules forward
microbatches. PP boundary costs are user-supplied evidence or an explicit effective
bandwidth model, never an implicit zero or an assumed calibrated hardware latency.
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
import uuid
from pathlib import Path
from inference_pipeline import schedule_forward


def digest(data):
    return hashlib.sha256(data).hexdigest()


def workload_signature(bundle):
    return digest(json.dumps(bundle['workload'],sort_keys=True).encode())


def parse_simai_csv(raw):
    # SimAI appends a differently shaped per-layer CSV after the two-row summary.
    lines=[line for line in raw.splitlines() if line.strip()]
    if len(lines)<2 or not lines[0].startswith('File name,'):
        raise ValueError('missing SimAI summary header')
    rows=list(csv.DictReader(lines[:2],skipinitialspace=True))
    if len(rows)!=1:raise ValueError('expected one SimAI summary row')
    row={k.strip():v.strip() for k,v in rows[0].items()}
    fields={'compute_us':'total comp','exposed_comm_us':'total exposed comm',
            'total_us':'Total time','tp_comm_us':'Expose TP comm','ep_comm_us':'Expose_EP_comm',
            'dp_comm_us':'Expose DP comm','dp_ep_comm_us':'Expose DP_EP comm',
            'pp_comm_us':'Expose_PP_comm','bubble_us':'bubble time'}
    out={name:float(row[key].split()[0]) for name,key in fields.items()}
    if any(not math.isfinite(v) or v<0 for v in out.values()):
        raise ValueError('invalid/nonfinite SimAI times')
    if any(out[k]!=0 for k in ('dp_comm_us','dp_ep_comm_us','pp_comm_us','bubble_us')):
        raise ValueError('isolated inference stage contains training/PP contributions')
    if abs(out['total_us']-out['compute_us']-out['exposed_comm_us'])>2:
        raise ValueError('SimAI total does not reconcile')
    if abs(out['exposed_comm_us']-out['tp_comm_us']-out['ep_comm_us'])>2:
        raise ValueError('unaccounted communication')
    return out


def run_simai_stage(text,world_size,root,out,*,gpus_per_node=8,nvlink=360,nic=48.5):
    root=Path(root).resolve();out=Path(out).resolve()
    binary=root/'bin/SimAI_analytical'
    if not binary.is_file():raise FileNotFoundError(str(binary))
    out.mkdir(parents=True,exist_ok=False)
    workload=out/'workload.txt';workload.write_text(text)
    prefix='aicb_pp_'+uuid.uuid4().hex+'_'
    command=[str(binary),'-w',str(workload),'-g',str(world_size),
        '-g_p_s',str(min(gpus_per_node,world_size)),'-n_p_s','8','-nv',str(nvlink),
        '-nic',str(nic),'-g_type','A100','-ep_o','0','-tp_o','0','-dp_o','0',
        '-pp_o','1','-r',prefix]
    binary_hash=digest(binary.read_bytes())
    with (out/'simai.log').open('w') as log:
        process=subprocess.run(command,cwd=str(root),stdout=log,stderr=subprocess.STDOUT,timeout=120)
    (out/'exit_code').write_text(str(process.returncode)+'\n')
    if process.returncode:raise RuntimeError('SimAI failed: '+str(out/'simai.log'))
    source=root/'results'/(prefix+'EndToEnd.csv')
    raw=source.read_text();(out/'EndToEnd.csv').write_text(raw)
    result=parse_simai_csv(raw)
    if digest(binary.read_bytes())!=binary_hash:raise ValueError('SimAI binary changed during run')
    result.update(workload_sha256=digest(text.encode()),output_sha256=digest(raw.encode()),
        simai_binary_sha256=binary_hash,command=command,cwd=str(root),exit_code=0,
        native_output=str(source),csv_rounding_us=1)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def effective_bandwidth_costs(bundle,bandwidth,latency):
    """Bandwidth is aggregate GB/s per replica/boundary, not per GPU or a product spec."""
    if not math.isfinite(bandwidth) or bandwidth<=0 or not math.isfinite(latency) or latency<0:
        raise ValueError('positive effective bandwidth and nonnegative latency required')
    payload=bundle['communication']['pp_payload_bytes_per_replica_per_boundary']
    return dict(schema='aicb_pp_costs_v1',workload_sha256=workload_signature(bundle),
        kind='analytical_effective_bandwidth',source='explicit user network parameters',
        aggregate_bytes_per_boundary_per_microbatch=payload,
        bandwidth_gbytes_per_sec=bandwidth,latency_us=latency,
        time_us=[latency+payload/(bandwidth*1000)]*(bundle['workload']['pp']-1),
        assumptions=['aggregate per-replica boundary bandwidth, GB=10^9 bytes',
                     'no shared-link contention across independent boundaries or DP replicas',
                     'not an automatically calibrated SimAI point-to-point measurement'])


def validate_bundle(bundle):
    if bundle.get('schema')!='aicb_forward_pipeline_v1':raise ValueError('unsupported bundle schema')
    w=bundle['workload'];stages=bundle['stages']
    if len(stages)!=w['pp'] or sum(w['stage_layers'])<=0:raise ValueError('invalid stage coverage')
    if w['dp']*w['pp']*w['tp']*w['ep']!=w['world_size']:raise ValueError('rank count mismatch')
    if w['dp']*w['microbatch_size']*w['num_microbatches']!=w['global_batch']:
        raise ValueError('global workload not conserved')
    start=0
    for index,stage in enumerate(stages):
        meta=stage['manifest'];text=stage['workload'];n=w['stage_layers'][index]
        if stage['pp_rank']!=index or meta['layer_range']!=[start,start+n]:
            raise ValueError('noncontiguous/mismatched stage layers')
        if digest(text.encode())!=meta['workload_sha256']:raise ValueError('workload hash mismatch')
        if int(text.splitlines()[1])!=len(text.splitlines())-2:raise ValueError('workload row count mismatch')
        if meta['placement_q'] not in bundle['capacity']['stages'][index]['feasible_q']:
            raise ValueError('infeasible stage placement')
        start+=n


def evaluate(bundle,*,simai_root,output,pp_costs=None,diagnostic=False,
             gpus_per_node=8,nvlink=360,nic=48.5):
    validate_bundle(bundle)
    w=bundle['workload'];p=w['pp']
    if p>1:
        if pp_costs is None:raise ValueError('UNKNOWN: explicit PP boundary costs required')
        if pp_costs.get('schema')!='aicb_pp_costs_v1' or not pp_costs.get('source'):
            raise ValueError('PP costs require schema and provenance source')
        if pp_costs.get('workload_sha256')!=workload_signature(bundle):
            raise ValueError('PP costs do not match workload signature')
        transfers=pp_costs['time_us']
    else:transfers=[]
    # Validate cost dimensions and numbers before invoking any backend.
    schedule_forward([0]*p,1,transfers)
    if type(gpus_per_node) is not int or gpus_per_node<=0:
        raise ValueError('positive gpus_per_node required')
    stage_world=w['world_size']//p
    if stage_world%min(stage_world,gpus_per_node):raise ValueError('unsupported partial-node stage layout')
    width=w['tp']*w['ep']
    if width%gpus_per_node and gpus_per_node%width:
        raise ValueError('stage isolation cannot preserve TP/EP node alignment')
    if any(not math.isfinite(x) or x<=0 for x in (nvlink,nic)):
        raise ValueError('positive network bandwidth required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n')
    if pp_costs:(output/'pp_costs.json').write_text(json.dumps(pp_costs,indent=2)+'\n')
    cache={};stage_results=[]
    for s in bundle['stages']:
        key=s['manifest']['workload_sha256']
        if key not in cache:
            cache[key]=run_simai_stage(s['workload'],stage_world,simai_root,
                output/('stage_'+str(s['pp_rank'])),gpus_per_node=gpus_per_node,nvlink=nvlink,nic=nic)
        stage_results.append(dict(cache[key],pp_rank=s['pp_rank']))
    schedule=schedule_forward([r['total_us'] for r in stage_results],w['num_microbatches'],transfers)
    result=dict(schema='aicb_forward_pipeline_result_v1',status='COMPLETE',
        stage_results=stage_results,pp_costs=pp_costs,schedule=schedule,
        total_us=schedule['total_us'],global_tokens=bundle['global_tokens'],
        throughput_tokens_per_sec=bundle['global_tokens']*1e6/schedule['total_us'] if schedule['total_us'] else None,
        diagnostic=diagnostic,ranking_scope='declared isolated-stage and explicit PP network model',
        physical_pp_validation=False,full_service_TTFT_supported=False,
        assumptions=bundle['assumptions'],
        bundle_sha256=digest((output/'bundle.json').read_bytes()),
        code_sha256={name:digest((Path(__file__).parent/name).read_bytes()) for name in
                    ('evaluate_inference_pipeline.py','inference_pipeline.py')})
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--simai-root',type=Path,default=Path(__file__).resolve().parent.parent)
    group=ap.add_mutually_exclusive_group()
    group.add_argument('--pp-costs',type=Path)
    group.add_argument('--pp-effective-bandwidth-gbytes-per-sec',type=float)
    ap.add_argument('--pp-boundary-latency-us',type=float)
    ap.add_argument('--gpus-per-node',type=int,default=8)
    ap.add_argument('--nvlink-bandwidth',type=float,default=360)
    ap.add_argument('--nic-bandwidth',type=float,default=48.5)
    ap.add_argument('--diagnostic',action='store_true')
    args=ap.parse_args();bundle=json.loads(args.bundle.read_text());costs=None
    if args.pp_costs:costs=json.loads(args.pp_costs.read_text())
    if args.pp_effective_bandwidth_gbytes_per_sec is not None:
        if args.pp_boundary_latency_us is None:ap.error('specify --pp-boundary-latency-us explicitly')
        costs=effective_bandwidth_costs(bundle,args.pp_effective_bandwidth_gbytes_per_sec,args.pp_boundary_latency_us)
    result=evaluate(bundle,simai_root=args.simai_root,output=args.output,pp_costs=costs,
        diagnostic=args.diagnostic,gpus_per_node=args.gpus_per_node,
        nvlink=args.nvlink_bandwidth,nic=args.nic_bandwidth)
    print(json.dumps({k:result[k] for k in ('status','total_us','global_tokens','diagnostic','physical_pp_validation')}))


if __name__=='__main__':main()
