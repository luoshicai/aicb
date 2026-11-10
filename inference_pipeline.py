"""Deterministic forward-only PP scheduler, with explicit boundary service costs.

TP/EP collective time is already in each stage's service time. PP transfers
reserve both adjacent stages (no compute/PP overlap); all microbatch boundary
buffers are reserved by the capacity model. Balanced DP replicas run in parallel.
"""
import math


def schedule_forward(stage_times_us, num_microbatches, transfer_times_us):
    p=len(stage_times_us)
    if not p or type(num_microbatches) is not int or num_microbatches<=0:
        raise ValueError('positive stage and microbatch counts required')
    if len(transfer_times_us)!=p-1:
        raise ValueError('one explicit transfer cost per PP boundary required')
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0
           for x in list(stage_times_us)+list(transfer_times_us)):
        raise ValueError('finite nonnegative service times required')
    tasks={}
    for mb in range(num_microbatches):
        for s,t in enumerate(stage_times_us):
            deps=[]
            if s:deps.append(('send',mb,s-1))
            if mb:deps.append(('compute',mb-1,s))
            tasks[('compute',mb,s)]=(t,(s,),deps)
            if s<p-1:
                tasks[('send',mb,s)]=(transfer_times_us[s],(s,s+1),[('compute',mb,s)])
    ends={};available=[0.0]*p;events=[]
    while tasks:
        ready=[]
        for key,(duration,resources,deps) in tasks.items():
            if all(d in ends for d in deps):
                start=max([available[r] for r in resources]+[ends[d] for d in deps]+[0.0])
                # FIFO mb order and downstream progression resolve equal-time ties.
                ready.append((start,key[1],-key[2],key))
        if not ready:raise ValueError('cyclic pipeline dependencies')
        start,_,_,key=min(ready)
        duration,resources,deps=tasks.pop(key);end=start+duration
        ends[key]=end
        for r in resources:available[r]=end
        events.append(dict(kind=key[0],microbatch=key[1],pp_rank=key[2],
                           start_us=start,end_us=end,resources=list(resources)))
    total=max(ends.values())
    stage_busy=[sum(e['end_us']-e['start_us'] for e in events if s in e['resources']) for s in range(p)]
    return dict(total_us=total,events=events,
        microbatch_completion_us=[ends[('compute',m,p-1)] for m in range(num_microbatches)],
        stage_busy_us=stage_busy,stage_idle_us=[max(0,total-b) for b in stage_busy],
        sum_stage_service_us=num_microbatches*sum(stage_times_us),
        sum_pp_transfer_us=num_microbatches*sum(transfer_times_us),
        note='service/idle sums are resource statistics, not additive system latency')
