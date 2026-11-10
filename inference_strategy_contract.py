"""Explicit fixed-global-batch contract for the HBF prefill experiments.

Capacity is a conservative analytical budget, not an allocator measurement.
Legacy micro_batch APIs are deliberately not changed by this module.
"""
from dataclasses import dataclass, asdict
import math
from estimate_prefill_memory import split_layers, qwen3_param_breakdown

GIB = 1024**3

@dataclass(frozen=True)
class RankWorkload:
    world_size: int
    global_batch: int
    seq: int
    tp: int
    ep: int
    dp: int
    replica_batch: int
    rank_batch: int
    replica_tokens: int
    rank_tokens: int
    experts: int
    tokens_per_expert: int
    local_intermediate: int
    pp: int = 1
    microbatch_size: int = 0
    num_microbatches: int = 1
    stage_layers: tuple = ()

    @classmethod
    def build(cls, model, *, world_size, global_batch, seq, tp, ep, pp=1, microbatch_size=None):
        if any(type(x) is not int or x <= 0 for x in (world_size,global_batch,seq,tp,ep,pp)):
            raise ValueError('positive integer sizes required')
        if pp > model['num_hidden_layers']:
            raise ValueError('empty pipeline stages are not supported')
        if world_size % (tp*ep*pp):
            raise ValueError('TP*EP*PP must divide world size')
        dp = world_size//(tp*ep*pp)
        divisibility = ((global_batch,dp), (model['num_experts'],ep),
                         (model['moe_intermediate_size'],tp),
                         (model['num_attention_heads'],tp), (model['num_key_value_heads'],tp))
        if any(a % b for a,b in divisibility):
            raise ValueError('batch/model dimensions cannot be partitioned exactly')
        batch = global_batch//dp
        mb = batch if microbatch_size is None else microbatch_size
        if type(mb) is not int or mb <= 0 or batch % mb:
            raise ValueError('microbatch_size must divide replica batch')
        if mb % ep or mb*seq*model['num_experts_per_tok'] % model['num_experts']:
            raise ValueError('rank batch and balanced expert tokens must be integral')
        return cls(world_size,global_batch,seq,tp,ep,dp,batch,mb//ep,mb*seq,
                   mb//ep*seq,model['num_experts']//ep,
                   mb*seq*model['num_experts_per_tok']//model['num_experts'],
                   model['moe_intermediate_size']//tp,pp,mb,batch//mb,
                   tuple(split_layers(model['num_hidden_layers'],pp)))

    def tag(self):
        base = f'tp{self.tp}_pp{self.pp}_ep{self.ep}_dp{self.dp}'
        return base if self.num_microbatches == 1 else f'{base}_mb{self.microbatch_size}'

    def stage_range(self, pp_rank):
        if type(pp_rank) is not int or not 0 <= pp_rank < self.pp:
            raise ValueError('invalid pipeline rank')
        return sum(self.stage_layers[:pp_rank]), sum(self.stage_layers[:pp_rank+1])

    def rank_id(self, dp_rank, pp_rank, ep_rank, tp_rank):
        for value, size in zip((dp_rank,pp_rank,ep_rank,tp_rank),(self.dp,self.pp,self.ep,self.tp)):
            if type(value) is not int or not 0 <= value < size:
                raise ValueError('invalid rank coordinate')
        return ((dp_rank*self.pp+pp_rank)*self.ep+ep_rank)*self.tp+tp_rank


def parameter_bytes(model, w, pp_rank=0):
    """Reuse AICB's original per-stage layer and endpoint allocation."""
    w.stage_range(pp_rank)
    config=dict(model,tensor_model_parallel_size=w.tp,
                expert_model_parallel_size=w.ep,pipeline_model_parallel=w.pp)
    parts=qwen3_param_breakdown(config,w.stage_layers[pp_rank],pp_rank,2)
    if model.get('tie_word_embeddings') and w.pp>1 and pp_rank==w.pp-1:
        parts['lm_head']=math.ceil(model['vocab_size']/w.tp)*model['hidden_size']*2
    return {name:parts[key] for name,key in (
        ('embedding','embedding'),('lm_head','lm_head'),('final_norm','final_norm'),
        ('layer_norms','transformer_norm'),('attention','transformer_attention'),
        ('router','transformer_router'),('experts','transformer_experts'))}


def _stage_capacity(model, w, pp_rank, *, hbm_gib=40, hbf_gib=128, utilization=.9, reserve_gib=2):
    if any(not math.isfinite(x) for x in (hbm_gib,hbf_gib,utilization,reserve_gib)) or hbm_gib <= 0 or hbf_gib < 0 or reserve_gib < 0 or not 0 < utilization <= 1:
        raise ValueError('invalid capacity budget')
    if model.get('torch_dtype') != 'bfloat16' or model.get('model_type') != 'qwen3_moe':
        raise ValueError('this budget supports BF16 Qwen3 MoE only')
    h=model['hidden_size']; rt=w.rank_tokens; n=w.replica_tokens; m=w.tokens_per_expert
    q=model['num_attention_heads']*model['head_dim']//w.tp
    kv=model['num_key_value_heads']*model['head_dim']//w.tp
    heads=model['num_attention_heads']//w.tp; k=model['num_experts_per_tok']
    params=parameter_bytes(model,w,pp_rank)
    layers=w.stage_layers[pp_rank]
    # All-layer KV is resident; current-layer K/V temporaries are also budgeted
    # below. This deliberately overcounts an alias, rather than hiding it.
    kv_bytes=2*rt*w.num_microbatches*kv*2*layers if model.get('use_cache',True) else 0
    components={
        'attention_eager': dict(hidden_and_residual=3*rt*h*2,
            qkv_output=rt*(2*q+2*kv)*2,
            scores_probabilities_cast_workspace=w.rank_batch*heads*w.seq*w.seq*12,
            mask=w.seq*w.seq*2),
        'router': dict(hidden_and_residual=3*rt*h*2,
            logits=rt*model['num_experts']*4, topk=rt*k*(4+8)),
        'experts_receive_side': dict(replica_input_output=2*n*h*2,
            gather_and_weighted_outputs=3*m*h*2,
            gate_up_activation=4*m*w.local_intermediate*2,
            indices=w.experts*m*8),
        'communication': dict(dispatch_send_receive_and_combine=4*rt*k*h*2),
        'lm_head': dict(hidden=rt*h*2,logits=rt*math.ceil(model['vocab_size']/w.tp)*4),
    }
    if w.tp > 1:
        components['expert_tp_reduce'] = dict(partial_output=n*h*2, collective_scratch=n*h*2)
        components['attention_tp_reduce'] = dict(hidden_and_residual=2*rt*h*2, collective_scratch=rt*h*2)
    if pp_rank != w.pp-1:
        components.pop('lm_head')
    # Explicit no-overlap schedule: only one stage workspace is live, KV and
    # weights remain resident. Reserve covers framework/library allocations.
    stages={name:sum(parts.values()) for name,parts in components.items()}
    peak_stage=max(stages,key=stages.get)
    # Store-and-forward: reserve all microbatch input/output boundary buffers.
    pipeline_buffers=2*rt*h*2*w.num_microbatches if w.pp>1 else 0
    base=sum(params.values())+kv_bytes+stages[peak_stage]+pipeline_buffers+int(reserve_gib*GIB)
    per_expert=params['experts']//w.experts
    hbm_limit=int(hbm_gib*GIB*utilization); hbf_limit=int(hbf_gib*GIB*utilization)
    placements=[]
    for count in range(w.experts+1):
        moved=count*per_expert
        placements.append(dict(q=count,expert_ids=list(range(count)),
            hbm_peak_bytes=base-moved,hbf_resident_bytes=moved,
            hbm_margin_bytes=hbm_limit-base+moved,hbf_margin_bytes=hbf_limit-moved,
            feasible=base-moved<=hbm_limit and moved<=hbf_limit))
    feasible=[p['q'] for p in placements if p['feasible']]
    return dict(strategy=w.tag(),workload=asdict(w),pp_rank=pp_rank,local_layers=layers,
        layer_range=list(w.stage_range(pp_rank)),parameter_bytes=params,kv_bytes=kv_bytes,
        pipeline_buffer_bytes=pipeline_buffers,
        workspace_components=components,workspace_stage_bytes=stages,peak_workspace_stage=peak_stage,
        reserve_bytes=int(reserve_gib*GIB),all_hbm_peak_bytes=base,
        per_expert_all_layers_bytes=per_expert,hbm_limit_bytes=hbm_limit,hbf_limit_bytes=hbf_limit,
        q_min=min(feasible) if feasible else None,feasible_q=feasible,placements=placements,
        budget_kind='conservative_analytical_not_allocator_verified',
        assumptions=['BF16 balanced routing; fixed direct weight placement',
        'sequential eager expert loop; full-replica receive-side input/output',
        'stages do not overlap; expert tensors and communication workspace not concurrent',
        'eager attention score/probability casts budgeted at 12 bytes per element',
        'KV plus current K/V temporaries conservatively counted; library reserve explicit',
        'no HBF staging copies, dynamic migration, bandwidth or paging model'])


def capacity_ledger(model,w,**kwargs):
    """Scalar legacy fields describe the peak all-HBM stage; stages are authoritative.

    q_min/feasible_q constrain a common q on every stage. Independent stage
    placements use q_min_by_stage/feasible, without requiring one common q.
    """
    stages=[_stage_capacity(model,w,s,**kwargs) for s in range(w.pp)]
    peak=max(stages,key=lambda s:s['all_hbm_peak_bytes'])
    result=dict(peak)
    common=sorted(set.intersection(*(set(s['feasible_q']) for s in stages)))
    result.update(stages=stages,peak_stage_pp_rank=peak['pp_rank'],
        q_min=min(common) if common else None,feasible_q=common,
        q_min_by_stage=[s['q_min'] for s in stages],
        feasible=all(s['q_min'] is not None for s in stages),schema='hbf_stage_capacity_v2',
        scope='balanced_prefill_forward_store_and_forward')
    return result


def communication_contract(model,w):
    payload=w.rank_tokens*model['num_experts_per_tok']*model['hidden_size']*2
    return dict(strategy=w.tag(),ep_dispatch_input_payload_bytes=payload,
        ep_combine_input_payload_bytes=payload,
        attention_tp_output_bytes=w.rank_tokens*model['hidden_size']*2 if w.tp>1 else 0,
        expert_tp_partial_assignments=w.experts*w.tokens_per_expert,
        expert_tp_reduction_required=w.tp>1,
        ep_local_fraction=1/w.ep,
        expected_peer_chunk_bytes=payload//w.ep,
        expected_remote_peer_send_bytes=(payload//w.ep)*(w.ep-1),
        pp=w.pp,pp_boundaries=w.pp-1,num_microbatches=w.num_microbatches,
        pp_payload_bytes_per_rank_per_microbatch=w.rank_tokens*model['hidden_size']*2 if w.pp>1 else 0,
        pp_payload_bytes_per_replica_per_boundary=w.rank_tokens*model['hidden_size']*2*w.tp*w.ep if w.pp>1 else 0,
        rank_order='dp,pp,ep,tp (tp contiguous)',
        status='TP1_EP_BYTE_SEMANTICS_REQUIRES_SIMAI_CHECK' if w.tp==1 else 'BLOCKED_EXPERT_TP_REDUCTION_AUDIT',
        note='Payload before peer/local traffic filtering; not network wire bytes. DP has no gradient communication.')
