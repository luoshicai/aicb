import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from test_inference_strategy_contract import MODEL
from inference_strategy_contract import RankWorkload,capacity_ledger,parameter_bytes
from capacity_strategy_search import enumerate_candidates
from build_strategy_body_workload import build,build_pipeline,STAGES
from inference_pipeline import schedule_forward
from evaluate_inference_pipeline import validate_bundle,parse_simai_csv,effective_bandwidth_costs,evaluate
from search_inference_pipeline import search,find_profile


def workload(pp=4,mb=8):
    return RankWorkload.build(MODEL,world_size=8*pp,global_batch=32,seq=256,
                              tp=1,ep=8,pp=pp,microbatch_size=mb)


def profile(w,q=0):
    return dict(status='SUPPORTED',sources=['synthetic unittest, not a performance label'],
        model_sha256=hashlib.sha256(json.dumps(MODEL,sort_keys=True).encode()).hexdigest(),
        shape_placement=dict(TP=w.tp,M=w.tokens_per_expert,E=w.experts,
            rank_batch=w.rank_batch,seq=w.seq,q=q,hbf_latency=100),
        stage_times_us={s:10.0 for s in STAGES})


class PipelineCapacityTests(unittest.TestCase):
    def test_uneven_layers_and_endpoint_parameter_conservation(self):
        w=workload();self.assertEqual(w.stage_layers,(24,24,23,23))
        parts=[parameter_bytes(MODEL,w,s) for s in range(w.pp)]
        whole=parameter_bytes(MODEL,workload(pp=1))
        for key in whole:self.assertEqual(sum(x[key] for x in parts),whole[key])
        self.assertTrue(parts[0]['embedding']>0 and parts[-1]['lm_head']>0)
        self.assertTrue(all(x['embedding']==0 for x in parts[1:]))
        self.assertTrue(all(x['lm_head']==0 for x in parts[:-1]))

    def test_kv_persists_across_microbatches_and_layers(self):
        allbatch=capacity_ledger(MODEL,workload(mb=32))
        split=capacity_ledger(MODEL,workload(mb=8))
        self.assertEqual([s['kv_bytes'] for s in allbatch['stages']],
                         [s['kv_bytes'] for s in split['stages']])
        whole=capacity_ledger(MODEL,workload(pp=1,mb=32))
        self.assertEqual(sum(s['kv_bytes'] for s in split['stages']),whole['kv_bytes'])
        self.assertGreater(allbatch['stages'][0]['workspace_stage_bytes']['attention_eager'],
                           split['stages'][0]['workspace_stage_bytes']['attention_eager'])

    def test_global_assignments_and_unique_rank_mapping(self):
        candidates,_=enumerate_candidates(MODEL,world_size=32,global_batch=32,seq=256,
                                          microbatch_sizes=(8,16,32))
        self.assertEqual(set(c['workload'].pp for c in candidates),{1,2,4})
        for c in candidates:
            w=c['workload']
            self.assertEqual(w.dp*w.microbatch_size*w.num_microbatches,32)
            self.assertEqual(w.dp*w.num_microbatches*w.ep*w.experts*w.tokens_per_expert,32*256*8)
            ids={w.rank_id(d,p,e,t) for d in range(w.dp) for p in range(w.pp)
                 for e in range(w.ep) for t in range(w.tp)}
            self.assertEqual(ids,set(range(32)))

    def test_capacity_every_stage_and_no_hbf_pool_merging(self):
        a=capacity_ledger(MODEL,workload(),hbf_gib=0)
        self.assertTrue(a['feasible']);self.assertEqual(a['q_min_by_stage'],[0]*4)
        b=capacity_ledger(MODEL,workload(pp=1),hbf_gib=0)
        self.assertFalse(b['feasible'])
        c=capacity_ledger(MODEL,workload(pp=1),hbf_gib=2793.968)
        self.assertTrue(c['feasible']);self.assertGreater(c['q_min'],0)
        d=capacity_ledger(MODEL,workload(),hbm_gib=1,hbf_gib=3000)
        self.assertFalse(d['feasible']) # Nonexpert weights/activations still need HBM.
        for stage in c['stages']:
            for placement in stage['placements']:
                self.assertEqual(placement['hbm_peak_bytes']+placement['hbf_resident_bytes'],stage['all_hbm_peak_bytes'])

    def test_invalid_microbatch_nan_and_empty_stages(self):
        for mb in (0,7,64,True):
            with self.assertRaises(ValueError):workload(mb=mb)
        with self.assertRaises(ValueError):capacity_ledger(MODEL,workload(),hbf_gib=float('nan'))
        with self.assertRaises(ValueError):
            RankWorkload.build(dict(MODEL,num_hidden_layers=2),world_size=32,global_batch=32,seq=256,tp=1,ep=8,pp=4)


class ForwardScheduleTests(unittest.TestCase):
    def test_balanced_and_uneven_hand_calculation(self):
        for p in (1,2,4):
            for m in (1,2,8):
                self.assertEqual(schedule_forward([10]*p,m,[0]*(p-1))['total_us'],(m+p-1)*10)
        self.assertEqual(schedule_forward([24,24,23,23],4,[0]*3)['total_us'],166)

    def test_nonzero_links_and_resource_exclusion(self):
        result=schedule_forward([10,10],2,[2])
        self.assertEqual(result['total_us'],34)
        for s in range(2):
            events=sorted([e for e in result['events'] if s in e['resources']],key=lambda e:e['start_us'])
            for a,b in zip(events,events[1:]):self.assertLessEqual(a['end_us'],b['start_us'])
        for m in range(2):
            events=[e for e in result['events'] if e['microbatch']==m]
            self.assertEqual([e['kind'] for e in events],['compute','send','compute'])

    def test_missing_or_invalid_costs_fail(self):
        for costs in ([],[float('inf')],[-1]):
            with self.assertRaises(ValueError):schedule_forward([10,10],1,costs)


class PipelineWorkloadTests(unittest.TestCase):
    def test_stage_coverage_units_and_global_work(self):
        w=workload();bundle=build_pipeline(MODEL,w,[profile(w)]*4,q_by_stage=[0]*4,hbf_latency=100)
        validate_bundle(bundle)
        self.assertEqual(sum(s['manifest']['layers'] for s in bundle['stages']),94)
        self.assertEqual(sum(s['manifest']['compute_sum_us'] for s in bundle['stages']),4700)
        for s in bundle['stages']:
            self.assertIn('pp: 1 all_gpus: 8',s['workload'].splitlines()[0])
            self.assertEqual(s['manifest']['profile']['shape_placement']['M'],128)
        self.assertEqual(bundle['global_tokens'],8192)
        with self.assertRaises(ValueError):build(MODEL,w,profile(w),q=0,hbf_latency=100)

    def test_shape_placement_and_bundle_tampering(self):
        w=workload();p=profile(w);p['shape_placement']['M']=512
        with self.assertRaises(ValueError):build_pipeline(MODEL,w,[p]*4,q_by_stage=[0]*4,hbf_latency=100)
        b=build_pipeline(MODEL,w,[profile(w)]*4,q_by_stage=[0]*4,hbf_latency=100)
        b['stages'][0]['workload']+='bad\n'
        with self.assertRaises(ValueError):validate_bundle(b)

    def test_pp_costs_explicit_units_and_required_before_backend(self):
        w=workload();b=build_pipeline(MODEL,w,[profile(w)]*4,q_by_stage=[0]*4,hbf_latency=100)
        cost=effective_bandwidth_costs(b,100,2)
        expected=2+(8*256*4096*2)/(100*1000)
        self.assertEqual(cost['time_us'],[expected]*3)
        with self.assertRaisesRegex(ValueError,'explicit PP'):
            evaluate(b,simai_root='/nonexistent',output='/nonexistent')

    def test_backend_rejects_training_and_nonfinite_output(self):
        header='File name,Expose DP comm,Expose DP_EP comm,Expose TP comm,Expose_EP_comm,Expose_PP_comm,bubble time,total comp,total exposed comm,Total time\n'
        valid=header+'test,0,0,3,7,0,0,100,10,110\n'
        self.assertEqual(parse_simai_csv(valid)['total_us'],110)
        appended=valid+'layer_name,fwd compute,algbw\nattention,100,-nan\n'
        self.assertEqual(parse_simai_csv(appended)['total_us'],110)
        for row in ('test,0,0,3,7,2,0,100,10,112\n','test,0,0,3,7,0,0,100,10,nan\n'):
            with self.assertRaises(ValueError):parse_simai_csv(header+row)


class FrozenPP1Regression(unittest.TestCase):
    def test_all_fifteen_legacy_capacity_points(self):
        path=os.environ.get('HBF_PP1_CONTRACT')
        if not path:self.skipTest('set HBF_PP1_CONTRACT to a frozen experiment source')
        spec=importlib.util.spec_from_file_location('frozen_pp1',path)
        legacy=importlib.util.module_from_spec(spec);sys.modules[spec.name]=legacy;spec.loader.exec_module(legacy)
        for tp in (1,2,4):
            for ep in (1,2,4,8,16,32):
                if 32%(tp*ep):continue
                args=dict(world_size=32,global_batch=32,seq=256,tp=tp,ep=ep)
                old=legacy.capacity_ledger(MODEL,legacy.RankWorkload.build(MODEL,**args))
                new=capacity_ledger(MODEL,RankWorkload.build(MODEL,**args))
                for key in ('parameter_bytes','kv_bytes','all_hbm_peak_bytes','q_min','placements','workspace_stage_bytes'):
                    self.assertEqual(old[key],new[key],(tp,ep,key))


class SearchTests(unittest.TestCase):
    def test_missing_profiles_are_unknown_not_infeasible_or_ranked(self):
        candidates,_=enumerate_candidates(MODEL,world_size=32,global_batch=32,seq=256,
            tps=(1,),eps=(8,),pps=(4,),microbatch_sizes=(8,),hbf_gib=0)
        with tempfile.TemporaryDirectory() as tmp:
            result=search(MODEL,candidates,[],output=Path(tmp)/'out',simai_root='/nonexistent',dry_run=True)
            self.assertEqual(result['results'][0]['status'],'UNKNOWN')
            self.assertIsNone(result['best_completed'])
            self.assertFalse(result['search_domain_fully_evaluated'])
            self.assertEqual(len(json.loads((Path(tmp)/'out/missing_profiles.json').read_text())),4)

    def test_ready_bundle_reaches_search_from_capacity_api(self):
        candidates,_=enumerate_candidates(MODEL,world_size=32,global_batch=32,seq=256,
            tps=(1,),eps=(8,),pps=(4,),microbatch_sizes=(8,),hbf_gib=0)
        w=candidates[0]['workload']
        with tempfile.TemporaryDirectory() as tmp:
            result=search(MODEL,candidates,[profile(w)],output=Path(tmp)/'out',
                simai_root='/nonexistent',hbf_latency=100,pp_bandwidth=100,pp_latency=2,dry_run=True)
            self.assertEqual(result['results'][0]['status'],'READY')
            self.assertFalse(result['search_domain_fully_evaluated'])

    def test_prediction_never_overrides_reference_and_conflicts_rejected(self):
        w=workload();ref=profile(w);pred=copy.deepcopy(ref)
        pred['mode']='prediction';pred['stage_times_us']['selfAttn']=1000
        found,gap=find_profile(MODEL,w,0,100,[pred,ref])
        self.assertEqual(found['stage_times_us']['selfAttn'],10)
        pred['mode']='reference'
        with self.assertRaisesRegex(ValueError,'conflicting'):
            find_profile(MODEL,w,0,100,[pred,ref])


if __name__=='__main__':unittest.main()
