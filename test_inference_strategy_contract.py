import json
import os
from pathlib import Path
import unittest
from inference_strategy_contract import RankWorkload,capacity_ledger,communication_contract,GIB
from strategy_compute_profiles import ComputeProfiles

MODEL=dict(hidden_size=4096,moe_intermediate_size=1536,num_experts=128,num_experts_per_tok=8,
    num_hidden_layers=94,num_attention_heads=64,num_key_value_heads=4,head_dim=128,
    vocab_size=151936,torch_dtype='bfloat16',model_type='qwen3_moe',use_cache=True)

class ContractTests(unittest.TestCase):
    def workload(self,tp=1,ep=4):
        return RankWorkload.build(MODEL,world_size=32,global_batch=32,seq=256,tp=tp,ep=ep)

    def test_global_work_and_assignments_conserved(self):
        for tp in (1,2,4):
            for ep in (1,2,4,8,16,32):
                if 32%(tp*ep):continue
                w=self.workload(tp,ep)
                self.assertEqual(w.dp*w.replica_batch,32)
                self.assertEqual(w.rank_batch*w.ep,w.replica_batch)
                self.assertEqual(w.experts*w.tokens_per_expert,2048*tp)
                self.assertEqual(w.rank_batch,tp)

    def test_invalid_partition_rejected(self):
        for kwargs in (dict(tp=3,ep=4),dict(tp=1,ep=4,pp=3)):
            with self.assertRaises(ValueError):
                RankWorkload.build(MODEL,world_size=32,global_batch=32,seq=256,**kwargs)

    def test_capacity_conservation_and_two_limits(self):
        w=self.workload()
        a=capacity_ledger(MODEL,w,hbf_gib=128)
        for p in a['placements']:
            self.assertEqual(p['hbm_peak_bytes']+p['hbf_resident_bytes'],a['all_hbm_peak_bytes'])
            self.assertEqual(p['feasible'],p['hbm_margin_bytes']>=0 and p['hbf_margin_bytes']>=0)
        self.assertEqual(a['q_min'],27)
        self.assertEqual(capacity_ledger(MODEL,w,hbf_gib=0)['feasible_q'],[])
        self.assertTrue(set(a['feasible_q'])<=set(capacity_ledger(MODEL,w,hbf_gib=256)['feasible_q']))

    def test_reserve_changes_borderline_placement(self):
        w=self.workload(ep=16)
        self.assertEqual(capacity_ledger(MODEL,w,reserve_gib=0)['q_min'],2)
        self.assertEqual(capacity_ledger(MODEL,w,reserve_gib=2)['q_min'],3)

    def test_kv_uses_rank_tokens(self):
        a=capacity_ledger(MODEL,self.workload(ep=4));b=capacity_ledger(MODEL,self.workload(ep=32))
        self.assertEqual(a['kv_bytes'],b['kv_bytes'])
        self.assertEqual(a['kv_bytes'],2*256*512*2*94)
        self.assertEqual(a['parameter_bytes']['experts'],8*b['parameter_bytes']['experts'])

    def test_tp1_payload_not_ep_proportional(self):
        values=[communication_contract(MODEL,self.workload(ep=e))['ep_dispatch_input_payload_bytes'] for e in (4,8,16,32)]
        self.assertEqual(values,[16*1024**2]*4)

    def test_profile_missing_is_not_zero(self):
        path=Path(os.environ.get('HBF_PROFILE_FIXTURE',str(Path(__file__).parent/'test_profiles.csv')))
        if not path.exists():self.skipTest('integration fixture not provided')
        p=ComputeProfiles(path)
        missing=p.query(tp=1,m=128,e=16,latency=500)
        self.assertEqual(missing['status'],'UNSUPPORTED');self.assertIsNone(missing['time_us'])
        self.assertEqual(p.query(tp=1,m=512,e=4,latency=500)['method'],'exact')
        self.assertEqual(p.query(tp=1,m=512,e=4,latency=300)['status'],'UNSUPPORTED')
        self.assertEqual(p.query(tp=1,m=64,e=1,latency=250,kind='bare')['method'],'bare_exact_shape_latency_interpolation')
        self.assertEqual(p.query(tp=1,m=64,e=1,latency=1100,kind='bare')['status'],'UNSUPPORTED')

if __name__=='__main__':unittest.main()
