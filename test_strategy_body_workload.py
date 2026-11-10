import hashlib,json,unittest
from build_strategy_body_workload import build,STAGES
from inference_strategy_contract import RankWorkload
from test_inference_strategy_contract import MODEL
class BodyTests(unittest.TestCase):
 def setUp(self):
  self.w=RankWorkload.build(MODEL,world_size=32,global_batch=32,seq=256,tp=1,ep=32)
  self.p=dict(status='SUPPORTED',sources=['test'],model_sha256=hashlib.sha256(json.dumps(MODEL,sort_keys=True).encode()).hexdigest(),shape_placement=dict(TP=1,M=512,E=4,rank_batch=1,seq=256,q=0,hbf_latency=100),stage_times_us={s:1.25 for s in STAGES})
 def test_units_communication_and_no_training_work(self):
  text,m=build(MODEL,self.w,self.p,q=0,hbf_latency=100)
  lines=text.splitlines();self.assertEqual(int(lines[1]),470);self.assertEqual(m['global_tokens'],8192)
  rows=[line.split('\t') for line in lines[2:]]
  self.assertEqual(sum(int(r[2]) for r in rows)/1000,94*5*1.25)
  self.assertEqual(sum(r[3]=='ALLTOALL_EP' for r in rows),188)
  self.assertTrue(all(int(r[4])==16777216 for r in rows if r[3]=='ALLTOALL_EP'))
  self.assertTrue(all(r[6]=='NONE' and r[9]=='NONE' for r in rows))
 def test_missing_or_wrong_shape_rejected(self):
  self.p['stage_times_us'].pop('selfAttn')
  with self.assertRaises(ValueError):build(MODEL,self.w,self.p,q=0,hbf_latency=100)
  self.setUp();self.p['shape_placement']['M']=256
  with self.assertRaises(ValueError):build(MODEL,self.w,self.p,q=0,hbf_latency=100)
 def test_infeasible_capacity_rejected(self):
  with self.assertRaises(ValueError):build(MODEL,self.w,self.p,q=0,hbf_latency=100,hbm_gib=1)
if __name__=='__main__':unittest.main()
