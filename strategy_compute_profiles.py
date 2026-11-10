"""Conservative reader for audited V2 compute labels. No cross-M fallback."""
import csv
import json

class ComputeProfiles:
    def __init__(self,path):
        with open(path) as f:
            self.rows=[r for r in csv.DictReader(f) if r['status'] in ('COMPLETE','REUSED_AUDITED')]
        versions={r['binary_sha256'] for r in self.rows}
        if len(versions)!=1:
            raise ValueError('mixed or missing simulator versions')
        self.binary_sha256=versions.pop()
        if not self.binary_sha256:
            raise ValueError('missing simulator hash')

    def query(self, *, tp,m,e,latency,kind='group',placement='full',dram_latency=100):
        if latency<100 or latency>1000 or dram_latency!=100:
            return self.unsupported('latency/hardware outside audited domain')
        rows=[r for r in self.rows if r['profile_id']==f'{kind}_tp{tp}_m{m}'+(f'_e{e}' if kind=='group' else '')
              and int(r['E_local'])==e and r['placement']==placement and int(r['dram_latency'])==dram_latency]
        exact=[r for r in rows if int(r['dram_hbf_latency'])==latency]
        if exact:
            return self.result(float(exact[0]['time_us']),'exact',exact)
        # Only bare exact-shape interpolation has independent latency validation.
        if kind=='bare' and placement=='full':
            anchors=[r for r in rows if int(r['dram_hbf_latency']) in (100,200,400,1000)]
            below=[r for r in anchors if int(r['dram_hbf_latency'])<latency]
            above=[r for r in anchors if int(r['dram_hbf_latency'])>latency]
            if below and above:
                a=max(below,key=lambda r:int(r['dram_hbf_latency']))
                b=min(above,key=lambda r:int(r['dram_hbf_latency']))
                frac=(latency-int(a['dram_hbf_latency']))/(int(b['dram_hbf_latency'])-int(a['dram_hbf_latency']))
                return self.result(float(a['time_us'])+frac*(float(b['time_us'])-float(a['time_us'])),
                                   'bare_exact_shape_latency_interpolation',[a,b])
        # The independent E validation covers precisely E16 at these combinations.
        if kind=='group' and placement=='full' and tp in (1,4) and m in (64,512) and e==16 and latency in (100,500,1000):
            fits=[]
            for fit_e in (1,4):
                match=[r for r in self.rows if r['profile_id']==f'group_tp{tp}_m{m}_e{fit_e}'
                       and r['placement']=='full' and int(r['dram_hbf_latency'])==latency and int(r['dram_latency'])==100]
                if len(match)!=1:
                    return self.unsupported('missing E calibration anchors')
                fits.extend(match)
            a,b=[float(r['time_us']) for r in fits]
            return self.result(a+(e-1)*(b-a)/3,'validated_E1_E4_affine',fits)
        return self.unsupported('missing exact group/placement or unvalidated composition; no cross-M fallback')

    def result(self,value,method,rows):
        return dict(status='SUPPORTED',time_us=value,method=method,binary_sha256=self.binary_sha256,
                    source_cases=[r['case_id'] for r in rows],
                    source_contexts=sorted({r['context'] for r in rows}),
                    source_hashes=sorted({r['source_sha256'] for r in rows}),
                    note='Microbenchmark context only; complete-layer bridge still required')

    @staticmethod
    def unsupported(reason):
        return dict(status='UNSUPPORTED',time_us=None,reason=reason)
