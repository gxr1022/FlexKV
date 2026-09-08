"""Teacher-forced scoring: fixed continuation, logprob_start_len=len(prompt) so the prompt
prefix is served from cache (FlexKV host after flush / SGLang device otherwise). batch=1."""
import os, sys, time, numpy as np
from sglang.test.kl_test_utils import get_input_ids, _generate, _flush_cache
BASE = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000")   # the sglang server under test
MODEL = os.environ.get("MODEL") or sys.exit("set MODEL=<path used as --model-path> (tokenizer for the prompts)")
NEW = 128
# Optional DP-rank pinning for dp>1 servers: generation (the store) on one rank, all three
# scorings on another, so the host restore is a cross-rank restore. Unset = server routing.
_r = lambda k: (int(os.environ[k]) if os.environ.get(k) not in (None, "") else None)
GEN_RANK, SCORE_RANK = _r("SWA_VERIFY_GEN_DP_RANK"), _r("SWA_VERIFY_SCORE_DP_RANK")
def flush():
    for _ in range(30):
        try: _flush_cache(BASE); return
        except Exception: time.sleep(1)
def det(r): d=r["meta_info"].get("cached_tokens_details") or {}; return d.get("host",0), d.get("device",0)
def kl(a,b): a=np.array(a);b=np.array(b);l=a-b;return float(np.mean((np.exp(l)-1)-l))
def score(seq, start):
    r=_generate(BASE,[seq],0,return_logprob=True,logprob_start_len=start,routed_dp_rank=SCORE_RANK)[0]
    return det(r), [x[0] for x in r["meta_info"]["input_token_logprobs"][-NEW:] if x[0] is not None]
ids=[p[:len(p)//256*256] for p in get_input_ids(MODEL, max_prompt_tokens=3000, num_samples=24, trust_remote_code=True)]  # page-aligned: restore boundary == scoring start
rows=[]; t0=time.time()
for i,p in enumerate(ids):
    cont=_generate(BASE,[p],NEW,routed_dp_rank=GEN_RANK)[0]["output_ids"][:NEW]; seq=p+cont; L=len(p)
    flush(); time.sleep(0.3)
    ch,lh=score(seq,L)          # host restore of prompt pages
    cd,ld=score(seq,L)          # device hit
    cd2,ld2=score(seq,L)        # device hit again
    rows.append((i,L,ch,cd,kl(lh,ld),float(np.max(np.abs(np.array(lh)-np.array(ld)))),kl(ld,ld2),float(np.max(np.abs(np.array(ld)-np.array(ld2))))))
print(f"{time.time()-t0:.0f}s total")
print(" i  plen   host(h,d)    dev(h,d)   KL(h,d)   max|dlp|(h,d)   KL(d,d2)  max|dlp|(d,d2)")
for r in rows: print(f"{r[0]:2d} {r[1]:5d} {str(r[2]):>12} {str(r[3]):>11}  {r[4]:.6f}  {r[5]:.4f}          {r[6]:.6f}  {r[7]:.4f}")
H=[r[4] for r in rows]; D=[r[6] for r in rows]
print(f"\nhost-restore vs device : KL avg={np.mean(H):.6f} max={np.max(H):.6f}  max|dlogprob| avg={np.mean([r[5] for r in rows]):.4f}")
print(f"device vs device (noise): KL avg={np.mean(D):.6f} max={np.max(D):.6f}  max|dlogprob| avg={np.mean([r[7] for r in rows]):.4f}")
print(f"bitwise identical logprobs: host-vs-device {sum(1 for r in rows if r[5]==0)}/24, device-vs-device {sum(1 for r in rows if r[7]==0)}/24")
print(f"all host runs hit host: {all(r[2][0]>0 for r in rows)}; all device runs hit device: {all(r[3][1]>0 for r in rows)}")
