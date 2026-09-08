"""Byte-exact check of FlexKV SWA restore on a live DSv4 server.

Server must run with SGLANG_DEBUG_SWA_DUMP_DIR=<dir> (dump hook) and ideally
SGLANG_DEBUG_ZERO_SWA_ON_FLUSH=1 (so restored bytes can only come from FlexKV).
Per prompt: gen (stores) -> device-hit score (dump BEFORE) -> flush -> host-restore
score -> device-hit score (dump AFTER). Compare BEFORE vs AFTER per rank/layer."""
import glob, os, sys, time, torch, numpy as np
from sglang.test.kl_test_utils import get_input_ids, _generate, _flush_cache
BASE = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000")   # the sglang server under test
MODEL = os.environ.get("MODEL") or sys.exit("set MODEL=<path used as --model-path> (tokenizer for the prompts)")
NEW = 128
# must be the same directory the server was started with (SGLANG_DEBUG_SWA_DUMP_DIR=<dir>)
DUMP = os.environ.get("SGLANG_DEBUG_SWA_DUMP_DIR") or sys.exit("set SGLANG_DEBUG_SWA_DUMP_DIR to the server's dump dir")
N=int(sys.argv[1]) if len(sys.argv)>1 else 12
def flush():
    for _ in range(30):
        try: _flush_cache(BASE); return
        except Exception: time.sleep(1)
def det(r): d=r["meta_info"].get("cached_tokens_details") or {}; return d.get("host",0), d.get("device",0)
def score(seq, start):
    r=_generate(BASE,[seq],0,return_logprob=True,logprob_start_len=start)[0]; return det(r)
def dumps_since(t0):
    fs=[f for f in glob.glob(f"{DUMP}/*.pt") if os.path.getmtime(f)>=t0]
    time.sleep(1.0)  # let all ranks finish writing
    return sorted(f for f in glob.glob(f"{DUMP}/*.pt") if os.path.getmtime(f)>=t0)
import random
FRESH=os.environ.get("FRESH","1")=="1"
ids=[p[:len(p)//256*256] for p in get_input_ids(MODEL, max_prompt_tokens=3000, num_samples=N, trust_remote_code=True)][:N]
if FRESH:  # prepend 256 random tokens (drawn from the prompt itself) so the path is new to the FlexKV host pool
    rng=random.Random(time.time_ns())
    ids=[[rng.choice(p) for _ in range(256)]+p for p in ids]
rows=[]; t_all=time.time()
for i,p in enumerate(ids):
    cont=_generate(BASE,[p],NEW)[0]["output_ids"][:NEW]; seq=p+cont; L=len(p)
    t0=time.time()-0.01; cd0=score(seq,L); before=dumps_since(t0)          # device hit -> dump BEFORE
    flush(); time.sleep(0.5)
    ch=score(seq,L)                                                        # host restore (no dump)
    t1=time.time()-0.01; cd1=score(seq,L); after=dumps_since(t1)           # device hit -> dump AFTER
    B={torch.load(f)["rank"]:torch.load(f) for f in before}; A={torch.load(f)["rank"]:torch.load(f) for f in after}
    ranks=sorted(set(B)&set(A)); eq_layers=0; tot=0; keys_match=all(B[r]["key"]==A[r]["key"] for r in ranks)
    per_rank=[]; nr=len(ranks)
    # FlexKV D2H in kv_shared_across_ranks_mode=sharded splits every page into nr byte-quarters, rank i writes
    # quarter i. Ranks may hold slightly different (rounding-level) KV, so the exact predicate is:
    #   AFTER[any rank][layer][quarter i] == BEFORE[rank i][layer][quarter i]
    mosaic_ok=0
    inter_rank_equal=sum(all(torch.equal(B[ranks[0]]["layers"][l],B[r]["layers"][l]) for r in ranks) for l in range(B[ranks[0]]["layers"].shape[0]))
    for r in ranks:
        lb,la=B[r]["layers"],A[r]["layers"]; same=[bool(torch.equal(lb[l],la[l])) for l in range(lb.shape[0])]
        eq_layers+=sum(same); tot+=len(same); per_rank.append(f"r{r}:{sum(same)}/{len(same)}")
        n=la.shape[-1]; q=n//nr
        for l in range(la.shape[0]):
            a=la[l].flatten()
            mosaic_ok+=all(torch.equal(a[i*q:(i+1)*q], B[ranks[i]]["layers"][l].flatten()[i*q:(i+1)*q]) for i in range(nr))
        if not all(same):
            fd=[]
            for l in range(lb.shape[0]):
                d=(lb[l]!=la[l]).flatten().nonzero()
                fd.append(-1 if d.numel()==0 else int(d[0])//448)   # byte offset -> token index in nope block (448 B/token)
            print(f"    r{r} first-diff token per layer: {fd}", flush=True)
    rows.append((i,L,ch,cd0,cd1,len(before),len(after),keys_match,eq_layers,tot,mosaic_ok))
    print(f"{i:2d} L={L:5d} host={ch} devB={cd0} devA={cd1} dumps={len(before)}/{len(after)} key_match={keys_match} "
          f"per-rank-equal={eq_layers}/{tot} mosaic-equal={mosaic_ok}/{tot} ranks-identical-layers={inter_rank_equal}/{tot//nr} [{' '.join(per_rank)}]", flush=True)
ok=sum(1 for r in rows if r[7] and r[8]==r[9] and r[9]>0); mos=sum(1 for r in rows if r[7] and r[10]==r[9] and r[9]>0)
print(f"\n{time.time()-t_all:.0f}s  prompts per-rank byte-identical: {ok}/{len(rows)};  mosaic-exact (sharded D2H predicate): {mos}/{len(rows)}")
print(f"all host scores hit host: {all(r[2][0]>0 for r in rows)}; all dumps present: {all(r[5]>0 and r[6]>0 for r in rows)}")
