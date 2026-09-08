import json, os, time, uuid, urllib.request
BASE = os.environ.get("SGLANG_BASE_URL", "http://127.0.0.1:30000")
def post(path, body=None):
    req=urllib.request.Request(BASE+path, data=(json.dumps(body).encode() if body is not None else b""), headers={"Content-Type":"application/json"}, method="POST")
    return json.load(urllib.request.urlopen(req, timeout=300))
def flush():
    for _ in range(30):
        try: post("/flush_cache"); return
        except Exception: time.sleep(0.5)
def gen(text, n):
    r=post("/generate", {"text":text,"sampling_params":{"max_new_tokens":n,"temperature":0}})
    m=r["meta_info"]; d=m.get("cached_tokens_details") or {}
    return r["text"], m["prompt_tokens"], d.get("host",0), d.get("device",0)

story=("Dr. Elena Marsh had spent eleven years studying the tidal caves of the northern coast. Every spring, when the storms subsided, she and her two assistants, Tomas and Priya, would row out to the limestone cliffs and map the passages that the winter tides had carved. In the spring of the eleventh year they found something new: a chamber, forty meters in, whose walls were covered in a thin blue mineral crust that glowed faintly when their lamps were switched off. Tomas wanted to scrape a sample immediately. Priya argued they should photograph everything first. Elena, remembering how a careless sample had ruined a site early in her career, sided with Priya. They spent three hours photographing before taking a single fingernail-sized flake, which Elena sealed in a glass vial and labeled with the date. Back at the field station, the flake continued to glow for exactly nineteen minutes after the lights went out, then faded. ")
prompts = [
 ("阅读理解", story*4 + "\n\nQuestion: Why did Elena side with Priya rather than Tomas, and how long did the sample glow at the field station?\nAnswer:", 80),
 ("代码", "# Python utilities for a small key-value store with LRU eviction.\n" + "\n".join(f"# Note {i}: the store must remain thread-safe and every public method must document its complexity." for i in range(60)) + "\n\nfrom collections import OrderedDict\nimport threading\n\nclass LRUStore:\n    \"\"\"Thread-safe LRU key-value store with a fixed capacity.\"\"\"\n\n    def __init__(self, capacity: int):\n", 120),
 ("数学", "Below are worked examples.\n" + "".join(f"Q: A tank holds {100+7*i} liters and drains {3+i%4} liters per minute. How many minutes to empty?\nA: {100+7*i} / {3+i%4} = {(100+7*i)/(3+i%4):.2f} minutes.\n\n" for i in range(40)) + "Q: A tank holds 512 liters and drains 8 liters per minute, but a pump adds 2 liters per minute. How many minutes to empty?\nA:", 60),
]
for name, text, n in prompts:
    text = f"[session {uuid.uuid4().hex[:8]}] " + text
    flush(); time.sleep(0.5)
    cold = gen(text, n); time.sleep(1.5); flush(); time.sleep(0.5)
    host = gen(text, n)
    dev  = gen(text, n)
    print(f"\n===== {name}  prompt_tokens={cold[1]} =====")
    for tag, r in (("cold ", cold), ("host ", host), ("dev  ", dev)):
        print(f"[{tag} host={r[2]:>5} dev={r[3]:>5}] {r[0]!r}")
    print("host==cold:", host[0]==cold[0], " host==dev:", host[0]==dev[0])
