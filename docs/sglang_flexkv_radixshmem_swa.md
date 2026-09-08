# SGLang + FlexKV + radixshmem（单机 + SWA）配置与启动

本文讲的是**单机**上把 SGLang 的 host-tier KV cache 接到 FlexKV 的
**radixshmem 后端**，并让 **SWA（滑窗）** 模型（DeepSeek-V4）的滑窗 KV 也走
FlexKV offload/reuse。从环境到启动到验证的一整套都在这里。

> 和 [`radixshmem_cross_node.md`](radixshmem_cross_node.md) 的区别：那篇是
> **跨节点 + vLLM**，要 etcd / RDMA / mooncake / Redis，并且明确写着 radixshmem
> **不支持 SWA**。本文是**单机 + SGLang + SWA**：world_size=1，不需要
> etcd/RDMA/mooncake/Redis 里的任何一个，而 radixshmem 后端在**本机范围内支持
> SWA**（滑窗 KV 存到 CPU SWA pool，SWA-aware GET 从中恢复）。

---

## 0. 名词速查

| 名词 | 一句话解释 |
|---|---|
| radixshmem | FlexKV 把 radix tree + slot mempool 放进 POSIX 共享内存（`/dev/shm`），同机所有 DP 进程直接读同一棵树，无需互相发消息，也无需 KVServer 进程。 |
| SWA component | radixshmem region 里除 Full-KV 外的第二个 slot pool，专存滑窗 KV。DSv4 的 128-token 窗口正好落在一个 256-token page 里，所以**一个窗口 = 一个 slot**（W=1）。 |
| SWA-aware GET | restore 时除 Full KV 外，把末尾窗口的 KV 也一起搬回 GPU 的 GET；由 DSv4 的 hybrid radix cache 自动发起。 |
| DSv4 | DeepSeek-V4 架构。SGLang 检测到它就自动进入 hybrid-SWA 路径，FlexKV 自动建 SWA pool。 |

---

## 1. 它是怎么工作的

单机 radixshmem 只有**本地**一棵树，没有跨节点那一层。SWA 在此基础上多一条并行通路：

```
SGLang（DSv4）
  └─ FlexKVHybridRadixCache  ← 因为 DSv4 是 hybrid-swa 模型，自动选这个而非普通版
       └─ FlexKV KVManager（radixshmem 后端，server_client_mode=False）
            ├─ Full KV  →  radixshmem FULL component  →  CPU / SSD pool
            └─ SWA KV   →  radixshmem SWA  component  →  CPU SWA pool（本机 only）
```

- **PUT**：Full KV 落 FULL 组件；滑窗 KV 落 SWA 组件（`take(k=1, component=SWA)`，
  all-or-none）。SWA 发布在 Full path 发布之后。
- **GET（SWA-aware）**：一次 local-only 的 `FULL|SWA` joint 查询拿到共同命中，Full
  H2D + 一个 k-slot 的 is_swa H2D 进同一 transfer graph，窗口正好贴在恢复前缀的尾部。
- 没有 etcd / RDMA / mooncake / Redis：`world_size=1` 时 region 名不带节点身份，
  跨节点的整条 bootstrap 分支被完全跳过；SWA 匹配强制 local-only（跨节点 SWA 尚未支持）。

---

## 2. 环境

本文对应的已验证环境：

| 组件 | 版本 |
|---|---|
| Python / torch | Python 3.12，torch 2.13.0+cu130（sglang 硬钉 torch 2.13，见 2.1） |
| sglang | 分支 `flexkv-dsv4-main`（PR #31781，`a2d01bcdb5`） |
| flexkv | 本仓库，分支 `radixshmem_rebase_v2`（radixshmem 后端 + SWA component，基于 main c69d7ce） |
| shmradix | 带 component API 的版本（`insert(..., *, component=)`，`background_evict_ratio`） |
| GPU | H20，4 卡 tp=4 |

下文命令用以下变量指代路径，按自己的机器设置一次即可：

```bash
export WORK=/path/to/workspace            # 三个源码仓库并排放在这里
export SGLANG_ROOT=$WORK/sglang            # sglang 源码
export FLEXKV_ROOT=$WORK/FlexKV            # 本仓库
export SHMRADIX_ROOT=$WORK/radixshmem      # shmradix 源码
export VENV=$SGLANG_ROOT/.venv-flexkv      # 本文使用的 venv
export MODEL=/path/to/DeepSeek-V4-Flash-FP8
export FLEXKV_CONFIG=$WORK/flexkv_dsv4.yaml
```

### 2.1 建 venv + 装 sglang

sglang 硬钉 `torch==2.13.0`（cu130 栈），不能复用 torch 2.9/cu128 的环境。

```bash
uv venv $VENV --python 3.12

# 先单独装 torch（sglang 的 setup 构建期需要它在场）
VIRTUAL_ENV=$VENV uv pip install torch==2.13.0

# 装 sglang（editable）。跳过 Rust 扩展（无 cargo 时）；用标准隔离让 uv 自解构建依赖
cd $SGLANG_ROOT
SGLANG_BUILD_RUST_EXTS=none VIRTUAL_ENV=$VENV uv pip install -e python/
```

> 坑：**不要**加 `--no-build-isolation`——那会要求手动补齐一长串未声明的构建依赖
> （cuda-tile→wheel_stub→kernels_data→…）。标准隔离 + `SGLANG_BUILD_RUST_EXTS=none`
> 即可。

### 2.2 flexkv c_ext 对 cu130 重编

c_ext 若是 torch 2.9 编的，在 torch 2.13 下会 `undefined symbol`（c10 CUDA 符号变了），
**必须对 CUDA 13.0 重编**：

```bash
cd $FLEXKV_ROOT
git submodule update --init third_party/spdlog          # c_ext 需要
FLEXKV_DEBUG=1 CUDA_HOME=/usr/local/cuda-13.0 MAX_JOBS=64 \
    $VENV/bin/python setup.py build_ext --inplace
```

> ⚠️ `flexkv/c_ext.so` 是**全局产物**，这一步会覆盖掉 torch 2.9/cu128 版——
> 其他 torch 版本的环境之后要用就得各自重编，或先备份两份 `.so`。

### 2.3 装 shmradix + flexkv 运行时依赖进 venv

```bash
# shmradix：源目录的 .so 与 torch 无关，但构建要 pybind11/cmake
VIRTUAL_ENV=$VENV uv pip install pybind11 cmake
VIRTUAL_ENV=$VENV uv pip install -e $SHMRADIX_ROOT/python --no-build-isolation --no-deps

# flexkv editable（若还没）+ 它的运行时依赖（venv 里默认缺）
VIRTUAL_ENV=$VENV uv pip install -e $FLEXKV_ROOT --no-deps
VIRTUAL_ENV=$VENV uv pip install expiring_dict nvtx pyzmq redis msgspec cloudpickle
```

### 2.4 冒烟检查 import 链

```bash
CUDA_MPS_PIPE_DIRECTORY=/nonexistent/mps $VENV/bin/python - <<'PY'
import sglang;            print("sglang", sglang.__version__)
import flexkv.c_ext;      print("flexkv.c_ext OK")
import shmradix;          print("shmradix", shmradix.ComponentType.FULL, shmradix.ComponentType.SWA)
from flexkv.integration.sglang.connector import FlexKVConnector; print("connector OK")
from flexkv.cache.radix_shmem_engine import COMPONENT_MASK_SWA;  print("radix SWA OK")
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

> **MPS 坑**：如果机器的系统默认管道 `/tmp/nvidia-mps` 后面挂着一个不属于你的 MPS
> 守护进程（共享机器上常见），任何 CUDA 程序看到它就会去连，结果 `Error 805` / `cuda.is_available()`
> 为 False。任何用到 CUDA 的命令前加 `CUDA_MPS_PIPE_DIRECTORY=/nonexistent/mps`
> 让 runtime 跳过 MPS。**目录必须真的不存在**，并且启动 FlexKV 时必须同时设
> `FLEXKV_ENABLE_MPS=0`（见 §3.1），否则 FlexKV 会在这个目录下真的起一个 MPS
> 守护进程，之后所有带这个变量的进程都变成它的客户端，它一死全部崩。

---

## 3. 配置与启动

### 3.1 环境变量

单机 radixshmem **只需要一个** env 打开后端，其余全用默认：

```bash
export FLEXKV_RADIX_SHMEM=1        # 唯一必须项。打开 radixshmem 后端
export FLEXKV_ENABLE_MPS=0         # 有系统级 MPS 守护进程的机器上必须。默认 1 会执行 nvidia-cuda-mps-control -d，
                                   # 与 CUDA_MPS_PIPE_DIRECTORY 绕行叠加就会造出游离的 MPS 守护进程
```

MPS 只是让 FlexKV 的 transfer worker 进程和推理进程共享 GPU context、拷贝与计算重叠的
性能优化，不是功能依赖；单机验证关掉即可。

`FLEXKV_RADIX_WORLD_SIZE` 默认 1（单机），因此**跨节点那套变量全都不用设**：
`FLEXKV_RADIX_REGISTRY`（etcd）、`FLEXKV_RADIX_RPC_ADDRESS`/`_INTERFACE`、
`FLEXKV_RADIX_RDMA_DEV`、`SHMRADIX_RHT_SLOTS`、`SHMRADIX_CLUSTER_ID` 等在
world_size=1 时根本不会被读取。

SWA 相关（可选）：

```bash
export FLEXKV_ENABLE_SWA_TRANSFER=1   # 默认就是 1（DSv4 上 SWA 数据面默认开）。
                                      # 设 0 = 关掉 SWA 字节搬运，干净退化成 full-KV-only（A/B 用）
```

单机跑**多个** FlexKV 实例时（一般用不到）才需要给每个实例区分名字/端口：

```bash
export FLEXKV_SHM_RADIX_ID=inst2                       # shm region 名 + TE channel 名，默认 flexkv
export FLEXKV_SERVER_RECV_PORT=ipc:///tmp/flexkv_inst2 # 派生的 gpu_register_port IPC 路径不撞
```

### 3.2 FlexKV YAML 配置

最小配置只要一行 CPU pool 大小。SWA **不需要**在 YAML 里配任何东西——DSv4 被自动
识别后 FlexKV 自动建 `SWAPoolConfig(enabled=True, window_blocks=1, num_slots=1024)`。

```yaml
# $FLEXKV_CONFIG
cpu_cache_gb: 64
```

可选字段（`example_config_mp.yaml` 里有全量注释）：

- `ssd_cache_gb`：设成 **严格大于** `cpu_cache_gb` 即开 SSD spill 层（否则 CacheConfig 报错）；
  不用 SSD 就别写。SWA 窗口本身是 CPU-only（默认 `num_ssd_slots=0`）。
- `kv_cache_dtype`：SGLang 用 `--kv-cache-dtype auto` 时 FlexKV 猜不出 dtype，在这里显式给。
- `enable_p2p_cpu` / `enable_p2p_ssd`：**单机不要开**——它们是跨节点 peer 复用开关，会连带
  拉起 Redis/Mooncake。单机 radixshmem 走本地树，两者保持默认 false。

> 关于 SWA pool 容量：`num_slots` 默认 1024，`window_blocks=1`（DSv4），1024 ≥ 1
> 满足启动检查。启动时若 `num_slots < window_blocks` 会直接
> `ValueError(... cannot hold one N-block SWA window ...)`——正常 DSv4 不会触发。

### 3.3 启动 SGLang 服务

```bash
cd $SGLANG_ROOT
CUDA_MPS_PIPE_DIRECTORY=/nonexistent/mps \
FLEXKV_ENABLE_MPS=0 \
FLEXKV_RADIX_SHMEM=1 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
$VENV/bin/python -m sglang.launch_server \
    --model-path $MODEL \
    --port 30000 \
    --tp-size 4 \
    --enable-flexkv \
    --flexkv-config-file $FLEXKV_CONFIG \
    --mem-fraction-static 0.75 \
    --max-running-requests 8 \
    > /tmp/sglang.log 2>&1
```

DSv4-Flash 有两种权重：FP8 版（约 274 GB）和 expert 为 fp4 的版本（约 149 GB）。H20 没有 FP4 硬件，用 FP8 版；
tp=4 时每卡权重 68.8 GB，`--mem-fraction-static` 至少 0.75（0.45 装不下权重）。权重在 page cache 里时约 25 秒加载完。

| 参数 | 必填 | 说明 |
|---|---|---|
| `--enable-flexkv` | ✅ | 把默认 RadixCache 路由到 FlexKV。单独用它就够，不必再加 `--radix-cache-backend=flexkv`（等价）。 |
| `--flexkv-config-file <path>` | ✅ | FlexKV YAML 路径（等价于 `FLEXKV_CONFIG_PATH` env）。 |
| `--tp-size N` | 看需要 | 单机设成用的 GPU 数。FP8 DSv4-Flash 在 H20 上至少 tp=2，推荐 4。 |
| `--page-size` | ❌ | **不要手动传**。DSv4 自动强制 page_size=256（多处 `assert page_size == 256`，传别的值直接崩）。 |

关于 SWA 自动化（无需任何开关）：

- SGLang 检测到 HF architectures 是 DeepSeek-V4 → `is_deepseek_v4_arch=True` →
  走 hybrid-SWA 路径 → 选 `FlexKVHybridRadixCache`。
- FlexKV `post_init_from_sglang_config` 看到 `is_dsv4` → 自动建
  `SWAPoolConfig(enabled=True, window_blocks=1, bytes_per_token_per_layer=585)`。
- 585 = `ceil(256×584, 576)×576 / 256`，匹配 DSv4 SWA GPU buffer 的对齐 stride。

确认服务起来：

```bash
grep -E "Constructed SWAPoolConfig|fired up|Connector ready" /tmp/sglang.log | tail -3
# 期望看到 [FlexKV sglang] Constructed SWAPoolConfig for DSv4: ... window_blocks=1 ...
#          The server is fired up and ready to roll!
```

---

## 4. 验证

### 4.1 基本 cache 命中

```bash
# 第一次：冷启动，FlexKV 存下 prefix
curl -s http://127.0.0.1:30000/generate -X POST -H "Content-Type: application/json" \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":5,"temperature":0}}'

# 清掉 GPU radix（FlexKV CPU pool 保留数据），再发同一条
curl -s http://127.0.0.1:30000/flush_cache -X POST
curl -s http://127.0.0.1:30000/generate -X POST -H "Content-Type: application/json" \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":5,"temperature":0}}'
```

> 上面 5 个 token 的示例 prompt 实际上**永远命中 0**：DSv4 一页 256 token，凑不满一页
> FlexKV 不会存。真要看命中，prompt 至少几百 token（例如一段话重复十几遍）。

看第二次响应的 `meta_info`：`"cached_tokens_details": {"device": 0, "host": N}`——
`host: N`（N>0）说明字节从 FlexKV 的 CPU pool 搬回来了。服务日志里也会有对应的
`H2D transfer request: ... GB ... GB/s`。

### 4.2 确认 SWA 真的在跑

DSv4 的 SWA 是自动的，确认它生效看三处：

1. 启动日志有 `Constructed SWAPoolConfig for DSv4: ... window_blocks=1, num_slots=1024, enable_swa_transfer=True`。
2. flush 后重发长 prompt，服务日志里那次 GET 的 H2D 有两条 transfer：N 个 Full block 加 **1 个 SWA block**
   （`direction=H2D blocks=1 ... data_size=0.0076GB`，256 token × 43 层 × 585 B）。
3. 正确性用 §4.4 的方法验证。

> `FLEXKV_ENABLE_SWA_TRANSFER=0` **不能**当正确性 A/B：关掉后 PUT 不存窗口，joint 查询直接 miss，
> host 命中变 0，输出当然一致。它只用来确认"SWA 通路在开时活跃"。

### 4.3 单测（不依赖模型权重）

```bash
cd $FLEXKV_ROOT
CUDA_MPS_PIPE_DIRECTORY=/nonexistent/mps \
  $VENV/bin/python -m pytest tests/test_radix_shmem_engine.py -q
# 覆盖 radixshmem 引擎 + SWA component（joint query / all-or-none take / W 泛化）
```

### 4.4 SWA 正确性验证方案

目标：证明 FlexKV 恢复回 GPU 的 SWA 窗口页**字节正确**，且**被模型真的读到**。分三层，
前两层不需要模型。

| 层 | 工具 | 判据 | 结论 |
|---|---|---|---|
| 元数据 | `tests/test_radix_shmem_engine.py` | 窗口 slot / `swa_start` / joint 查询 | 31 passed |
| 字节（合成数据） | `tests/test_layerwise_dsv4_multi_group_swa_roundtrip.py`、`tests/test_swa_control_plane_e2e.py` | GPU→CPU→GPU 逐字节相等 | 通过 |
| 字节（真模型） | `scripts/swa_verify/swa_bytes_compare.py` + dump 钩子 | flush 前 GPU 页 vs FlexKV 恢复页，sharded 谓词（见 4.4.5） | 24/24 满足拼接谓词，22/24 逐位相同，差异已解释 |
| 模型 | 下面的 teacher-forced KL 配对测试 + 两个负向对照 | host 恢复 vs device 命中的 logprob | 一致（见结果表） |

#### 4.4.1 为什么不能用"生成结果相等"

- DSv4 长文本贪心解码本身 run-to-run 不确定：同一请求连发两次、都是 device 命中，24 条 LongBench
  prompt 里只有 3 条在 128 token 内完全相同。自回归会把一次近似打平的翻转放大到后面所有 token。
- sglang `kl_test_utils` 原方法用 `logprob_start_len=0` 给整段打分，sglang 会把可复用前缀压到 0
  （`schedule_batch.py` `_compute_max_prefix_len`），`cached_tokens=0`，FlexKV 根本不参与。
- 上游自带的 `storage/flexkv/verify_outputs.py`（linhu-nv，为 Qwen3-8B page 1 写的）三个 prompt 只有
  5/19/127 token，不满一页，DSv4 上命中恒为 0，且命中率只打印不断言，`Total mismatches: 0` 没有意义。

#### 4.4.2 有效方法：页对齐 teacher-forced 打分

每条 prompt：截到 256 的整数倍 → 生成 128 token 作为固定续写 → 三次对 `prompt+续写` 打分，
`logprob_start_len=len(prompt)`，`max_new_tokens=0`，batch=1：

1. `flush_cache` 后打分：prompt 前缀从 FlexKV host 恢复（含 SWA 末页）。
2. 紧接着再打分：device 命中。
3. 再打分一次：device 命中，作噪声底。

比较续写 128 个 token 的 logprob：KL(host, device) 与 KL(device, device) 配对比较。
两个要点：

- **必须页对齐**。否则 prompt 尾部最多 255 个未缓存 token 会被重新 prefill，它们把续写的 128 窗口和
  恢复页完全隔开，测试对 SWA 无感（我们第一版就栽在这里）。
- **batch=1**，让 host / device 两次走完全相同的 kernel 路径，差异只剩 KV 内容。

核心代码（复用 sglang 的 `kl_test_utils` 原语；LongBench-v2 首次下载需要代理）：

```python
import numpy as np
from sglang.test.kl_test_utils import get_input_ids, _generate, _flush_cache
BASE, MODEL, NEW = "http://127.0.0.1:30000", "/path/to/DeepSeek-V4-Flash-FP8", 128
ids = [p[:len(p)//256*256] for p in get_input_ids(MODEL, max_prompt_tokens=3000,
                                                  num_samples=24, trust_remote_code=True)]
def score(seq, start):
    r = _generate(BASE, [seq], 0, return_logprob=True, logprob_start_len=start)[0]
    d = r["meta_info"].get("cached_tokens_details") or {}
    return (d.get("host", 0), d.get("device", 0)), \
           [x[0] for x in r["meta_info"]["input_token_logprobs"][-NEW:] if x[0] is not None]
def kl(a, b):
    a, b = np.array(a), np.array(b); l = a - b; return float(np.mean((np.exp(l) - 1) - l))
for p in ids:
    cont = _generate(BASE, [p], NEW)[0]["output_ids"][:NEW]; seq = p + cont; L = len(p)
    _flush_cache(BASE); ch, lh = score(seq, L)      # host 恢复，断言 ch[0] > 0
    cd, ld = score(seq, L); cd2, ld2 = score(seq, L)  # device 命中 ×2，断言 cd[1] > 0
    print(L, kl(lh, ld), kl(ld, ld2))
```

判据（配对）：`mean KL(host,device)` 不超过 `mean KL(device,device)` 的 1.5 倍，且逐位相同的 prompt
数不少于 device 对 device 的。短 prompt（约 2000 token 以下）三次打分应逐位相同。

#### 4.4.3 负向对照（证明测试测的是 SWA）

在 `sglang/.../storage/flexkv/flexkv_hybrid_radix_cache.py` 里加了两个环境变量门控的调试钩子
（默认关闭，当前只在工作区未提交）：

| 环境变量 | 行为 | 期望 |
|---|---|---|
| `SGLANG_DEBUG_ZERO_SWA_TAIL=1` | device 命中时把命中末页对应的 SWA pool 页清零 | KL 爆炸 → 证明 attention 确实读缓存末页的 SWA，测试敏感 |
| `SGLANG_DEBUG_ZERO_SWA_ON_FLUSH=1` | `flush_cache` 时清空整个 SWA pool | host 恢复仍与 device 一致 → 证明正确窗口只能来自 FlexKV，且写在 attention 读的 slot 上 |

第二个对照是必需的：不清零时 flush 释放的 SWA slot 会被原样分回来，里面残留着 flush 前的正确数据，
会掩盖 FlexKV 写错位置的 bug。

> **不要用 FlexKV 侧"H2D 从 src+1 slot 读"做注错**（我们试过并回退）。SGLang 对同一路径 PUT 两次
> （prefill 结束 + 请求结束），第二次 `take` 到的新 slot 先被 D2H 写入同样字节、再被 insert 判重回收，
> `src+1` 里就是一份一模一样的副本，注错无效。tp=1 单次 PUT 的 e2e 测试能测出来，服务上测不出来。

#### 4.4.4 实测结果（24 条 LongBench-v2，1280 到 4352 token，tp=4）

| 配置 | KL(host,device) 均值 / 最大 | KL(device,device) 均值 / 最大 | 逐位相同 host-dev / dev-dev |
|---|---|---|---|
| 正常 | 0.0079 / 0.060 | 0.0058 / 0.026 | 7/24 / 7/24 |
| `ZERO_SWA_ON_FLUSH=1` | 0.0051 / 0.026 | 0.0068 / 0.039 | 7/24 / 7/24 |
| `ZERO_SWA_TAIL=1` | 2.5e3 / 3.7e4 | 0.021 / 0.17 | 0/24 / 7/24 |

host 对 device 与 device 对 device 逐 prompt 配对跟随，逐位相同的是同样 7 条短 prompt；末页清零后
24 条全部改变，KL 高出噪声 5 个数量级。KL 的绝对值随每次生成的续写不同而波动，只看组间相对关系。
结论：**FlexKV 恢复的 SWA 窗口字节正确且被模型消费**。

#### 4.4.5 字节级比对（真模型）与 sharded D2H 的发现

比 logprob 更直接的判据：restore 是一次拷贝，恢复回 GPU 的 SWA 页应与 offload 前 GPU 上的原页相等。
参照物是**原页**，不是"不用 radixshmem 的 FlexKV"（两种后端共用同一套搬运，且同一页会落在不同 CPU slot）。

做法（`scripts/swa_verify/swa_bytes_compare.py`，服务需开 `SGLANG_DEBUG_SWA_DUMP_DIR=<dir>` 和
`SGLANG_DEBUG_ZERO_SWA_ON_FLUSH=1`）：每条 prompt 加 256 个随机 token 前缀保证未存过 → 生成（触发 store）→
device 命中，钩子按 token 序列哈希 + TP rank dump 命中末页的 SWA pool 页（"前"）→ flush（pool 清零）→
host 恢复 → device 命中再 dump（"后"）→ 逐 rank 逐层比对。

**发现**：24 条里 22 条四 rank 43 层逐字节相等；2 条 L=1536 的第 0 层相等、其余 42 层不等，与解码长度无关，
L=1792 对照全过。根因是两件事叠加：

1. L=1536 时 **四个 TP rank 在 flush 前各自的 SWA 页就不逐位相同**（仅第 0 层一致，其余约 25% 字节差 1 个值，
   allreduce 级舍入抖动）。这是 TP 推理的正常现象，不是 SGLang bug。
2. FlexKV D2H 默认 `FLEXKV_KV_SHARED_ACROSS_RANKS_MODE=sharded`：假定 MLA KV 各 rank 逐位相同，把每页按字节切成
   tp 段，rank i 只拷第 i 段。rank 分歧时存下的是四段拼接，恢复后广播给所有 rank。逐字节验证：恢复页第 i 段 ==
   rank i 原页第 i 段，43 层 × 4 rank 全部成立。

后果是舍入级的，输出影响在噪声内（与 4.4.4 一致）。但"恢复页 == 某 rank 原页"在 sharded 模式下不是正确期望，
脚本判据已改为上述拼接谓词；`sharded` 的隐含假设值得在 FlexKV 侧文档化，`all_write` 是替代模式。

#### 4.4.6 复现步骤

```bash
# 1. 按 §3.3 启动服务，多加环境变量（需要 4.4.3 / 4.4.5 的钩子在 sglang 工作区）
SGLANG_DEBUG_ZERO_SWA_ON_FLUSH=1 SGLANG_DEBUG_SWA_DUMP_DIR=$WORK/swa_dump ... python -m sglang.launch_server ...
# 2. 字节级（脚本从 $MODEL 读 tokenizer，dump 目录须与服务一致）：
SGLANG_DEBUG_SWA_DUMP_DIR=$WORK/swa_dump $VENV/bin/python scripts/swa_verify/swa_bytes_compare.py 24   # 约 100 s，dump 约 1.2 GB
# 3. logprob：
$VENV/bin/python scripts/swa_verify/kl_tf_aligned.py        # 约 50 s；首次需 HTTPS_PROXY 下载 LongBench-v2
# 4. 可选：换 SGLANG_DEBUG_ZERO_SWA_TAIL=1 重启再跑 3，确认 KL 爆炸
```

脚本在 `scripts/swa_verify/`（环境变量见其 README）。尚未做的：正式化并接 CI；kv-canary 的 DSv4 适配器补全（长期字节级护栏）；
`mixed_prefix_gsm8k` 两遍法作为发版粗门。

---

## 5. 注意事项与限制

- **仅单机**：radixshmem 的 SWA 匹配强制 local-only；跨节点 SWA 复用尚未支持
  （`FULL|SWA` 且非 local-only 会 `ValueError`）。跨节点只对 Full KV 有效（那需要另一套
  etcd/RDMA/mooncake 配置，见 `radixshmem_cross_node.md`）。
- **SWA 只对 DSv4 自动开**：由 HF architectures 判定（`is_deepseek_v4_arch`）。
  其他滑窗模型（gpt-oss/gemma）在 SGLang 侧走的不是这条 DSv4 路径，本文不覆盖。
- **窗口驱逐后不补发**（已接受的 MVP 限制）：某前缀的 Full 路径仍缓存、但其 SWA 窗口被
  SWA pool 独立 LRU 驱逐后，该前缀的重复 PUT 不会补发窗口，直到 Full 路径本身老化。期间该
  前缀的 SWA-aware GET 会 miss。
- **不能和 `enable_remote`（第三方 PCFS 存储）同开**：`KVManager` 直接报错。
- **不要开 `enable_p2p_cpu/ssd`**：那是跨节点开关，单机不需要，还会拉起 Redis/Mooncake。
- **c_ext `.so` 是全局的**：cu130 版会覆盖 cu128 版，多环境需各自重编或备份。
- **MPS**：CUDA 前加 `CUDA_MPS_PIPE_DIRECTORY=/nonexistent/mps`，并 `FLEXKV_ENABLE_MPS=0`。
  杀任何 MPS 守护进程之前先停掉所有可能是它客户端的服务（`/proc/<pid>/fd` 看不出客户端关系），
  否则客户端进程全部 `cudaErrorMpsRpcFailure`。

---

## 6. 排错速查

| 现象 | 原因 |
|---|---|
| `undefined symbol: ...c10...cuda...` import c_ext 时 | c_ext 是别的 torch 版本编的，对当前 venv 的 torch 重编（§2.2） |
| `cuda.is_available()` 为 False / `Error 805 ... MPS` | 系统默认 MPS 管道不可用，加 `CUDA_MPS_PIPE_DIRECTORY=/nonexistent/mps`（§2.4）；若某组 GPU 单独不可用，查 `pgrep -af nvidia-cuda-mps` 有没有游离守护进程占着它们 |
| `cudaErrorMpsRpcFailure` 所有 rank 同时崩 | 服务是某个 MPS 守护进程的客户端而守护进程被杀了；重启服务并加 `FLEXKV_ENABLE_MPS=0` |
| `nvidia-cuda-mps-control` 越跑越多、每个绑一个 `/tmp/mps_*` 目录 | 没设 `FLEXKV_ENABLE_MPS=0`，FlexKV 每次启动都在 `CUDA_MPS_PIPE_DIRECTORY` 下起一个；`echo quit \| CUDA_MPS_PIPE_DIRECTORY=<dir> nvidia-cuda-mps-control` 逐个退掉 |
| `assert page_size == 256` 启动崩 | 手动传了 `--page-size` 非 256；DSv4 别传这个参数（§3.3） |
| `cannot hold one N-block SWA window` 启动 ValueError | SWA `num_slots < window_blocks`，正常 DSv4 不会触发；异常时调大 num_slots（§3.2） |
| 日志无 `Constructed SWAPoolConfig` | 模型未被识别为 DSv4，或 `--disable-hybrid-swa-memory` 关了 hybrid → 不建 SWA pool |
| `cached_tokens_details host:0` 恒为 0 | FlexKV 没命中：确认 `--enable-flexkv` 生效、`FLEXKV_RADIX_SHMEM=1` 已 export、flush 后再发 |
| `ModuleNotFoundError: flexkv/shmradix/expiring_dict/...` | venv 里 editable/runtime 依赖没装全（§2.3） |
| `shm_open attach failed` / `Timed out attaching to shm radix region` | 同机多实例没给不同 `FLEXKV_SHM_RADIX_ID`；或 `/dev/shm` 里有陈旧 region，清掉重启 |
