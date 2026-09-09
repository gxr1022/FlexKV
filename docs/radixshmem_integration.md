# FlexKV × radixshmem 集成（多 DP 路径）

将 FlexKV 的多 DP 路径从 "N 个 CE 客户端 → zmq → 单 KVServer 进程" 改造为 "N 个 CE 在自己地址空间里 attach 同一段共享 shm radix tree、并行查询，1 个共享 TE 通过 N 条 ShmChannel 接收 transfer graph"。结果是端到端 mean 时延快 **~3.4×**（中位）/最坏 ~2×、QPS 高 **~80%**（中位）/最坏 ~50%、命中率不变，并且消除 baseline 路径下高 QPS 时观察到的多种 hang/race（测量细节见 git 历史中的两阶段 benchmark）。

## 目录

1. [radixshmem 是什么 / 解决了什么](#1-radixshmem-是什么--解决了什么)
2. [架构](#2-架构)
3. [编译与依赖](#3-编译与依赖)
4. [启用方法 / 配置项](#4-启用方法--配置项)
5. [容量估算](#5-容量估算以-qwen3-8b8-dp-为例)
6. [已知坑](#6-已知坑)

---

## 1. radixshmem 是什么 / 解决了什么

### 1.1 原 multi-DP 路径的问题

FlexKV 原本的 multi-DP 路径走 `server_client_mode=True`：N 个推理引擎 scheduler 进程里各跑一个 `KVDPClient`，所有请求经 zmq PUSH/PULL 汇集到唯一的 `KVServer` 进程，由 `KVServer.run()` 单线程串行 dispatch 到 `KVTaskEngine` → `GlobalCacheEngine` → 单棵 RadixTree。

这个架构在 multi-DP 场景下有两个独立瓶颈：

- **N→1 query 漏斗**：所有 DP 的 `get_match` / `put_match` 都被串行化到一个 zmq 事件循环。`get_match` 单次 ~440 µs（含 zmq 往返 + 排队），高 QPS 下时延爆涨。
- **跨 DP 共享 cache 是 KVServer 进程的副产物**：cache 共享是好事（任意 DP put 进的都能被任意 DP get 出来），但代价是必须用单个 server 进程承接所有访问。

### 1.2 radixshmem 做了什么

[radixshmem](https://gitlab-master.nvidia.com/zhuofanl/radixshmem/-/tree/dev?ref_type=heads)（同事写的库，`dev` 分支）把一棵 **radix tree** 整体放在 POSIX 共享内存里（`shm_open` + `mmap`），用 process-shared `pthread_rwlock` 协调，自带 buddy allocator + slot mempool。多个进程 attach 到同一段 shm 之后，可以**并行**做 prefix match / insert，**不需要任何中心进程**。

FlexKV 端做了三件事：

1. **新增 `CacheEngineRadixShmem` 后端**（`flexkv/cache/radix_shmem_engine.py`），跟原有 `CacheEngineAccel` 接口对齐，让 `GlobalCacheEngine` 透明 dispatch。
2. **去掉 KVServer**，多 DP 模式下每个 DP scheduler 进程在自己地址空间里持有 `KVTaskEngine`，所有 DP 通过 shm 共享同一棵 radix tree。
3. **新增 `ShmChannel` (SPSC ring + futex wake)**（`flexkv/transfer/shm_channel.py`）替换 zmq，让 N 个 CE 进程跟唯一的 TE 进程通讯。每 CE 一条独立 channel。

**两条独立维度的价值**：

| 维度 | baseline | shmradix |
|---|---|---|
| **Cache 跨 DP 共享** | ✅ 通过单 KVServer 共享 | ✅ 通过共享 shm tree |
| **Query 路径无 N→1 漏斗** | ❌ 所有查询过单 server | ✅ 每 DP 在自己进程并行 |

vllm 自带 `--enable-prefix-caching` 是 per-DP GPU-local 的，**两条维度都没有**（路由命中只有 1/DP 概率），所以在 multi-DP 高 QPS 下比 baseline FlexKV 还慢一个量级。

---

## 2. 架构

```
DP scheduler 进程 0   ...   DP scheduler 进程 N-1
  ┌──────────────────┐         ┌──────────────────┐
  │ KVDPClient (本地)│         │ KVDPClient (本地)│
  │ KVTaskEngine     │   ...   │ KVTaskEngine     │
  │ GlobalCacheEngine│         │ GlobalCacheEngine│
  │  (radixshmem CE) │         │  (radixshmem CE) │
  └────────┬─────────┘         └────────┬─────────┘
           │                            │
        ┌──┴────────────────────────────┴──┐
        ▼                                  ▼
  /dev/shm/shmradix_<id>_cpu       (radix 索引)   ── 所有 CE 直接读写
  /dev/shm/shmradix_<id>_cpu_data  (SlotStore = CPU KV 池) ── TE / worker 按名字 attach
        │                                  │
        ▼ ShmChannel × N (SPSC ring + futex)
  ┌──────────────────────────────────────────┐
  │  TransferEngine 进程（单实例）           │
  │  - StorageEngine (SlotStore 视图 / GPU)  │
  │  - 多通道 selector loop                  │
  └──────────────────────────────────────────┘
```

总进程数：`N (DP scheduler) + 1 (TE) + 1 (radix-server)`。**KVServer 进程被完全去掉**。

radix-server 是 radixshmem 的 `RadixServer`：DP-0 的 KVManager 把它作为子进程拉起
（`FLEXKV_RADIX_SERVER_LAUNCH_MODE=embedded`，默认），或者由运维预先起好（`external`）。它拥有三样东西：

- radix 索引 shm（`/dev/shm/shmradix_<id>_cpu`）；
- SlotStore shm（`/dev/shm/shmradix_<id>_cpu_data`）：**这就是 FlexKV 的 CPU KV 池**，一个 slot 一个 block，
  slot id 就是 block 下标。FlexKV 不再自己 `torch.empty` 一块 CPU buffer 再用 `TensorSharedHandle`
  传给 worker；TE 和每个 transfer worker 按名字 attach 这块 SlotStore，把池当成普通 tensor 做 H2D/D2H；
- 跨节点时的 mooncake 传输引擎和 etcd 登记（见 `radixshmem_cross_node.md`）。

所有 DP 进程用 `shmradix.RadixClient(name)` 挂载它：索引操作（query / insert / allocate_slots）直接走 shm，
gRPC（`/dev/shm/shmradix_<id>_cpu.sock`）只在挂载握手和跨节点拉取时用到。

跟原架构的对比：

| 组件 | baseline | shmradix |
|---|---|---|
| `KVServer` 子进程 | ✅ 单进程 / zmq 串行 | ❌ 不需要 |
| `KVTaskEngine` | 在 KVServer 进程里 | 每个 DP 进程里一份 |
| `GlobalCacheEngine` | 在 KVServer 进程里 | 每个 DP 进程里一份 |
| RadixTree（实际数据） | 在 KVServer 的 Python heap 里 | 在 POSIX shm，所有 DP attach |
| CE↔TE 通信 | mp.Pipe (single CE) 或 zmq+KVServer (multi DP) | **N 条 ShmChannel**（SPSC ring + futex） |
| dispatch | `KVServer.run()` 单线程 | `_TEShmDispatcher` 用 selector 在 N 个 channel 上 multiplex |

---

## 3. 编译与依赖

### 3.1 radixshmem 库

源码（独立仓库，同事维护）：<https://gitlab-master.nvidia.com/zhuofanl/radixshmem/-/tree/dev?ref_type=heads>，本文档对应 `dev` 分支。

从源码编译：

```bash
git clone -b dev ssh://git@gitlab-master.nvidia.com:12051/zhuofanl/radixshmem.git
cd radixshmem
# 系统依赖：libxxhash-dev、liburing-dev、cmake、pybind11
#   - libxxhash-dev / liburing-dev 没有就从 host 拷 header，或用 mooncake_transfer_engine.libs 里 ship 的 liburing.so.2
#   - cmake、pybind11 可以 pip install
# 主仓库构建
mkdir build && cd build && cmake .. && make -j
# Python binding（产物：python/shmradix/_core.cpython-3xx-x86_64-linux-gnu.so）
cd ../python && pip install -e . --no-build-isolation
```

容器场景下需要把构建产物 `python/shmradix/_core.cpython-<py>-x86_64-linux-gnu.so` 放到 `PYTHONPATH=/work/radixshmem/python` 能看到的位置（bind-mount 整个仓库即可）。Python 3.12 + torch 2.10 + glibc 2.35 已验证。

### 3.2 FlexKV

```bash
cd FlexKV
pip install -e . --no-build-isolation
```

依赖：Cython（构建时）、torch、numpy、xxhash、liburing、expiring_dict、zmq、redis 等。详见 `requirements.txt`。

### 3.3 运行时 Python path

```bash
export PYTHONPATH=/work/FlexKV:/work/radixshmem/python:$PYTHONPATH
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
```

---

## 4. 启用方法 / 配置项

### 4.1 触发 shmradix 路径

```bash
export FLEXKV_RADIX_SHMEM=1
export FLEXKV_SHM_RADIX_ID=<unique-id>     # 同机多 vllm 实例时区分用
export FLEXKV_DP_SIZE=8                    # 必须跟 vllm --data-parallel-size 一致
export FLEXKV_CPU_CACHE_GB=200             # CPU cache 大小，见 §5
```

`FLEXKV_RADIX_SHMEM=1` 时 `KVManager` 走 `use_radix_shmem=True` 分支，每个 DP 进程内部构造 `KVTaskEngine`，绕过 `KVServer.create_server()`。Bootstrap DP（`instance_id=0 && dp_client_id=0`）拉起本节点的 radix-server 子进程（索引 + SlotStore = CPU 池）并 spawn 共享 TE；其它 DP attach。

radixshmem 模式只有 CPU 层：`ssd_cache_gb` 必须为 0，`enable_remote` 不能开。

radix-server 相关的可选项（都有默认值）：

```bash
export FLEXKV_RADIX_SERVER_LAUNCH_MODE=embedded   # embedded：DP-0 拉起子进程；external：挂载运维预先起好的 radix-server
export FLEXKV_RADIX_ENDPOINT=                     # radix-server gRPC 端点，空 = unix:///dev/shm/shmradix_<id>_cpu.sock
export FLEXKV_RADIX_PREFAULT=1                    # 启动时预触 SlotStore 所有页（D2H 首次不再缺页；启动多花几十秒）
export FLEXKV_RADIX_DATA_POOL_RATIO=8             # 索引 DataPool 大小系数
```

跨节点（`FLEXKV_RADIX_WORLD_SIZE > 1`）的变量见 `radixshmem_cross_node.md`。

### 4.2 vllm 启动参数

```bash
vllm serve <model_path> \
    --tensor-parallel-size 1 --data-parallel-size 8 \
    --port 31002 \
    --max-num-seqs 256 --max-num-batched-tokens 8192 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.28 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
```

| 参数 | 必填 | 解释 |
|---|---|---|
| `--data-parallel-size N` | ✅ | 跟 `FLEXKV_DP_SIZE` 对齐 |
| `--no-enable-prefix-caching` | 强烈建议 | 关掉 vllm 自带 prefix cache，让 FlexKV 路径独占；否则两层 cache 混在一起难评估 |
| `--kv-transfer-config '{...}'` | ✅ | 启用 FlexKV connector |
| `--gpu-memory-utilization` | 看卡 | 留 KV cache 余量，见 §5 |

### 4.3 验证生效

启动几个 request 后，看 vllm log：

```bash
grep "FlexKV Hit Ratio" /path/to/vllm.log | tail -3
# 应输出类似：
# [FlexKV] Metric of Recent 100 Requests: ... FlexKV Hit Ratio: 87.45%, Get/Put Token Ratio: 105.32%.

grep "H2D transfer request" /path/to/vllm.log | tail -3
# 应输出类似：
# [FLEXKV] INFO ... H2D transfer request: 60 finished transfer data size: 0.013 GB ... transfer bandwidth: 6.58 GB/s
```

- `Hit Ratio` < 30 %（在你预期命中场景下）= cache 容量不够 / PUT 在 race
- `Get/Put Token Ratio` < 50 % = PUT 大量失败
- `H2D transfer request: ...` 出现 = KV 真实从 CPU 搬到 GPU（不是仅 metadata 命中）

### 4.4 可选诊断 env（**线上关掉**）

```bash
export FLEXKV_TIME_QUERY=50      # 每 50 次 get_match/put_match 聚合一行时延 log
export FLEXKV_NUM_LOG_INTERVAL_REQUESTS=100   # 命中率日志频率
```

---

## 5. 容量估算（以 Qwen3-8B、8 DP 为例）

```
per_token_KV = num_kv_heads × head_dim × 2(K+V) × dtype_bytes × num_layers
             = 8 × 128 × 2 × 2 × 36 = 144 KB

per_block_KV = per_token_KV × block_size (16) = 2.25 MB
```

| 项 | 大小 | 备注 |
|---|---:|---|
| **CPU cache 配置** (`FLEXKV_CPU_CACHE_GB`) | 200 GB | 全部落在 radix-server 的 SlotStore（`/dev/shm` 或 hugetlbfs） |
| GPU 每卡物理容量 | ~143 GB | H20 |
| vllm 占用 (`--gpu-memory-utilization 0.28`) | ~40 GB / 卡 | |
| 模型权重 (Qwen3-8B bf16) | ~16 GB / 卡 | |
| **每卡 GPU KV 空间** | ~24 GB | = 40 − 16 |
| 每卡 GPU KV 容量 | ~170 K tokens / ~10 600 blocks | 24 GB / 144 KB |
| 8 DP 合计 GPU KV | ~192 GB | 每 DP 独立、不共享 |



---

## 6. 已知坑

| 坑 | 现象 | 处理 |
|---|---|---|
| 2. `pkill -f "vllm serve"` 杀不全 | 留下 `VLLM::EngineCore_*` 子进程 | 同时 `pkill -f "VLLM::"` |
| 3. vllm v1 默认路由不是 prefix-aware | r2 落 r1 当时 DP 的概率仅 1/N | 这正是 shmradix 的价值；不要试图修 vllm 端 |
| 5. `docker stop` 报 "did not receive an exit event" | 看着像没停 | 等 30-90 秒，容器自然变 Exited(137) |
| 6. vllm 不把所有 env 传给 engine 子进程 | 自定义 `FLEXKV_*` env 在 engine 进程里 `os.getenv` 拿不到 | env 的 default 行为只继承一部分；FlexKV 已用 `FlexKVConfig.from_env` 序列化进 config；若新加 env 需要 lazy resolve |
| 7. `--shm-size` 一旦设定，container restart 不能改 | 修不了 | `docker rm` 重建 |
| 8. `FLEXKV_SHM_RADIX_ID` 重复 | 多个 vllm 实例 attach 同一段 shm tree，状态串了 | 每个实例用唯一 ID |
| 8b. `/dev/shm` 容量 | CPU KV 池整体是 radix-server 建在 `/dev/shm` 的 SlotStore：容器 `--shm-size` 小于 `cpu_cache_gb` 时 radix-server 起不来 | `--shm-size` ≥ `cpu_cache_gb` + 余量；或 `use_hugepage_cpu_buffer: true` + `FLEXKV_HUGETLBFS_DIR` 走 hugetlbfs |
| 8c. `attached radixshmem regions do not match FlexKV's configuration` | TE 算出的 CPU 块字节数 / 块数与 DP-0 启动 radix-server 时不同 | 日志里有两边的数值；多 PP 不均分或 DSv4 sidecar 组没在 KVManager 之前记录时会出现 |
| 9. CPU cache 跨 DP 共享、但 GPU KV cache 是 per-DP | 容易混淆容量估算 | 见 §5，CPU 是 200 GB 共池，GPU 是 8 × 24 GB 独立 |
| 10. `flexkv/transfer/worker.py` 里曾在 `_transfer_impl` 后 `torch.cuda.synchronize()` | 把 D2H/H2D op 串行成 device-wide drain，R2 mean / p95 / p99 各 ~10-12% degrade | device-wide sync 已移除；`GPUCPUTransferWorker._transfer_impl` 现把 `sync=True` 下推给 C++ `transfer_kv_blocks`，收尾只做 stream-scoped 的 `cudaStreamSynchronize(stream)`（`csrc/transfer.cu:300`），不再 drain 整个 device。下游 worker 队列自带 stream-aware 排序，多余的 device sync 没有保护任何 invariant。NIXL worker 的 sync（`worker.py:2411`，由 PR #142 引入）属于另一条路径，与本项无关 |

---

## 附录：本次工作中修复的几处 race

为达到上面的 0-hang 表现，本轮调试一并修复了 4 类 race（每条都在 `flexkv/...` 里有对应改动）：

| race | 修复 |
|---|---|
| `batch merged graph_id` 跨 DP 撞 ID（`merge_to_batch_graph` 用 batch_id 覆盖全局 disjoint 分配） | 删除 `set_graph_id(batch_id)`，让 merged graph 用 `TransferOpGraph()` 的 disjoint range（`flexkv/common/transfer.py:399`） |
| `ShmChannel.result_send` 满时抛异常 → TE result 线程死 → 丢 completion | 改成永久 spin + 周期错误日志（`flexkv/transfer/shm_channel.py:344`），ring 容量 256 → 1024 |
| `CacheEngineRadixShmem.match(lock=True)` 的 ref 每次泄漏一次 → 长跑后节点钉死无法 evict | 引入 `MatchResultAccel.pre_locked_node` 字段，4 个 `_impl` 方法所有 success/early-return 路径显式释放（`flexkv/cache/cache_engine.py`） |
| `KVTaskManager._pending_release_tasks` 在多 DP 共用 KVTaskEngine 下被任意 DP 抢先清理 → `NOTFOUND` race | 上游 PR #164（commit `0d6b1ed`）通过 `shed_heavy_resources()` + owner-observed release 修复 |
| `submit_send` 只 bump channel wake、TE idle 时等的是 ctrl wake → 多等 100ms futex timeout | `submit()` / `submit_batch()` 后显式 `_ctrl.notify()`（`flexkv/transfer/shm_channel_handle.py`） |

这些都已 commit 在 `main` / 当前分支。

---

## 联系

- 主仓库：[`FlexKV`](https://github.com/taco-project/FlexKV)
- radixshmem 依赖：<https://gitlab-master.nvidia.com/zhuofanl/radixshmem/-/tree/dev?ref_type=heads>（`dev` 分支）
- 跨节点、SWA：`radixshmem_cross_node.md`、`sglang_flexkv_radixshmem_swa.md`
