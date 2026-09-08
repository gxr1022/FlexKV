# radixshmem 跨节点 KV reuse（多机 prefix 复用）

本文讲的是**让不同机器上的 FlexKV 互相复用 CPU 层的 KV cache**：它是什么、怎么装、怎么配、怎么验证。

跨节点路径完全由 radixshmem 承担：索引（谁持有这段前缀）、内存（CPU KV 池就是 radixshmem 的
SlotStore）、传输（radix-server 之间的 RDMA READ）。FlexKV 侧不再有自己的 P2P worker、mooncake
引擎或 Redis 地址簿。本文对应 radixshmem 的 `RadixServer` / `RadixClient` 接口（transfer-server 分支及之后）。

---

## 0. 名词速查

| 名词 | 一句话解释 |
|---|---|
| radix-server | 每个节点一个进程：拥有 radix 索引 shm、SlotStore shm（本节点的 CPU KV 池）、mooncake 传输引擎和 etcd 登记，通过 gRPC（`/dev/shm/<name>.sock`）服务本机的 RadixClient。 |
| node | 一套 FlexKV：一个 radix-server + 一个 transfer engine（TE）+ 若干 DP。通常一台机器一个节点，一台机器也可以起多个（验证脚本就是这么干的）。 |
| cluster | 参加同一次跨节点复用的所有节点。节点数 = `FLEXKV_RADIX_WORLD_SIZE`。 |
| cluster rank | etcd 在启动时给每个节点分配的编号 `0..world_size-1`。 |
| etcd | 小型分布式键值存储。这里做两件事：让节点在启动时互相认识（成员表 + 分配 rank），登记每个节点数据面的地址。 |
| prefetch | sglang 在请求入队时调 FlexKV 的 `prefetch_async`；radixshmem 模式下它就是"把对端持有的前缀拉回本节点"（`RadixClient.get_async`）。 |

---

## 1. 它是怎么工作的

```
请求到达 DP_k scheduler
 └ connector.prefetch_async(rid, tokens)
    └ RadixClient.get_async(hashes)：
         控制面  本地树走完 -> 查 cluster 的 router hash table -> RDMA 单边读 walk 对端的树
         数据面  为对端那段分配本地 SlotStore slot -> 本节点 radix-server 从对端 SlotStore RDMA READ
         发布    字节落地后 insert 进本地树；job 完成
 └ sglang 每轮 check_prefetch_progress -> 完成后 pop_prefetch_loaded_tokens（记为 storage hit）
 └ match_prefix -> lookup_kv -> get_match：只查本地树，对端块此时已是本地块 -> H2D
```

- **控制面**没有中心索引服务：查询进程用 RDMA 单边读直接翻对端的树，对端 CPU 不参与。
- **数据面**由 radix-server 执行：对端由 etcd `radix/<cluster_id>/data/<node>` 解析，字节直接落进本地
  SlotStore 的 slot，再由本进程的 completer 发布进树。
- `get_match` 永远只看本地。对端命中只对走过 prefetch 的请求生效；请求在 prefetch 完成前留在
  sglang 的 waiting queue（与 sglang 对 mooncake prefetch 的处理一样）。

只支持 CPU 层（`ssd_cache_gb` 必须为 0）。

---

## 2. 安装

### 2.0 系统依赖

```bash
# shmradix 编译要的；跨节点还要 libibverbs（RDMA）和 Go toolchain（它的 etcd client 是 Go 写的）
apt-get update && apt-get install -y \
    cmake build-essential libxxhash-dev liburing-dev libibverbs-dev golang-go
pip install pybind11 grpcio

# etcd 服务端（管成员表和数据面登记），全 cluster 一个
apt-get install -y etcd-server etcd-client
```

不再需要 Redis。

### 2.1 shmradix：必须带 RDMA + etcd + mooncake

源码：<https://gitlab-master.nvidia.com/zhuofanl/radixshmem>。radixshmem 自带 mooncake 传输后端
（`_data` 扩展），不需要 FlexKV 侧安装 mooncake。

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/zhuofanl/radixshmem.git
cd radixshmem
mkdir build && cd build && cmake .. && make -j          # libshmradix + _data
cd ../python && pip install -e . --no-build-isolation   # _core + shmradix 包
```

运行时 `LD_LIBRARY_PATH` 要包含 `radixshmem/build/lib` 和 `radixshmem/build/runtime-deps`。

### 2.2 FlexKV

```bash
cd FlexKV
pip install -e . --no-build-isolation
```

FlexKV 不需要 `FLEXKV_ENABLE_P2P=1`（那是老的 Redis/mooncake 路径的编译开关），也不需要 mooncake wheel。

---

## 3. 配置与启动

### 3.1 环境变量

全 cluster 一致：

```bash
export FLEXKV_RADIX_SHMEM=1
export FLEXKV_RADIX_WORLD_SIZE=2                      # 实际节点数，必须精确
export FLEXKV_RADIX_REGISTRY=etcd://10.0.0.1:2379     # 同一个 etcd
export FLEXKV_RADIX_CLUSTER_ID=flexkv                 # etcd 命名空间；同一 etcd 上跑多套集群必须各起一个名字
export FLEXKV_RADIX_RHT_SLOTS=4                       # RHT 每桶槽位数，1/2/4/8；1 会让跨节点复用失效，默认 4
```

每节点不同：

```bash
export FLEXKV_RADIX_RPC_ADDRESS=10.0.0.12    # 本机 IP，写进 etcd 给对端拨回
# 或 export FLEXKV_RADIX_RPC_INTERFACE=eth0  # 给网卡名让它自己取（两个都设时用 interface）
export FLEXKV_RADIX_INDEX_DEV=mlx5_0         # 索引控制面用的 HCA；空 = 第一个可用
export FLEXKV_RADIX_TRANSFER_DEV=mlx5_0      # 数据面（mooncake）用的 HCA，逗号分隔可多个；空 = 全部
export FLEXKV_DP_SIZE=8                      # 本节点的 --data-parallel-size
export FLEXKV_CONFIG_PATH=/etc/flexkv/node2.json
```

其余可选项：

```bash
export FLEXKV_RADIX_NODE_NAME=node2              # 节点在 etcd 里的身份，默认 node<bind-ip>；同机多节点必须各给一个
export FLEXKV_RADIX_GID_IDX=3                    # RoCE GID index
export FLEXKV_RADIX_REMOTE_OP_TRANSPORT=dc       # 远程 walk 的通道：dc（默认）或 zmq
export FLEXKV_RADIX_BOOTSTRAP_TIMEOUT_SEC=120    # 等全员到齐的上限
export FLEXKV_RADIX_SERVER_LAUNCH_MODE=embedded  # embedded：DP-0 拉起 radix-server 子进程；external：运维自己起 radix-server
export FLEXKV_RADIX_ENDPOINT=                    # radix-server 的 gRPC 端点；空 = unix:///dev/shm/shmradix_<id>_cpu.sock
export FLEXKV_RADIX_PREFAULT=1                   # 启动时预触 SlotStore 全部页面（可预测的 D2H 时延，启动慢一些）
export FLEXKV_RADIX_PREFETCH_TIMEOUT_MS=5000     # 一次对端拉取的 server 端期限，到期退化为本地命中
export FLEXKV_RADIX_PREFETCH_MAX_INFLIGHT=128    # 每个 DP 进程同时在飞的拉取数上限，超过则本次 prefetch 不走对端
```

**节点身份**必须每节点唯一：两个节点算出同一个身份就等于只有一个 peer 报到，bootstrap 会一直等到超时。
一台机器一个节点时各机 IP 天然不同；一台机器上跑多个节点时给每个节点不同的
`FLEXKV_RADIX_NODE_NAME`（以及不同的 `FLEXKV_SHM_RADIX_ID` 和 `FLEXKV_SERVER_RECV_PORT`）。
**别填 `0.0.0.0`** 当 `FLEXKV_RADIX_RPC_ADDRESS`，那样所有节点会算出同一个身份。

### 3.2 JSON 配置文件

```json
{
  "cpu_cache_gb": 200,
  "ssd_cache_gb": 0,
  "enable_p2p_cpu": true
}
```

- `ssd_cache_gb` 必须为 0：radixshmem 模式只有 CPU 层。
- `enable_p2p_cpu` 只是让 sglang connector 打开 prefetch；对端拉取本身由 `FLEXKV_RADIX_WORLD_SIZE > 1` 决定。
- 各节点的 `cpu_cache_gb` 可以不同，但 **block 的字节布局必须一样**：同一个模型、同样的
  `tokens_per_block`、同样的 KV dtype。radix-server 会在启动时校验各节点 SlotStore 的 slot 形状一致。
- 老配置里的 `redis_*`、`local_zmq_*`、`local_ip`、`mooncake_config_path` 在这条路上不再读取。

### 3.3 启动

**先起 etcd**（全 cluster 一个）：

```bash
etcd --name t --data-dir /tmp/flexkv_etcd.etcd \
     --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://10.0.0.1:2379 \
     --listen-peer-urls http://0.0.0.0:2380 \
     --initial-advertise-peer-urls http://10.0.0.1:2380 \
     --initial-cluster t=http://10.0.0.1:2380 &
```

**再起引擎**。每个节点起自己的 sglang（或 vllm）：DP-0 的 KVManager 会拉起本节点的 radix-server
子进程（embedded 模式），其余 DP、TE 和 transfer worker 按名字挂载它。

**所有节点必须一起拉起。** bootstrap 是 collective 的：每个节点的 radix-server 阻塞在里面等全员到齐
（上限 `FLEXKV_RADIX_BOOTSTRAP_TIMEOUT_SEC`），超时就整体失败。

### 3.4 内存

CPU KV cache 整体是 radix-server 建的 SlotStore，落在 `/dev/shm`（或 hugetlbfs）。容器的
`--shm-size` 必须 ≥ `cpu_cache_gb` 加索引，或者用 `use_hugepage_cpu_buffer: true` +
`FLEXKV_HUGETLBFS_DIR` 走 hugetlbfs（需预留 hugepages）。

---

## 4. 验证

### 4.1 单机预演

同一台机器上模拟两个节点（各占一张 GPU），走完整链路：node 0 PUT，node 1 `prefetch_async` 拉回，
再本地 `get_match` + H2D，逐字节比对。需要 ≥ 2 张 GPU、一个 ACTIVE 的 RDMA 口、PATH 上有 `etcd`
（或 `FLEXKV_TEST_RADIX_REGISTRY` 指向现成的 etcd）：

```bash
python tests/test_e2e_radix_prefetch_p2p.py
```

不需要 GPU 的两节点 RDMA 测试（radix-server + FlexKV engine，字节比对）：

```bash
FLEXKV_RUN_RADIX_PEER_TEST=1 python -m pytest tests/test_radix_shmem_engine.py -k over_rdma
```

### 4.2 真正的多台机器上

起服务时带上 `FLEXKV_TRACE_RADIX_PEER=1`（诊断用，线上关掉）。先只往节点 A 发一批 prompt，等它的异步
PUT 落地，再把同一批 prompt 发给节点 B，在**节点 B** 的日志里看：

```bash
grep "radix-server for" vllm.log            # cluster rank，各节点必须不同
grep "RADIX PEER PREFETCH" vllm.log | tail   # local_hit / planned_hit：planned > local 就是找到了对端
grep "act=peer_pull" vllm.log | tail         # pulled_blocks / bytes / source_rank：字节真的搬过来了
```

sglang 侧 `cached_tokens_storage`（`storage_hit_length`）记的就是从对端拉回的 token 数。

---

## 5. 排错速查

| 现象 | 原因 |
|---|---|
| 所有节点卡在启动，最后 `radix-server did not become ready` | bootstrap 是 collective 的：只起了一部分节点 / etcd 不通 / `WORLD_SIZE` 比实际节点数大 / 多个节点身份相同（§3.1） |
| `radix-server failed to start: No RDMA devices found` | 容器里没有 `/dev/infiniband`（起容器要 `--device /dev/infiniband --cap-add IPC_LOCK`），或 shmradix 编的时候没开 RDMA |
| `transfer backend unavailable` | shmradix 的 `_data` 没带 mooncake，或 `FLEXKV_RADIX_TRANSFER_DEV` 指的 HCA 不存在 |
| `attached radixshmem regions do not match FlexKV's configuration` | TE 算出的 CPU 块字节数或块数与 DP-0 启动 radix-server 时不同（多 PP 不均分、DSv4 sidecar 组没在 KVManager 之前记录），日志里有两边的数值 |
| bootstrap 成功但 `planned_hit` 恒等于 `local_hit` | `FLEXKV_RADIX_RHT_SLOTS` 为 1；或等得不够久（异步 PUT 还没落地）；或两节点不在同一 `FLEXKV_RADIX_CLUSTER_ID` |
| `planned_hit > local_hit` 但 `pulled_blocks=0` | 数据面没通：对端 `transfer_dev` / GID / `rpc_address` 不可达，或 `FLEXKV_RADIX_PREFETCH_TIMEOUT_MS` 太短 |
| 拉回的 KV 解码出乱码 | block 字节布局不一致（模型 / `tokens_per_block` / KV dtype 各节点必须一样） |
| `radix_shmem backs the CPU tier only` | 配了 `ssd_cache_gb > 0`；这条路上没有 SSD 层 |
