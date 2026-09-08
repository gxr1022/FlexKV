"""End-to-end DATA check for the radixshmem peer path: prefetch pulls a peer's
blocks, GET serves them from the local pool, the GPU holds the peer's bytes.

Two full FlexKV nodes on one host (two processes, two GPUs), one radixshmem
cluster:

  * each node's KVManager launches its own radix-server (index + SlotStore =
    the node's CPU pool + RDMA transfer engine) under a distinct
    FLEXKV_SHM_RADIX_ID; the two servers rendezvous in one etcd namespace
    (FLEXKV_RADIX_WORLD_SIZE=2), get dense cluster ranks and an RHT to route by;
  * node 0 PUTs a window of GPU blocks holding a per-block pattern;
  * node 1 calls `KVManager.prefetch_async` for the same tokens: the index walk
    finds the prefix on node 0 over RDMA, node 1's radix-server RDMA-reads the
    bytes into node 1's SlotStore and the blocks are published in node 1's tree;
    `try_wait` reports the pulled range;
  * node 1 then does the ordinary local `get_match` + `launch` + `wait` and the
    GPU blocks are compared byte for byte with what node 0 wrote.

A second window checks the extension case: node 1 already holds the first
LOCAL_HEAD_BLOCKS of it (its own bytes), the prefetch pulls only the tail, and
the GET serves head and tail from the right writers.

Requires: >=2 CUDA devices, an ACTIVE RDMA port, a shmradix built with RDMA +
etcd + mooncake, and an etcd (FLEXKV_TEST_RADIX_REGISTRY, or `etcd` on PATH for
a private one). Run inside the container:

    PYTHONPATH=/path/to/FlexKV:/path/to/radixshmem/python \\
    LD_LIBRARY_PATH=$RADIXSHMEM_LIBS:$TORCH_LIB:$LD_LIBRARY_PATH \\
    python3 tests/test_e2e_radix_prefetch_p2p.py
"""
from __future__ import annotations

import contextlib
import glob
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

WORLD_SIZE = 2
TOKENS_PER_BLOCK = 16
NUM_GPU_BLOCKS = 128
NUM_CPU_BLOCKS = 256
NUM_REQUEST_BLOCKS = 32
LOCAL_HEAD_BLOCKS = 10
FIRST_BLOCK = 8
SECOND_FIRST_BLOCK = 64
PATTERN_SEED = 0x5EED
SECOND_SEED = 0xBEEF


def node_tag_for(run_id: str, rank: int) -> str:
    return f"{run_id}_r{rank}"


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _active_rdma_devices() -> list:
    """RDMA devices with at least one ACTIVE port, honoring the env override."""
    override = os.getenv("FLEXKV_TEST_RDMA_DEVICES", "").strip()
    names = ([d for d in override.split(",") if d] if override
             else sorted(os.path.basename(p)
                         for p in glob.glob("/sys/class/infiniband/*")))
    active = []
    for name in names:
        for state in glob.glob(f"/sys/class/infiniband/{name}/ports/*/state"):
            with contextlib.suppress(OSError):
                with open(state) as handle:
                    if "ACTIVE" in handle.read():
                        active.append(name)
                        break
    return active


def _block_pattern(layer: int, block_id: int, shape, dtype,
                   writer: int = 0) -> torch.Tensor:
    """Deterministic content for one (layer, block) as written by node `writer`."""
    generator = torch.Generator().manual_seed(
        PATTERN_SEED + writer * 1_000_003 + block_id * 128 + layer)
    return torch.randn(tuple(shape), generator=generator).to(dtype)


def _write_pattern(gpu_tensors, block_ids, writer: int = 0) -> None:
    for layer, tensor in enumerate(gpu_tensors):
        for block_id in block_ids:
            block = tensor[:, block_id]
            block.copy_(_block_pattern(layer, int(block_id), block.shape,
                                       tensor.dtype, writer))
    torch.cuda.synchronize()


def _clear_blocks(gpu_tensors, block_ids) -> None:
    for tensor in gpu_tensors:
        for block_id in block_ids:
            tensor[:, block_id].zero_()
    torch.cuda.synchronize()


def _mismatched_blocks(gpu_tensors, block_ids, writer: int = 0) -> list:
    bad = []
    for layer, tensor in enumerate(gpu_tensors):
        for block_id in block_ids:
            got = tensor[:, block_id].cpu()
            want = _block_pattern(layer, int(block_id), got.shape, got.dtype, writer)
            if not torch.equal(got, want):
                bad.append((layer, int(block_id)))
    return bad


def _build_request(num_blocks: int, first_block: int, seed: int):
    rng = np.random.default_rng(seed)
    block_ids = np.arange(first_block, first_block + num_blocks, dtype=np.int64)
    slot_mapping = (np.repeat(block_ids, TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK
                    + np.tile(np.arange(TOKENS_PER_BLOCK), num_blocks))
    token_ids = rng.integers(0, 32000, size=slot_mapping.shape, dtype=np.int64)
    return token_ids, slot_mapping, block_ids


def _put_prefix(kvm, token_ids, slot_mapping, num_blocks: int) -> bool:
    from flexkv.common.request import KVResponseStatus
    num_tokens = num_blocks * TOKENS_PER_BLOCK
    task_id = kvm.put_async(token_ids=token_ids[:num_tokens],
                            slot_mapping=slot_mapping[:num_tokens])
    status = kvm.wait([task_id], timeout=120, completely=True)
    return all(r.status == KVResponseStatus.SUCCESS for r in status.values())


def _prefetch_until(kvm, token_ids, want_pulled_blocks: int, timeout: float = 60.0):
    """prefetch_async + try_wait until the pulled range reaches the expectation
    (the peer's RHT publication is asynchronous, early rounds may miss).
    Returns (pulled_blocks, rounds)."""
    from flexkv.common.request import KVResponseStatus
    deadline = time.monotonic() + timeout
    rounds = 0
    pulled = 0
    while time.monotonic() < deadline:
        rounds += 1
        task_id = kvm.prefetch_async(token_ids=token_ids)
        response = None
        while time.monotonic() < deadline:
            done = kvm.try_wait([task_id])
            if task_id in done and done[task_id].status != KVResponseStatus.TIMEOUT:
                response = done[task_id]
                break
            time.sleep(0.02)
        if response is None:
            break
        mask = response.return_mask
        pulled = int(np.count_nonzero(mask)) // TOKENS_PER_BLOCK if mask is not None else 0
        if pulled >= want_pulled_blocks:
            break
        time.sleep(0.5)
    return pulled, rounds


def _get_and_load(kvm, token_ids, slot_mapping) -> int:
    """Local get_match + launch + wait; returns the matched block count."""
    from flexkv.common.request import KVResponseStatus
    task_id, mask = kvm.get_match(token_ids=token_ids)
    hit_tokens = int(np.count_nonzero(mask))
    if hit_tokens == 0:
        return 0
    kvm.launch([task_id], [slot_mapping[:hit_tokens]])
    response = kvm.wait([task_id], timeout=120, completely=True)[task_id]
    if response.status != KVResponseStatus.SUCCESS:
        return 0
    return hit_tokens // TOKENS_PER_BLOCK


def _tp_client_proc(server_recv_port, model_config, cache_config,
                    num_gpu_blocks, child_conn):
    """Holds the node's GPU tensors and hands their IPC handles back."""
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    from flexkv.common.memory_handle import TensorSharedHandle
    from flexkv.server.client import KVTPClient

    tp_client = KVTPClient(server_recv_port, 0, 0)
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads // model_config.tp_size,
        head_size=model_config.head_size,
        kv_dim=model_config.kv_dim,
    )
    gpu_blocks = [
        torch.zeros(size=tuple(gpu_layout.kv_shape[1:]),
                    dtype=model_config.dtype).cuda(0)
        for _ in range(model_config.num_layers)
    ]
    tp_client.register_to_server(gpu_blocks, gpu_layout)
    child_conn.send([TensorSharedHandle(t) for t in gpu_blocks])
    child_conn.close()
    while True:
        time.sleep(1)


def _node_proc(rank, gpu_id, run_id, registry, cluster_id, rdma_dev,
               reader_ready, written, read_done, result_q):
    """One FlexKV node: rank 0 writes the windows, rank 1 prefetches and reads."""
    # Before any CUDA context exists: each node drives a different device while
    # addressing it as device 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    node_tag = node_tag_for(run_id, rank)
    recv_port = f"ipc:///tmp/flexkv_{node_tag}"
    os.environ.update({
        "FLEXKV_RADIX_SHMEM": "1",
        "FLEXKV_SHM_RADIX_ID": node_tag,
        "FLEXKV_RADIX_WORLD_SIZE": str(WORLD_SIZE),
        "FLEXKV_RADIX_REGISTRY": registry,
        # One etcd namespace per run keeps concurrent runs apart.
        "FLEXKV_RADIX_CLUSTER_ID": cluster_id,
        # etcd keys membership by node identity, which defaults to the bind IP
        # the co-located nodes share -- so name each node.
        "FLEXKV_RADIX_NODE_NAME": node_tag,
        "FLEXKV_RADIX_RPC_ADDRESS": "127.0.0.1",
        "FLEXKV_RADIX_INDEX_DEV": rdma_dev,
        "FLEXKV_RADIX_TRANSFER_DEV": rdma_dev,
        "FLEXKV_RADIX_RHT_SLOTS": "4",
        "FLEXKV_RADIX_PREFAULT": "0",
        "FLEXKV_ENABLE_MPS": "0",
        "FLEXKV_SERVER_RECV_PORT": recv_port,
        "FLEXKV_TRACE_RADIX_PEER": "1",
    })

    from flexkv.common.config import (CacheConfig, GLOBAL_CONFIG_FROM_ENV,
                                      ModelConfig)
    from flexkv.kvmanager import KVManager

    # Built from env at import time; set the fields that matter explicitly in
    # case a parent import happened earlier in this process.
    GLOBAL_CONFIG_FROM_ENV.radix_shmem = True
    GLOBAL_CONFIG_FROM_ENV.shm_radix_id = node_tag
    GLOBAL_CONFIG_FROM_ENV.radix_world_size = WORLD_SIZE
    GLOBAL_CONFIG_FROM_ENV.radix_registry = registry
    GLOBAL_CONFIG_FROM_ENV.radix_cluster_id = cluster_id
    GLOBAL_CONFIG_FROM_ENV.radix_node_name = node_tag
    GLOBAL_CONFIG_FROM_ENV.radix_rpc_address = "127.0.0.1"
    GLOBAL_CONFIG_FROM_ENV.radix_index_dev = rdma_dev
    GLOBAL_CONFIG_FROM_ENV.radix_transfer_devices = [rdma_dev]
    GLOBAL_CONFIG_FROM_ENV.radix_rht_slots = 4
    GLOBAL_CONFIG_FROM_ENV.radix_prefault = False
    GLOBAL_CONFIG_FROM_ENV.enable_mps = False
    GLOBAL_CONFIG_FROM_ENV.server_recv_port = recv_port

    tag = f"[node r{rank}]"
    model_config = ModelConfig(
        num_layers=2, num_kv_heads=4, head_size=128,
        dtype=torch.float16, tp_size=1, dp_size=1,
    )
    cache_config = CacheConfig(
        tokens_per_block=TOKENS_PER_BLOCK,
        enable_cpu=True, enable_ssd=False, enable_remote=False,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        enable_p2p_cpu=True,
    )

    report = {"rank": rank}
    kvm = None
    tp_proc = None
    try:
        kvm = KVManager(model_config, cache_config, dp_client_id=0)
        kvm.start()

        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        tp_proc = ctx.Process(
            target=_tp_client_proc,
            args=(kvm.gpu_register_port, model_config, cache_config,
                  NUM_GPU_BLOCKS, child_conn),
            daemon=True,
        )
        tp_proc.start()
        handles = parent_conn.recv()
        gpu_tensors = [handle.get_tensor() for handle in handles]

        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and not kvm.is_ready():
            time.sleep(0.1)
        if not kvm.is_ready():
            raise RuntimeError("KVManager not ready in 180s")
        report["node_id"] = int(cache_config.distributed_node_id)
        print(f"{tag} READY as cluster rank {report['node_id']}", flush=True)

        tokens_a, slots_a, blocks_a = _build_request(NUM_REQUEST_BLOCKS, FIRST_BLOCK,
                                                     PATTERN_SEED)
        tokens_b, slots_b, blocks_b = _build_request(NUM_REQUEST_BLOCKS,
                                                     SECOND_FIRST_BLOCK, SECOND_SEED)

        if rank == 1:
            # Window B: this node owns the head before node 0 publishes anything.
            _write_pattern(gpu_tensors, blocks_b[:LOCAL_HEAD_BLOCKS], writer=1)
            report["head_put_ok"] = _put_prefix(kvm, tokens_b, slots_b, LOCAL_HEAD_BLOCKS)
            print(f"{tag} head put ok={report['head_put_ok']}", flush=True)
            reader_ready.set()
            if not written.wait(240):
                raise TimeoutError("writer did not publish in 240s")

            # 1) Window A, nothing local: the prefetch pulls all of it off node 0,
            #    then the GET serves it from this node's pool.
            pulled, rounds = _prefetch_until(kvm, tokens_a, NUM_REQUEST_BLOCKS)
            report["a_pulled_blocks"], report["a_prefetch_rounds"] = pulled, rounds
            _clear_blocks(gpu_tensors, blocks_a)
            report["a_hit_blocks"] = _get_and_load(kvm, tokens_a, slots_a)
            report["a_mismatched"] = _mismatched_blocks(gpu_tensors, blocks_a, writer=0)
            print(f"{tag} window A: pulled={pulled} rounds={rounds} "
                  f"hit={report['a_hit_blocks']} mismatched={len(report['a_mismatched'])}",
                  flush=True)

            # 2) Window B, local head: the prefetch pulls only the tail.
            pulled, rounds = _prefetch_until(kvm, tokens_b,
                                             NUM_REQUEST_BLOCKS - LOCAL_HEAD_BLOCKS)
            report["b_pulled_blocks"], report["b_prefetch_rounds"] = pulled, rounds
            _clear_blocks(gpu_tensors, blocks_b)
            report["b_hit_blocks"] = _get_and_load(kvm, tokens_b, slots_b)
            report["b_head_mismatched"] = _mismatched_blocks(
                gpu_tensors, blocks_b[:LOCAL_HEAD_BLOCKS], writer=1)
            report["b_tail_mismatched"] = _mismatched_blocks(
                gpu_tensors, blocks_b[LOCAL_HEAD_BLOCKS:], writer=0)
            print(f"{tag} window B: pulled={pulled} rounds={rounds} "
                  f"hit={report['b_hit_blocks']} "
                  f"head_mismatched={len(report['b_head_mismatched'])} "
                  f"tail_mismatched={len(report['b_tail_mismatched'])}", flush=True)
            read_done.set()
        else:
            if not reader_ready.wait(240):
                raise TimeoutError("reader did not lay down its head in 240s")
            _write_pattern(gpu_tensors, blocks_a, writer=0)
            ok = _put_prefix(kvm, tokens_a, slots_a, NUM_REQUEST_BLOCKS)
            _write_pattern(gpu_tensors, blocks_b, writer=0)
            report["put_ok"] = _put_prefix(kvm, tokens_b, slots_b, NUM_REQUEST_BLOCKS) and ok
            print(f"{tag} put ok={report['put_ok']}", flush=True)
            written.set()
            # Stay up: the reader's radix-server reads THIS node's SlotStore.
            if not read_done.wait(300):
                raise TimeoutError("reader did not finish in 300s")
    except Exception:
        import traceback
        report["error"] = traceback.format_exc()
        print(f"{tag} FAILED\n{report['error']}", flush=True)
        reader_ready.set()
        written.set()
        read_done.set()
    finally:
        if tp_proc is not None:
            tp_proc.terminate()
            tp_proc.join(timeout=10)
        if kvm is not None:
            with contextlib.suppress(Exception):
                kvm.shutdown()
        result_q.put(report)


def _start_private_etcd():
    etcd = shutil.which("etcd")
    if not etcd:
        return None, None, None
    client_port, peer_port = _free_port(), _free_port()
    workdir = tempfile.mkdtemp(prefix="flexkv_p2p_etcd_")
    proc = subprocess.Popen(
        [etcd, "--name", "t", "--data-dir", os.path.join(workdir, "data"),
         "--listen-client-urls", f"http://127.0.0.1:{client_port}",
         "--advertise-client-urls", f"http://127.0.0.1:{client_port}",
         "--listen-peer-urls", f"http://127.0.0.1:{peer_port}",
         "--initial-advertise-peer-urls", f"http://127.0.0.1:{peer_port}",
         "--initial-cluster", f"t=http://127.0.0.1:{peer_port}"],
        stdout=open(os.path.join(workdir, "etcd.log"), "w"), stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with contextlib.suppress(OSError):
            with socket.create_connection(("127.0.0.1", client_port), timeout=0.5):
                return proc, workdir, f"etcd://127.0.0.1:{client_port}"
        time.sleep(0.2)
    proc.kill()
    shutil.rmtree(workdir, ignore_errors=True)
    return None, None, None


def main() -> int:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        print(f"SKIP: need >={WORLD_SIZE} CUDA devices")
        return 0
    devices = _active_rdma_devices()
    if not devices:
        print("SKIP: no ACTIVE RDMA port (checked /sys/class/infiniband/*)")
        return 0
    registry = os.getenv("FLEXKV_TEST_RADIX_REGISTRY", "")
    etcd_proc, etcd_dir = None, None
    if not registry:
        etcd_proc, etcd_dir, registry = _start_private_etcd()
        if not registry:
            print("SKIP: set FLEXKV_TEST_RADIX_REGISTRY or put etcd on PATH")
            return 0

    rdma_dev = devices[0]
    run_id = f"p2p{os.getpid()}"
    cluster_id = f"flexkv_p2p_{os.getpid()}"
    print(f"etcd {registry}, rdma {rdma_dev}, run {run_id}", flush=True)

    ctx = mp.get_context("spawn")
    reader_ready, written, read_done = ctx.Event(), ctx.Event(), ctx.Event()
    result_q = ctx.Queue()
    procs = []
    reports = {}
    try:
        for rank in range(WORLD_SIZE):
            proc = ctx.Process(
                target=_node_proc,
                args=(rank, rank, run_id, registry, cluster_id, rdma_dev,
                      reader_ready, written, read_done, result_q),
                daemon=False,
            )
            proc.start()
            procs.append(proc)

        deadline = time.monotonic() + 600
        while len(reports) < WORLD_SIZE and time.monotonic() < deadline:
            try:
                report = result_q.get(timeout=5)
                reports[report["rank"]] = report
            except Exception:
                if not any(proc.is_alive() for proc in procs):
                    break
    finally:
        for proc in procs:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
        if etcd_proc is not None:
            etcd_proc.terminate()
            with contextlib.suppress(Exception):
                etcd_proc.wait(10)
            shutil.rmtree(etcd_dir, ignore_errors=True)
        for rank in range(WORLD_SIZE):
            base = f"shmradix_{node_tag_for(run_id, rank)}_cpu"
            for root in ("/dev/shm", "/dev/hugepages"):
                for stale in glob.glob(f"{root}/{base}*"):
                    with contextlib.suppress(OSError):
                        os.unlink(stale)

    print("\n=== RESULTS ===")
    for rank in sorted(reports):
        printable = {k: v for k, v in reports[rank].items() if k != "error"}
        print(rank, printable)
        if "error" in reports[rank]:
            print(reports[rank]["error"])

    if len(reports) < WORLD_SIZE:
        print(f"FAIL: only {len(reports)}/{WORLD_SIZE} nodes reported")
        return 1
    if any("error" in report for report in reports.values()):
        print("FAIL: a node raised")
        return 1
    writer, reader = reports[0], reports[1]
    if writer.get("node_id") == reader.get("node_id"):
        print(f"FAIL: both nodes got cluster rank {writer.get('node_id')}")
        return 1
    if not writer.get("put_ok"):
        print("FAIL: writer's puts did not complete")
        return 1
    if not reader.get("head_put_ok"):
        print("FAIL: reader's head put did not complete")
        return 1
    tail = NUM_REQUEST_BLOCKS - LOCAL_HEAD_BLOCKS
    checks = [
        (reader.get("a_pulled_blocks", 0) >= NUM_REQUEST_BLOCKS,
         f"window A: prefetch pulled {reader.get('a_pulled_blocks')}/{NUM_REQUEST_BLOCKS}"),
        (reader.get("a_hit_blocks", 0) == NUM_REQUEST_BLOCKS,
         f"window A: local GET matched {reader.get('a_hit_blocks')}/{NUM_REQUEST_BLOCKS}"),
        (not reader.get("a_mismatched"),
         f"window A: wrong bytes in {reader.get('a_mismatched')}"),
        (reader.get("b_pulled_blocks", 0) >= tail,
         f"window B: prefetch pulled {reader.get('b_pulled_blocks')}/{tail} tail blocks"),
        (reader.get("b_hit_blocks", 0) == NUM_REQUEST_BLOCKS,
         f"window B: local GET matched {reader.get('b_hit_blocks')}/{NUM_REQUEST_BLOCKS}"),
        (not reader.get("b_head_mismatched"),
         f"window B: head does not hold node 1's bytes: {reader.get('b_head_mismatched')}"),
        (not reader.get("b_tail_mismatched"),
         f"window B: tail does not hold node 0's bytes: {reader.get('b_tail_mismatched')}"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        for msg in failed:
            print(f"FAIL: {msg}")
        return 1
    print(f"PASS: node 1 prefetched {NUM_REQUEST_BLOCKS} blocks off node 0 and served "
          f"them locally byte for byte; with a {LOCAL_HEAD_BLOCKS}-block local head "
          f"it pulled only the {tail}-block tail and the GET served head (node 1) "
          f"and tail (node 0) from the right writers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
