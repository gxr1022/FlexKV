# SPDX-License-Identifier: Apache-2.0
"""
RadixShmem-backed CacheEngine for the CPU tier.

A drop-in for `flexkv.cache.cache_engine.CacheEngineAccel` whose RadixTree, slot
mempool AND the CPU KV memory itself belong to radixshmem: a `radix-server`
process per node (see `flexkv.server.shm_radix_bootstrap`) owns the index shm,
the SlotStore (one slot per block) and the RDMA transfer engine. Every DP
scheduler process attaches with `shmradix.RadixClient(name)` and runs prefix
queries / inserts in parallel, serialised only by a process-shared rwlock.

Differences from `CacheEngineAccel`:

- The slot mempool is radixshmem's, so there is no `flexkv.cache.mempool.Mempool`
  here: `take()`/`recycle()` forward to `allocate_slots()`/`recycle_slots()`. Slot
  ids come back int32, cast to int64 at the boundary, and a slot id IS the block
  index into the SlotStore pool the TE maps as its CPU buffer. Eviction is
  implicit inside `allocate_slots`.
- No node_id (nodes split on later inserts), so everything is addressed by hash
  path + (start, length): `insert()` hands nothing back and `take()` accepts no
  `protected_node`. The one refcount FlexKV holds is the READ side's:
  `query(lock=True)`, released via `QueryResult.finalize`.
- `match()` returns a `ShmRadixMatch` and is LOCAL ONLY. Peer blocks are not
  spliced into a GET: `prefetch()` (`RadixClient.get_async`) asks the whole
  cluster, has the server pull the peer's run into local staging slots over RDMA
  and publishes them here, after which an ordinary local `match()` finds them.

Insert happens AFTER transfer. There is no "ready" bit; a block is published by
being in the tree at all, so the order is `take() -> transfer -> insert()`.
Consequences: `insert()` must be called from the completion path, not while
building the graph; a graph that never completes leaves its slots attached to
nothing -- neither reachable nor evictable, so they leak; and
`insert(auto_recycle=True)` takes slot ownership, so the caller must not recycle
the same slots again.

Components: a region can also carry an SWA component (`SWAPoolConfig`). All
queries use radixshmem's component overload: Full-only passes
`COMPONENT_MASK_FULL`; `FULL|SWA` returns the joint hit plus the W-block window
ending there. SWA slots live in their own all-or-none mempool, addressed by the
`component=` parameter, and publish only after the Full path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import DeviceType

if TYPE_CHECKING:
    # These pull in the FlexKV C++ extension transitively; keep them out of
    # import-time so this module loads without CUDA/libtorch.
    from flexkv.common.block import SequenceMeta
    from flexkv.common.config import SWAPoolConfig
    from flexkv.integration.dynamo.collector import KVEventCollector

try:
    import shmradix
except ImportError as e:  # pragma: no cover
    shmradix = None
    _SHMRADIX_IMPORT_ERROR = e
else:
    _SHMRADIX_IMPORT_ERROR = None

if shmradix is not None:
    COMPONENT_MASK_FULL = int(shmradix.COMPONENT_MASK_FULL)
    COMPONENT_MASK_SWA = int(shmradix.COMPONENT_MASK_SWA)
    COMPONENT_FULL = shmradix.ComponentType.FULL
    COMPONENT_SWA = shmradix.ComponentType.SWA
else:  # keep the module importable for type checks / doc builds
    COMPONENT_MASK_FULL = 1
    COMPONENT_MASK_SWA = 2
    COMPONENT_FULL = None
    COMPONENT_SWA = None


_DEVICE_TYPE_NAMES = ['CPU', 'GPU', 'SSD', 'REMOTE']


def _empty_i64() -> np.ndarray:
    return np.empty(0, dtype=np.int64)


@dataclass
class ShmRadixMatch:
    """One local radixshmem prefix query.

        block index   0                               num_matched_blocks
                      |  local_slots (this node's SlotStore slot ids)  |

    The query ran with `lock=True`, so the matched prefix is pinned until
    `release()`, which must run on every path or it stays pinned for the
    region's life.

    A `FULL|SWA` match also carries the window: `swa_slots` are local SWA-pool
    slot ids covering `[swa_start, num_matched_blocks)`, where
    `num_matched_blocks` is the joint `common_hit` -- possibly shorter than a
    Full-only hit. Both stay empty under a Full-only mask.
    """
    num_matched_blocks: int = 0
    local_slots: np.ndarray = field(default_factory=_empty_i64)
    swa_start: int = 0
    swa_slots: np.ndarray = field(default_factory=_empty_i64)
    finalize: Optional[Callable[[], None]] = None

    @property
    def num_local_blocks(self) -> int:
        return self.num_matched_blocks

    def local_range(self, first: int, last: int) -> np.ndarray:
        """The slots of absolute block range [first, last), clipped to the hit.

        The slice does all the bounding both ways, so callers need no boundary
        arithmetic; overrunning the hit is not an error, those blocks simply are
        not held. Both bounds must be non-negative.
        """
        return self.local_slots[first:last]

    def release(self) -> None:
        """Drop the query's pin. Idempotent."""
        finalize, self.finalize = self.finalize, None
        if finalize is not None:
            finalize()


class StagedRadixInsert:
    """Attach staged slots to a radixshmem tree once their data has landed.

    radixshmem admits a block only when it already holds data, so the insert has
    to run from a completion callback. Until then the slots are owned by nobody
    but this object, and ``publish`` must run or they are lost for the life of
    the region: it hands them to the tree, which takes ownership and recycles
    whatever did not attach.

    ``publish`` takes no ref on what it attached (the transfer is over, so the
    span has no reader), but it IS only legal while the ref keeping the local
    tree reaching the span's start is held -- pass that as ``holds``, which it
    drops afterwards. ``abort`` is the other exit: the plan was cancelled before
    its graph ran, so the slots go back to the mempool and the holds drop.

    ``component`` is one ComponentType (default FULL), not a query mask. A
    Full+SWA PUT arms two instances, FULL first: SWA's insert refuses paths
    the Full tree does not reach yet.
    """

    def __init__(self,
                 engine: "CacheEngineRadixShmem",
                 sequence_meta: "SequenceMeta",
                 slots: np.ndarray,
                 path_end: int,
                 label: str,
                 holds: Sequence[Callable[[], None]] = (),
                 component: "shmradix.ComponentType" = COMPONENT_FULL) -> None:
        self._engine = engine
        self._sequence_meta = sequence_meta
        self._slots = slots
        self._path_end = path_end
        self._label = label
        self._holds = list(holds)
        self._component = component
        self._settled = False

    def publish(self) -> None:
        if self._settled:
            return
        self._settled = True
        try:
            # The staged span always ENDS at path_end, so insert() derives
            # start = path_end - len(slots) on its own.
            self._engine.insert(
                self._sequence_meta,
                self._slots,
                num_insert_blocks=self._path_end,
                component=self._component,
            )
        except Exception as e:
            flexkv_logger.error(
                f"radixshmem {self._label}: insert of {len(self._slots)} "
                f"staged slots failed: {e}; returning them to the mempool"
            )
            self._engine.recycle(self._slots, component=self._component)
        finally:
            self._release_holds()

    def abort(self) -> None:
        """The graph never ran: hand the staged slots back and drop the holds.

        Exclusive with ``publish``; whichever runs first settles the insert.
        """
        if self._settled:
            return
        self._settled = True
        try:
            self._engine.recycle(self._slots, component=self._component)
        except Exception as e:
            flexkv_logger.error(
                f"radixshmem {self._label}: recycle of {len(self._slots)} "
                f"staged slots failed: {e}"
            )
        finally:
            self._release_holds()

    def _release_holds(self) -> None:
        # In a finally, and one try each: a ref left taken pins its prefix for
        # the life of the shm region, and no attached process can undo that.
        holds, self._holds = self._holds, []
        for release in holds:
            try:
                release()
            except Exception as e:  # keep dropping the rest
                flexkv_logger.error(
                    f"radixshmem {self._label}: ref release failed: {e}")


def _ensure_shmradix():
    if shmradix is None:
        raise ImportError(
            "shmradix is not installed; install it from radixshmem repo "
            "(pip install -e radixshmem/python). Original error: "
            f"{_SHMRADIX_IMPORT_ERROR}"
        )


class CacheEngineRadixShmem:
    """Radixshmem-backed cache engine for the CPU tier.

    Multiple instances (one per DP scheduler process) attach to the same
    radix-server by name and concurrently query / insert / prefetch.
    """

    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,  # -1 => recover from region via block_size()
                 shm_name: str,
                 evict_ratio: float = 0.05,
                 evict_start_threshold: float = 1.0,
                 hit_reward_seconds: int = 0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector=None,
                 protected_threshold: int = 2,
                 peer_enabled: bool = False,
                 swa_config: Optional["SWAPoolConfig"] = None):
        """Attach to the radix-server named ``shm_name`` (the base index name,
        `shm_radix_bootstrap.radix_index_name`); the server must be running or
        starting (see `shm_radix_bootstrap.RadixServerProcess`).

        `peer_enabled` turns on cross-node reuse: `prefetch()` walks the cluster
        and pulls a peer's run into this node. It only takes effect on a
        clustered region (world_size > 1).
        """
        _ensure_shmradix()
        if device_type != DeviceType.CPU:
            raise NotImplementedError(
                f"radixshmem backs the CPU tier only, not {device_type}")

        if eviction_policy != "lru":
            flexkv_logger.warning(
                f"radixshmem only supports LRU eviction; ignoring "
                f"eviction_policy={eviction_policy!r}"
            )
        # Not expressible in radixshmem yet; accepted for ABI compatibility with
        # CacheEngineAccel.
        if hit_reward_seconds != 0 or protected_threshold != 2:
            flexkv_logger.debug(
                "radixshmem ignores hit_reward_seconds and protected_threshold"
            )

        self.device_type = device_type
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self._swa_config = (swa_config.for_cache_tier(device_type)
                            if swa_config is not None else None)

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        self._trace_peer = os.getenv("FLEXKV_TRACE_RADIX_PEER", "0") == "1"

        from flexkv.server.shm_radix_bootstrap import (attach_radix_client,
                                                       radix_cluster_rank)
        self._client = attach_radix_client(shm_name)
        # Index methods pass through the RadixClient to the underlying index.
        self._tree = self._client
        # A distributed region extends the name FlexKV asked for with shmradix's
        # node identity; log the resolved one, not the prefix.
        self.shm_name = self._client.info.index_name
        self.cluster_rank = radix_cluster_rank(self._client)
        self.is_distributed = bool(self._client.is_distributed())
        self.peer_enabled = bool(peer_enabled) and self.is_distributed
        if peer_enabled and not self.is_distributed:
            flexkv_logger.warning(
                f"radixshmem peer reuse is enabled for {self.shm_name} but the "
                f"attached region has world_size=1; prefetch stays local-only"
            )

        # -1 => recover tokens_per_block from the region itself
        # (RadixClient.block_size(), written by the owner on create).
        if tokens_per_block is None or tokens_per_block < 0:
            tokens_per_block = int(self._client.block_size())
        elif int(self._client.block_size()) != int(tokens_per_block):
            raise ValueError(
                f"radix-server {self.shm_name} has tokens_per_block="
                f"{self._client.block_size()}, FlexKV is configured with "
                f"{tokens_per_block}")
        self.tokens_per_block = tokens_per_block
        capacity = int(self._client.mempool_total())
        if num_total_blocks > 0 and capacity != int(num_total_blocks):
            flexkv_logger.warning(
                f"radix-server {self.shm_name} has {capacity} FULL slots, FlexKV "
                f"expected {num_total_blocks}; the index is authoritative"
            )
        self.num_total_blocks = capacity

    # ---------- Attachment ----------

    @property
    def client(self):
        """The `shmradix.RadixClient` (index ops, `store`, `get_async`)."""
        return self._client

    @property
    def store(self):
        """The SlotStore mapping: this node's CPU KV pool."""
        return self._client.store

    # ---------- Mempool view (compatibility shims for CacheEngineAccel API) ----------

    @property
    def mempool(self) -> _MempoolView:
        return _MempoolView(self._tree)

    @property
    def swa_enabled(self) -> bool:
        return self._swa_config is not None and self._swa_config.num_slots > 0

    # ---------- Lifecycle ----------

    def reset(self) -> None:
        """Clear this node's cached data in place.

        Invalidates every slot id and match result taken before the call, so it is
        only safe with no transfer in flight.
        """
        self._tree.reset()

    def close(self) -> None:
        client, self._client, self._tree = self._client, None, None
        if client is not None:
            client.close()

    def start(self) -> None:
        """No-op; the peer-capable cache engine lifecycle calls this."""

    # ---------- Hashes ----------

    @staticmethod
    def _hashes(sequence_meta: "SequenceMeta", query_end: Optional[int]) -> np.ndarray:
        sequence_meta.gen_hashes()
        # SequenceMeta.block_hashes is int64; radixshmem expects uint64. They
        # share the same byte width, so view-cast is safe.
        hashes = sequence_meta.block_hashes.view(np.uint64)
        if query_end is not None:
            hashes = hashes[:query_end]
        return hashes

    # ---------- Match (local) ----------

    def match(self,
              sequence_meta: SequenceMeta,
              *,
              component_mask: int = COMPONENT_MASK_FULL,
              query_end: Optional[int] = None,
              gpu_matched_blocks: int = 0) -> ShmRadixMatch:
        """Prefix-match against this node's tree, pinning the hit.

        `component_mask` defaults to Full-only, whose `common_hit` is the plain
        prefix hit; `FULL|SWA` adds the joint hit and the window. `query_end`
        caps the queried path, so the window ends where the caller's restore
        will. `gpu_matched_blocks` is accepted for parity with the accel/hie
        engines.

        Peer blocks never appear here: `prefetch()` brings them into the local
        tree first.
        """
        hashes = self._hashes(sequence_meta, query_end)

        # lock=True inc_ref's every component pin; the returned finalize, owned
        # by the cache_engine layer, is the only thing that releases them.
        qr = self._tree.query(hashes, mask=component_mask,
                              local_only=True, lock=True)
        # A refused query (status != OK) has zeroed fields and an unarmed
        # finalize, so it falls through as an empty match -- like legacy query.
        common_hit = int(qr.common_hit)

        local_slots = _empty_i64()
        for _source_rank, _offset, slot_ids in qr.full_fragments:
            local_slots = np.asarray(slot_ids, dtype=np.int64)
        if len(local_slots) != common_hit:
            self._finalize_and_raise(
                qr,
                f"radixshmem local query covers {len(local_slots)} blocks "
                f"of a {common_hit}-block hit"
            )

        swa_slots = np.asarray(qr.swa_slots, dtype=np.int64)
        swa_start = int(qr.swa_start) if len(swa_slots) > 0 else 0
        if len(swa_slots) > 0 and swa_start + len(swa_slots) != common_hit:
            self._finalize_and_raise(
                qr,
                f"radixshmem returned {len(swa_slots)} SWA slots at "
                f"swa_start={swa_start} for a {common_hit}-block joint hit"
            )

        return ShmRadixMatch(
            num_matched_blocks=common_hit,
            local_slots=local_slots,
            swa_start=swa_start,
            swa_slots=swa_slots,
            finalize=qr.finalize,
        )

    @staticmethod
    def _finalize_and_raise(qr, message: str) -> None:
        """Drop the query's refs before propagating a consistency failure —
        otherwise the locked prefix stays pinned for the region's lifetime."""
        if qr.finalize is not None:
            qr.finalize()
        raise RuntimeError(message)

    # ---------- Prefetch (peer pull) ----------

    def prefetch(self,
                 sequence_meta: SequenceMeta,
                 *,
                 component_mask: int = COMPONENT_MASK_FULL,
                 query_end: Optional[int] = None,
                 timeout_ms: int = 30000) -> Any:
        """Start `RadixClient.get_async` for the prefix and return its `GetJob`.

        The query walks the local tree and continues onto one peer's; the
        server RDMA-reads the peer's run into local staging slots and this
        client's completer publishes them into the local tree when the job
        completes. `job.local_hit` / `job.planned_hit` are known on return.
        `lock=False`: the transfer window is pinned by the query inside
        radixshmem, nothing stays pinned afterwards (the caller's later
        `match()` takes its own pin). `block=False`: a saturated client
        completes the job at once with the local hit instead of blocking.

        Returns None on a non-distributed region: there is no peer to pull
        from, and the caller's local match already says what is here.
        """
        if not self.peer_enabled:
            return None
        hashes = self._hashes(sequence_meta, query_end)
        job = self._client.get_async(hashes, component_mask, lock=False,
                                     timeout_ms=int(timeout_ms), block=False)
        if self._trace_peer:
            flexkv_logger.info(
                f"[RADIX PEER PREFETCH] shm={self.shm_name} mask={component_mask:#x} "
                f"blocks={len(hashes)} local_hit={job.local_hit} "
                f"planned_hit={job.planned_hit} job={job.job_id}"
            )
        return job

    # ---------- Publish (insert) ----------

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int,
               component: "shmradix.ComponentType" = COMPONENT_FULL) -> None:
        """Attach already-written slots to the tree. Call ONLY after the transfer
        into `physical_block_ids` has landed -- an entry in the tree is by
        definition complete and servable.

        `physical_block_ids[i]` holds logical block `start + i`, with
        `start = num_insert_blocks - len(physical_block_ids)`. `num_insert_blocks`
        is required and has no default: the two lengths visible here give the
        span's extent, never its POSITION, and assuming it reaches the path end
        fails silently for any caller whose window stops short. radixshmem owns
        the disagreement between `start` and its own matched prefix, so nothing is
        re-derived from a (by now stale) match result.

        Slot ownership transfers (`auto_recycle=True`); the caller must not
        recycle them again. No ref is taken on what landed -- the transfer is over,
        so the span has no reader -- and nothing is returned, since what did not
        attach was already recycled internally.
        """
        sequence_meta.gen_hashes()
        hashes = sequence_meta.block_hashes.view(np.uint64)

        slots = np.ascontiguousarray(physical_block_ids, dtype=np.int32)
        num_slots = len(slots)
        if num_slots == 0:
            return

        # Full logical path this insert reaches the end of. Overshooting the
        # sequence is clamped; undershooting the staged span trips start < 0.
        path_end = min(int(num_insert_blocks), len(hashes))
        start = path_end - num_slots
        if start < 0:
            # A caller bug, not a race: radixshmem absorbs a matched prefix that
            # moved under us. Raising keeps slot ownership with the caller.
            raise ValueError(
                f"radixshmem insert of {num_slots} slots overruns the "
                f"{path_end}-block path on {self.shm_name}"
            )

        target_hashes = hashes[:path_end]
        # `start` positions FULL slots only. SWA/MAMBA inserts are right-aligned
        # by radixshmem itself (slots cover [max(0, n-W), n)) and refuse a
        # non-zero start with BAD_REQUEST.
        tree_start = start if component == COMPONENT_FULL else 0
        result = self._tree.insert(target_hashes, slots, start=tree_start,
                                   auto_recycle=True, component=component)

        unused = len(result.unused_slots)
        landed = num_slots - unused
        if result.error == shmradix.InsertError.FULL_PATH_MISSING:
            flexkv_logger.warning(
                f"radixshmem {component} insert on {self.shm_name}: full path "
                f"[0, {path_end}) was evicted before the window published "
                f"(slots were auto-recycled)"
            )
        elif result.error != shmradix.InsertError.OK:
            flexkv_logger.warning(
                f"radixshmem {component} insert on {self.shm_name} returned "
                f"{result.error}: {landed}/{num_slots} blocks landed at "
                f"start={start} (unused slots were auto-recycled)"
            )

        if landed <= 0:
            # Nothing new attached (concurrent writer, or pool exhausted);
            # auto_recycle already returned the slots.
            return

        if self.peer_enabled and component == COMPONENT_FULL:
            # Until the RHT publication drains, this node's new blocks are
            # invisible cluster-wide. SWA windows are local-only by design, so
            # they have nothing to publish.
            self._tree.flush()

        if (self.event_collector is not None and component == COMPONENT_FULL
                and result.error == shmradix.InsertError.OK):
            # `landed` is a COUNT, not a range -- unused_slots merges leading
            # redundancy with an unlanded tail. Only on the error-free path can it
            # name blocks: we always supply exactly `path_end - start` slots, so
            # insert cannot run short, leaving redundancy as the only cause and
            # putting what landed at the END of the path.
            attached_hashes = sequence_meta.block_hashes[path_end - landed : path_end]
            self.event_collector.publish_stored(
                block_hashes=attached_hashes,
                block_size=self.tokens_per_block,
                medium=_DEVICE_TYPE_NAMES[self.device_type]
            )

    # ---------- Mempool ops (take/recycle) ----------

    def take(self,
             num_required_blocks: int,
             strict: bool = True,
             component: "shmradix.ComponentType" = COMPONENT_FULL) -> np.ndarray:
        """Allocate slots from one component's pool, auto-evicting LRU as needed.

        No `protected_node` to accept: the prefix a caller would want pinned is
        already held by its match's query ref, which auto-evict honours. The
        returned slots are un-evictable AND un-reclaimable until `insert()` or
        `recycle()`.
        """
        slots_i32 = self._tree.allocate_slots(num_required_blocks,
                                              component=component)

        slots = np.asarray(slots_i32, dtype=np.int64)

        if strict and len(slots) < num_required_blocks:
            # Caller will recycle whatever we returned; mirror CacheEngineAccel
            # by raising on shortfall.
            self._tree.recycle_slots(np.asarray(slots, dtype=np.int32),
                                     component=component)
            raise RuntimeError(
                f"radixshmem: not enough free {component} blocks to take, "
                f"required: {num_required_blocks}, available: {len(slots)}"
            )

        if (self._metrics_collector is not None and len(slots) > 0
                and component == COMPONENT_FULL):
            # SWA slots live in their own pool; counting them as tier block
            # allocations would skew the Full mempool metrics.
            self._metrics_collector.record_allocation(
                _DEVICE_TYPE_NAMES[self.device_type].lower(), len(slots)
            )
        return slots

    def recycle(self,
                physical_blocks: np.ndarray,
                component: "shmradix.ComponentType" = COMPONENT_FULL) -> None:
        """Give slots back to their component's mempool.

        Also the only way back for slots whose transfer never completed: outside
        the tree they are unreachable by any query and invisible to eviction.
        """
        if physical_blocks is None or len(physical_blocks) == 0:
            return
        slots_i32 = np.ascontiguousarray(physical_blocks, dtype=np.int32)
        self._tree.recycle_slots(slots_i32, component=component)

    # ---------- Stats passthrough ----------

    @property
    def num_free_blocks(self) -> int:
        return int(self._tree.mempool_free())

    @property
    def num_used_blocks(self) -> int:
        return int(self._tree.mempool_used())

    @property
    def total_nodes(self) -> int:
        return int(self._tree.total_radix_nodes())


@dataclass
class _MempoolView:
    """Read-only mempool view that lets `cache_engine.py` query free/used
    counts via `engine.mempool.num_free_blocks` etc."""
    _tree: Any  # shmradix.RadixClient; the extension ships no stubs

    @property
    def num_total_blocks(self) -> int:
        # mempool_total() is the slot capacity; total_blocks() counts blocks
        # currently held by the tree.
        return int(self._tree.mempool_total())

    @property
    def num_free_blocks(self) -> int:
        return int(self._tree.mempool_free())

    @property
    def num_used_blocks(self) -> int:
        return int(self._tree.mempool_used())
