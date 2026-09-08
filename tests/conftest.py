"""
Pytest configuration file for FlexKV tests.
This file contains shared fixtures and setup code for all tests.
"""
import contextlib
import os
import shutil

import pytest

# Import fixtures from common_utils so pytest can discover them
from common_utils import model_config, cache_config, test_config

# Geometry for `radix_shmem_env`. Small on purpose: the tests below assert on
# exact free-block counts, and a 64-block pool makes an accidental leak obvious.
RADIX_TOKENS_PER_BLOCK = 16
RADIX_NUM_BLOCKS = 64


class RadixShmemEnv:
    """A live `GlobalCacheEngine` on a radix-server, plus its geometry."""

    def __init__(self, engine, tokens_per_block: int, num_blocks: int):
        self.engine = engine
        self.cpu = engine.cpu_cache_engine
        self.tokens_per_block = tokens_per_block
        self.num_blocks = num_blocks

    def seq(self, token_ids):
        from flexkv.common.block import SequenceMeta
        import numpy as np
        return SequenceMeta(token_ids=np.asarray(token_ids).copy(),
                            tokens_per_block=self.tokens_per_block)

    def hit(self, tier, token_ids) -> int:
        """Local match length, dropping the query's pin before returning."""
        match = tier.match(self.seq(token_ids))
        match.release()
        return match.num_matched_blocks

    def slots(self, tier, token_ids, num_blocks: int):
        """The tier's own slot ids for the first `num_blocks` matched blocks."""
        import numpy as np
        match = tier.match(self.seq(token_ids))
        slots = np.asarray(match.local_range(0, num_blocks), dtype=np.int64)
        match.release()
        return slots

    def free(self) -> int:
        return self.cpu.num_free_blocks


@pytest.fixture(scope="module")
def radix_shmem_env(request):
    """A radix-server (index + SlotStore) and a `GlobalCacheEngine` bound to it.

    Module-scoped because the server is cheap to start but the tests mutate the
    tree: each module gets its own region names (derived from the module and
    the pid) so a parallel or repeated run never attaches to a live one. Skips
    when shmradix is missing/stale or when `flexkv.c_ext` cannot load.
    """
    try:
        import shmradix  # noqa: F401
    except ImportError as exc:
        # ImportError, not ModuleNotFoundError, is what a stale `_core.so` next to
        # a newer `__init__.py` raises.
        pytest.skip(f"shmradix unusable ({exc}); rebuild the extension")
    try:
        import torch
        from flexkv.cache.cache_engine import GlobalCacheEngine
    except Exception as exc:  # c_ext links libcudart
        pytest.skip(f"GlobalCacheEngine unavailable: {exc}")

    from flexkv.common.config import (CacheConfig, GLOBAL_CONFIG_FROM_ENV,
                                      ModelConfig)
    from flexkv.server.shm_radix_bootstrap import (RadixServerProcess,
                                                   build_radix_server_config,
                                                   radix_data_name,
                                                   radix_index_name,
                                                   radix_socket_path)

    shm_radix_id = f"{request.module.__name__.replace('_', '')}{os.getpid()}"

    saved = {name: getattr(GLOBAL_CONFIG_FROM_ENV, name)
             for name in ("radix_shmem", "shm_radix_id",
                          "radix_world_size", "radix_endpoint")}
    GLOBAL_CONFIG_FROM_ENV.radix_shmem = True
    GLOBAL_CONFIG_FROM_ENV.shm_radix_id = shm_radix_id
    GLOBAL_CONFIG_FROM_ENV.radix_world_size = 1
    GLOBAL_CONFIG_FROM_ENV.radix_endpoint = ""

    model_config = ModelConfig(num_layers=2, num_kv_heads=4, head_size=64,
                               dtype=torch.float16,
                               tp_size=1, dp_size=1)
    cache_config = CacheConfig(
        tokens_per_block=RADIX_TOKENS_PER_BLOCK,
        enable_cpu=True, enable_ssd=False, enable_remote=False,
        num_cpu_blocks=RADIX_NUM_BLOCKS,
    )

    def _sweep() -> None:
        for path in (radix_socket_path(shm_radix_id),
                     "/dev/shm" + radix_index_name(shm_radix_id),
                     "/dev/shm" + radix_data_name(shm_radix_id)):
            with contextlib.suppress(FileNotFoundError):
                os.unlink(path)

    _sweep()
    server = None
    engine = None
    try:
        server = RadixServerProcess(
            build_radix_server_config(model_config, cache_config, shm_radix_id)).start()
        engine = GlobalCacheEngine(cache_config, model_config)
        assert engine.use_radix_shmem, "fixture did not get the radixshmem backend"
        yield RadixShmemEnv(engine, RADIX_TOKENS_PER_BLOCK, RADIX_NUM_BLOCKS)
    finally:
        if engine is not None and engine.cpu_cache_engine is not None:
            engine.cpu_cache_engine.close()
        if server is not None:
            server.shutdown()
        _sweep()
        for name, value in saved.items():
            setattr(GLOBAL_CONFIG_FROM_ENV, name, value)
