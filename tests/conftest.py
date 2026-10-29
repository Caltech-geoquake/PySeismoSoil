import multiprocessing
import multiprocessing.pool
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture
def created_pools(
        monkeypatch: pytest.MonkeyPatch,
) -> Iterator[list[multiprocessing.pool.Pool]]:
    # The pools that `multiprocessing.Pool()` creates during the test, so that
    # the test can check that the code closed them: `join()` raises
    # `ValueError` if a pool is still running. (The `ResourceWarning` of an
    # unclosed pool is not a reliable check: the pool can be garbage-collected
    # in another thread, after the code under test has returned.)
    pools = []
    create_pool = multiprocessing.Pool

    def create_and_record_pool(
            *args: Any, **kwargs: Any
    ) -> multiprocessing.pool.Pool:
        pool = create_pool(*args, **kwargs)
        pools.append(pool)
        return pool

    monkeypatch.setattr(multiprocessing, 'Pool', create_and_record_pool)
    yield pools
    for pool in pools:
        pool.terminate()  # in case the code under test did not close it
