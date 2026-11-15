"""Training-loop measurements of the data path, under NanoVVM's W&B names (nanovvm train.py)."""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import Any

CACHE_COUNTERS = (
    "demand_requests",
    "demand_hits",
    "demand_misses",
    "demand_bytes",
    "prefetch_downloads",
    "prefetch_bytes",
    "prefetch_used",
    "evictions",
    "evicted_bytes",
    "transport_requests",
    "transport_retries",
    "demand_wait_seconds",
)


def timed_batches(batches: Iterable[Any], wait: list[float]) -> Iterator[Any]:
    """Yield batches, adding the seconds spent waiting for each one to wait[0]."""

    iterator = iter(batches)
    while True:
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            return
        wait[0] += time.perf_counter() - started
        yield batch


def cache_metrics(stats: dict, previous: dict) -> dict[str, float]:
    """W&B scalars of the node video cache: interval deltas of its counters and its current size."""

    delta = {key: stats.get(key, 0) - previous.get(key, 0) for key in CACHE_COUNTERS}
    requests = delta["demand_requests"]
    return {
        "cache/hit_rate": delta["demand_hits"] / requests if requests else 1.0,
        "cache/demand_downloads": delta["demand_misses"],
        "cache/prefetch_downloads": delta["prefetch_downloads"],
        "cache/prefetch_used": delta["prefetch_used"],
        "cache/fetched_gib": (delta["demand_bytes"] + delta["prefetch_bytes"]) / 2**30,
        "cache/requests": delta["transport_requests"],
        "cache/retries": delta["transport_retries"],
        "cache/evictions": delta["evictions"],
        "cache/demand_wait_seconds": delta["demand_wait_seconds"],
        "cache/used_gib": stats.get("used_bytes", 0) / 2**30,
        "cache/inflight_downloads": stats.get("inflight_downloads", 0),
    }


__all__ = ["CACHE_COUNTERS", "cache_metrics", "timed_batches"]
