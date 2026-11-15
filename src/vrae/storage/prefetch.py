"""Bounded background prefetch of the videos a reader will need next, in the order it will need them.

The sampler announces each epoch's rank-local order (start) and the position the DataLoader has drawn
(advance), so the window follows the real order, including a resumed epoch's remainder and the DataLoader's
own read-ahead. Prefetch only warms the node cache: it never changes, reorders or skips what is yielded, and
it uses no global RNG. Requests are deduplicated per video, the queue holds at most `depth` items, `workers`
threads per rank download, and the cache caps prefetch downloads and reserved bytes node-wide (prefetch never
waits for room; the foreground read fetches anything prefetch skipped).
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import OrderedDict

log = logging.getLogger(__name__)


class Prefetcher:
    def __init__(self, video_for_item, key_for_item, prefetch_one, *, depth: int = 64, workers: int = 4):
        """video_for_item(item) -> object to prefetch (or None for a local file);
        key_for_item(item) -> hashable video identity for deduplication (cheap; runs on the reader's thread);
        prefetch_one(video) -> status string."""
        if depth < 1 or workers < 1:
            raise ValueError("prefetch depth and workers must be positive")
        self.video_for_item, self.key_for_item, self.prefetch_one = video_for_item, key_for_item, prefetch_one
        self.depth, self.workers = depth, workers
        self._lock = threading.Lock()
        self._queue: queue.Queue = queue.Queue(maxsize=depth)
        self._generation = 0
        self._order, self._next = (), 0
        self._recent: OrderedDict = OrderedDict()
        self._threads: list = []
        self._stop = threading.Event()
        self._errors = 0
        self.counters = {"scheduled": 0, "deduplicated": 0, "errors": 0}

    def __getstate__(self):
        raise TypeError("a Prefetcher lives in the process that owns the sampler; it is not picklable")

    def _start_threads(self):
        if self._threads:
            return
        for index in range(self.workers):
            thread = threading.Thread(target=self._run, name=f"video-prefetch-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def start(self, order) -> None:
        """A new sequence (epoch): forget the old queue and follow `order` from its beginning."""
        with self._lock:
            self._generation += 1
            self._order, self._next = order, 0
            self._recent.clear()
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._start_threads()

    def advance(self, position: int) -> None:
        """The reader has drawn order[position]; schedule order[position + 1 : position + 1 + depth]."""
        with self._lock:
            order = self._order
            self._next = max(self._next, position + 1)
            target = min(len(order), position + 1 + self.depth)
            while self._next < target:
                item = order[self._next]
                key = self.key_for_item(item)
                if key in self._recent:
                    self.counters["deduplicated"] += 1
                else:
                    try:
                        self._queue.put_nowait((self._generation, item))
                    except queue.Full:
                        return  # resume from here at the next advance
                    self._recent[key] = None
                    if len(self._recent) > 4 * self.depth:
                        self._recent.popitem(last=False)
                    self.counters["scheduled"] += 1
                self._next += 1

    def _run(self):
        while not self._stop.is_set():
            try:
                generation, item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if generation != self._generation:
                continue
            try:
                video = self.video_for_item(item)
                if video is not None:
                    self.prefetch_one(video)
            except Exception as error:  # noqa: BLE001 - the foreground read reports real failures
                self.counters["errors"] += 1
                self._errors += 1
                if self._errors <= 5 or self._errors % 1000 == 0:
                    log.warning("video prefetch failed (the foreground read will retry): %s", error)

    def close(self, timeout: float = 30.0) -> None:
        self._stop.set()
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        self._threads = []


class ObservedOrder:
    """Adapter for sequential readers (FVD): prefetch along a fixed list of paths."""

    def __init__(self, prefetcher: Prefetcher, items):
        self.prefetcher = prefetcher
        prefetcher.start(list(range(len(items))))

    def advance(self, position):
        self.prefetcher.advance(position)

    def close(self):
        self.prefetcher.close()
