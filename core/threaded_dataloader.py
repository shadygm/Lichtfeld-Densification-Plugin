"""Threaded prefetch loader used in embedded Python hosts.

This avoids ``multiprocessing`` and uses a ``ThreadPoolExecutor`` so the
loader works inside the host process on Windows.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Optional, Protocol, TypeVar

T = TypeVar("T")


class _IndexableDataset(Protocol[T]):
    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> T:
        ...


@dataclass
class _LoaderValue(Generic[T]):
    value: T


@dataclass
class _LoaderError:
    error: BaseException


class _LoaderDone:
    pass


_OutputItem = _LoaderValue[T] | _LoaderError | _LoaderDone


class ThreadedReferenceLoader(Iterator[T], Generic[T]):
    """Bounded threaded prefetch iterator for indexable datasets."""

    def __init__(
        self,
        dataset: _IndexableDataset[T],
        num_workers: int = 4,
        prefetch_size: int = 8,
        cancel_requested: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._dataset = dataset
        self._num_workers = max(1, int(num_workers))
        self._prefetch_size = max(1, int(prefetch_size))
        self._cancel_requested = cancel_requested

        self._index_queue: queue.Queue[Optional[int]] = queue.Queue(maxsize=self._prefetch_size)
        self._output_queue: queue.Queue[_OutputItem[T]] = queue.Queue(maxsize=self._prefetch_size)
        self._stop_event = threading.Event()
        self._feeder_done = threading.Event()

        self._executor: Optional[ThreadPoolExecutor] = None
        self._worker_futures: list[Future[None]] = []
        self._feeder_thread: Optional[threading.Thread] = None
        self._close_lock = threading.Lock()
        self._done_workers = 0
        self._started = False
        self._closed = False

    def __iter__(self) -> ThreadedReferenceLoader[T]:
        self._start()
        return self

    def __next__(self) -> T:
        self._start()
        while True:
            if self._is_cancelled():
                self.close(wait=True)
                raise StopIteration
            if self._closed:
                raise StopIteration

            try:
                item = self._output_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._is_cancelled():
                self.close(wait=True)
                raise StopIteration

            if isinstance(item, _LoaderDone):
                self._done_workers += 1
                if self._done_workers >= self._num_workers:
                    self.close(wait=True)
                    raise StopIteration
                continue

            if isinstance(item, _LoaderError):
                self.close(wait=True)
                raise item.error

            return item.value

    def close(self, wait: bool = True) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()

        self._drain_queue(self._index_queue)
        self._drain_queue(self._output_queue)

        # Wake workers waiting for new indices.
        for _ in range(self._num_workers):
            try:
                self._index_queue.put_nowait(None)
            except queue.Full:
                break

        feeder_thread = self._feeder_thread
        if feeder_thread is not None:
            feeder_thread.join(timeout=None if wait else 0.0)
            self._feeder_thread = None

        executor = self._executor
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None
            self._worker_futures = []

    def _start(self) -> None:
        if self._started or self._closed:
            return

        self._executor = ThreadPoolExecutor(
            max_workers=self._num_workers,
            thread_name_prefix="lf-pack",
        )
        for _ in range(self._num_workers):
            future = self._executor.submit(self._worker_loop)
            self._worker_futures.append(future)

        self._feeder_thread = threading.Thread(
            target=self._feeder_loop,
            name="lf-pack-feeder",
            daemon=True,
        )
        self._feeder_thread.start()
        self._started = True

    def _is_cancelled(self) -> bool:
        if self._stop_event.is_set():
            return True
        callback = self._cancel_requested
        if callback is None:
            return False
        try:
            return bool(callback())
        except Exception:
            return False

    def _feeder_loop(self) -> None:
        try:
            total = len(self._dataset)
            for index in range(total):
                if self._closed or self._is_cancelled():
                    break
                while not self._closed:
                    if self._is_cancelled():
                        break
                    try:
                        self._index_queue.put(index, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._is_cancelled():
                    break
        finally:
            self._feeder_done.set()
            for _ in range(self._num_workers):
                while not self._closed:
                    try:
                        self._index_queue.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def _worker_loop(self) -> None:
        try:
            while not self._closed:
                if self._is_cancelled():
                    break

                try:
                    index = self._index_queue.get(timeout=0.1)
                except queue.Empty:
                    if self._feeder_done.is_set():
                        break
                    continue

                if index is None:
                    break

                if self._is_cancelled():
                    break

                try:
                    value = self._dataset[index]
                except BaseException as exc:
                    self._stop_event.set()
                    self._queue_output(_LoaderError(exc))
                    break

                if self._is_cancelled():
                    break

                if not self._queue_output(_LoaderValue(value)):
                    break
        finally:
            self._queue_output(_LoaderDone())

    def _queue_output(self, item: _OutputItem[T]) -> bool:
        while not self._closed:
            if self._stop_event.is_set() and isinstance(item, _LoaderValue):
                return False
            try:
                self._output_queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    @staticmethod
    def _drain_queue(q: queue.Queue) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
