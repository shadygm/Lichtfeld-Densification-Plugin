"""Bounded threaded prefetch for embedded Python hosts (including Windows)."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Callable, Generic, Iterator, Optional, Protocol, TypeVar

T = TypeVar("T")


class _IndexableDataset(Protocol[T]):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> T: ...


class ThreadedReferenceLoader(Iterator[T], Generic[T]):
    """Yield completed packages while bounding the number of in-flight loads."""

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
        self._executor: Optional[ThreadPoolExecutor] = None
        self._pending: set[Future[T]] = set()
        self._next_index = 0
        self._closed = False

    def __iter__(self) -> ThreadedReferenceLoader[T]:
        return self

    def __next__(self) -> T:
        if self._closed:
            raise StopIteration
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._num_workers, thread_name_prefix="lf-pack",
            )
        try:
            while True:
                if self._is_cancelled():
                    self.close()
                    raise StopIteration
                while len(self._pending) < self._prefetch_size and self._next_index < len(self._dataset):
                    self._pending.add(self._executor.submit(self._dataset.__getitem__, self._next_index))
                    self._next_index += 1
                if not self._pending:
                    self.close()
                    raise StopIteration
                completed, _ = wait(self._pending, timeout=0.1, return_when=FIRST_COMPLETED)
                if not completed:
                    continue
                future = completed.pop()
                self._pending.remove(future)
                # Surface loading failures instead of silently treating them as cancellation.
                value = future.result()
                if self._is_cancelled():
                    self.close()
                    raise StopIteration
                return value
        except BaseException:
            self.close()
            raise

    def close(self, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None
        self._pending.clear()

    def _is_cancelled(self) -> bool:
        if self._cancel_requested is None:
            return False
        try:
            return bool(self._cancel_requested())
        except Exception:
            return False
