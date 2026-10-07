"""Lifecycle and backpressure checks that do not need LichtFeld or CUDA."""
import importlib.util
from pathlib import Path
import threading
import time
import unittest

spec = importlib.util.spec_from_file_location("prefetch", Path(__file__).parents[1] / "pipeline" / "prefetch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ThreadedReferenceLoader = module.ThreadedReferenceLoader


class LoaderTests(unittest.TestCase):
    def test_every_package_once_and_empty_dataset(self):
        for values in ([], list(range(25))):
            loader = ThreadedReferenceLoader(values, num_workers=3, prefetch_size=4)
            self.assertEqual(sorted(loader), values)
            self.assertTrue(loader._closed)
            self.assertFalse(loader._pending)

    def test_failure_reaches_consumer(self):
        class BrokenDataset:
            def __len__(self): return 1
            def __getitem__(self, index): raise ValueError("bad image")
        loader = ThreadedReferenceLoader(BrokenDataset())
        with self.assertRaisesRegex(ValueError, "bad image"):
            next(loader)
        self.assertTrue(loader._closed)

    def test_cancellation_and_idempotent_close(self):
        cancelled = threading.Event()
        loader = ThreadedReferenceLoader(list(range(20)), cancel_requested=cancelled.is_set)
        next(loader)
        cancelled.set()
        self.assertEqual(list(loader), [])
        loader.close()
        loader.close()
        self.assertFalse(loader._pending)

    def test_loading_is_bounded_when_consumer_pauses(self):
        started = []
        class Dataset:
            def __len__(self): return 100
            def __getitem__(self, index):
                started.append(index)
                return index
        loader = ThreadedReferenceLoader(Dataset(), num_workers=2, prefetch_size=3)
        next(loader)
        time.sleep(0.05)
        self.assertLessEqual(len(started), 3)
        loader.close()
        self.assertFalse(any(t.name.startswith("lf-pack") for t in threading.enumerate()))

    def test_close_before_start(self):
        loader = ThreadedReferenceLoader([1])
        loader.close()
        self.assertEqual(list(loader), [])


if __name__ == "__main__":
    unittest.main()
