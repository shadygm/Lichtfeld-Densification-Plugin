"""Allocator cleanup is optional and does not change live buffer ownership."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from core.runtime import memory


class RuntimeMemoryTests(unittest.TestCase):
    def setUp(self):
        memory._malloc_trim.cache_clear()
        self.addCleanup(memory._malloc_trim.cache_clear)

    def test_glibc_releases_only_free_pages(self):
        trim = Mock(return_value=1)
        with patch.object(memory.sys, 'platform', 'linux'), \
                patch.object(memory.ctypes, 'CDLL', return_value=SimpleNamespace(malloc_trim=trim)):
            self.assertTrue(memory.release_free_memory())
        trim.assert_called_once_with(0)
        self.assertEqual(trim.argtypes, [memory.ctypes.c_size_t])
        self.assertIs(trim.restype, memory.ctypes.c_int)

    def test_unsupported_platform_does_not_load_native_symbols(self):
        with patch.object(memory.sys, 'platform', 'win32'), \
                patch.object(memory.ctypes, 'CDLL') as loader:
            self.assertFalse(memory.release_free_memory())
        loader.assert_not_called()

    def test_other_linux_allocator_without_extension_is_supported(self):
        with patch.object(memory.sys, 'platform', 'linux'), \
                patch.object(memory.ctypes, 'CDLL', return_value=SimpleNamespace()):
            self.assertFalse(memory.release_free_memory())
