"""Only completed matching work contributes to the throughput display."""
import unittest
from unittest.mock import Mock, patch

from core.pipeline.control import _report_matching_progress


class ProgressTests(unittest.TestCase):
    def test_no_rate_before_first_match(self):
        callback = Mock()
        _report_matching_progress(callback, 0, 10, 100)
        callback.assert_called_once_with(10., 'Matching 0/10')

    def test_rate_uses_completed_references_and_matching_clock(self):
        callback = Mock()
        with patch('core.pipeline.control.time.perf_counter', return_value=110):
            _report_matching_progress(callback, 5, 10, 100)
        callback.assert_called_once_with(50., 'Matching 5/10 | 0.5 it/s')

    def test_skipped_references_advance_progress_without_inflating_rate(self):
        callback = Mock()
        with patch('core.pipeline.control.time.perf_counter', return_value=110):
            _report_matching_progress(callback, 5, 10, 100, matched_refs=2)
        callback.assert_called_once_with(50., 'Matching 5/10 (3 skipped) | 0.2 it/s')

    def test_no_rate_when_only_skipped_references_have_been_seen(self):
        callback = Mock()
        _report_matching_progress(callback, 3, 10, 100, matched_refs=0)
        callback.assert_called_once_with(34., 'Matching 3/10 (3 skipped)')
