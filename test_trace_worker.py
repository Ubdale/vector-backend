import os
import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from trace_worker import _run_isolated, run_trace_job, TraceTimeout, TraceFailed


def hang(connection):
    time.sleep(60)


def crash(connection):
    os._exit(17)


def staged_hang(connection,raw,width,height,settings,directory):
    (Path(directory)/'temporary-input.png').write_bytes(raw)
    Path(settings['marker']).write_text(directory,encoding='utf8')
    time.sleep(60)


class IsolationTests(unittest.TestCase):
    def test_parent_removes_staged_images_after_forced_timeout(self):
        with tempfile.TemporaryDirectory() as outer:
            marker=Path(outer)/'marker.txt'
            with patch('trace_worker._trace_entry',staged_hang):
                with self.assertRaises(TraceTimeout):
                    run_trace_job(b'input image',1,1,{'marker':str(marker)},1)
            self.assertTrue(marker.is_file())
            self.assertFalse(Path(marker.read_text(encoding='utf8')).exists())

    def test_timeout_and_crash_do_not_poison_next_job(self):
        started=time.monotonic()
        with self.assertRaises(TraceTimeout):
            _run_isolated(hang,(),.2)
        self.assertLess(time.monotonic()-started,5)
        with self.assertRaises(TraceFailed):
            _run_isolated(crash,(),10)
        svg=run_trace_job(bytes([20,80,140,255])*100,10,10,{'edgeStyle':'smooth'},10)
        self.assertIn('<path',svg)
