import os
import time
import unittest
from trace_worker import _run_isolated, run_trace_job, TraceTimeout, TraceFailed


def hang(connection):
    time.sleep(60)


def crash(connection):
    os._exit(17)


class IsolationTests(unittest.TestCase):
    def test_timeout_and_crash_do_not_poison_next_job(self):
        started=time.monotonic()
        with self.assertRaises(TraceTimeout):
            _run_isolated(hang,(),.2)
        self.assertLess(time.monotonic()-started,5)
        with self.assertRaises(TraceFailed):
            _run_isolated(crash,(),10)
        svg=run_trace_job(bytes([20,80,140,255])*100,10,10,{'edgeStyle':'smooth'},10)
        self.assertIn('<path',svg)
