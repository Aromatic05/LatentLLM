"""Behavior-level persistence of completed experiment traces."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentllm.experiment.run import Experiment, boxed_answer


class ExperimentTests(unittest.TestCase):
    def test_answer_uses_visible_boxed_result_only(self):
        self.assertEqual(boxed_answer('<think>Wrong: \boxed{29}</think>\boxed{47}'), '47')
        self.assertIsNone(boxed_answer('<think>\boxed{47}</think>Original plan without final answer'))

    def test_finished_run_can_be_reloaded(self):
        with TemporaryDirectory() as temporary:
            store = Path(temporary)
            experiment = Experiment(('127.0.0.1:50051', '127.0.0.1:50052'), store=store)
            run = {
                'id': 'completed-demo', 'prompt': 'Calculate 8 + 7', 'media': 'kv',
                'status': 'completed', 'startedAt': '2026-10-08T15:00:00+00:00',
                'finishedAt': '2026-10-08T15:00:02+00:00',
                'agents': [], 'events': [], 'answer': '\\boxed{15}',
                'transfers': [{'bytes': 512}, {'bytes': 1024}],
            }
            experiment.archive(run)
            loaded = Experiment(('127.0.0.1:50051', '127.0.0.1:50052'), store=store)
            records = loaded.snapshot()['status']['history']
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]['answer'], '\\boxed{15}')
            self.assertEqual(records[0]['bytes'], 1536)
            self.assertEqual(records[0]['durationMs'], 2000)
