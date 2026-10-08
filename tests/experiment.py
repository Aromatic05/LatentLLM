"""Behavior-level persistence of completed experiment traces."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from latentllm.experiment.run import Experiment
from latentllm.experiment.answer import extract_answer, equivalent_answer


class ExperimentTests(unittest.TestCase):
    def test_answer_uses_visible_boxed_result_only(self):
        self.assertEqual(extract_answer(r'<think>Wrong: \boxed{29}</think>\boxed{47}'), '47')
        self.assertIsNone(extract_answer(r'<think>\boxed{47}</think>Original plan without final answer'))

    def test_answer_from_explicit_final_phrase(self):
        output = '中间运算得到 58，但是不符合约束。最终正确的答案应该是47。'
        self.assertEqual(extract_answer(output), '47')
        self.assertEqual(extract_answer('The final answer is 47.'), '47')
        self.assertEqual(extract_answer('最终答案：47。'), '47')
        self.assertTrue(equivalent_answer('47.0', '47'))

    def test_ambiguous_numbers_are_not_scored(self):
        self.assertIsNone(extract_answer('前面有两个数字 11 与 27，得出差为 3。'))
        self.assertIsNone(extract_answer(r'<think>\boxed{47}</think>过程是 11 - 3 = 8。'))

    def test_past_experiment_is_regraded_when_loaded(self):
        with TemporaryDirectory() as temporary:
            store = Path(temporary)
            old = {
                'id': 'old-scored-run', 'prompt': 'Find the two-digit number',
                'media': 'token', 'status': 'completed', 'expected': '47',
                'prediction': None, 'correct': False,
                'startedAt': '2026-10-08T15:00:00+00:00',
                'finishedAt': '2026-10-08T15:00:02+00:00',
                'agents': [], 'transfers': [], 'events': [],
                'answer': '所以正确的答案应该是47。',
            }
            (store / 'run-old-scored-run.json').write_text(__import__('json').dumps(old))
            experiment = Experiment(('127.0.0.1:50051', '127.0.0.1:50052'), store=store)
            historical = experiment.snapshot()['status']['history'][0]
            self.assertEqual(historical['prediction'], '47')
            self.assertIs(historical['correct'], True)

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
