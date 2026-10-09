"""Model-neutral Agent prompt behavior across Token and latent communication modes."""
import unittest

from latentllm.agent.prompt import ROLES, build_messages


class PromptTests(unittest.TestCase):
    def test_all_roles_work_without_model_name(self):
        for media in ('token', 'kv'):
            for role in ROLES:
                with self.subTest(media=media, role=role):
                    messages = build_messages(role, 'Compute 8 + 7.', '[Planner]: plan', media=media)
                    self.assertEqual([part['role'] for part in messages], ['system', 'user'])
                    self.assertIn('Compute 8 + 7.', messages[1]['content'])
                    self.assertNotIn('Qwen', str(messages))
                    self.assertNotIn('Alibaba', str(messages))

    def test_token_contains_actual_prior_text_but_latent_never_invents_it(self):
        token = build_messages('critic', 'Solve X', 'prior-plan-content', media='token')
        latent = build_messages('critic', 'Solve X', 'prior-plan-content', media='kv')
        self.assertIn('prior-plan-content', token[1]['content'])
        self.assertNotIn('prior-plan-content', latent[1]['content'])
        self.assertIn('latent attention-cache', latent[1]['content'])

    def test_judger_uses_final_answer_format(self):
        for media in ('token', 'kv'):
            self.assertIn('\\boxed{YOUR_FINAL_ANSWER}', build_messages('judger', '8 + 7?', media=media)[1]['content'])

    def test_invalid_role_and_media_fail_explicitly(self):
        with self.assertRaises(ValueError):
            build_messages('unknown', 'question', media='token')
        with self.assertRaises(ValueError):
            build_messages('planner', 'question', media='unknown')
