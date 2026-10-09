"""Model-independent, role-specific prompts for distributed Agent handoff."""
from __future__ import annotations


ROLES = ('planner', 'critic', 'refiner', 'judger')
MEDIA = ('token', 'kv')


def build_messages(role: str, question: str, context: str = '', *, media: str) -> list[dict[str, str]]:
    """Build a semantic role prompt; let each model's tokenizer render its own chat template.

    Qwen, Llama and other chat models receive the same task instructions.
    Latent handoffs have no readable previous-agent output, only a restored KV state.
    """
    if role not in ROLES:
        raise ValueError(f'Unknown Agent role: {role}')
    if media not in MEDIA:
        raise ValueError(f'Unknown communication medium: {media}')

    system = 'You are a helpful AI assistant collaborating with other agents to solve the user task.'
    if role == 'planner':
        instruction = (
            'You are the Planner Agent. Outline a concise, logically ordered plan for solving '
            'the question. Explain the main steps, but do not give the final answer.'
        )
    elif role == 'critic':
        instruction = (
            'You are the Critic Agent. Evaluate the earlier plan for correctness, gaps '
            'and assumptions. Give constructive feedback that will help a Refiner Agent.'
        )
    elif role == 'refiner':
        instruction = (
            'You are the Refiner Agent. Improve the earlier plan using the critique. '
            'Produce a clear and concise plan for the final solver.'
        )
    else:
        instruction = (
            'You are the final Judger Agent. Solve the target question using the earlier '
            'agents as optional reference. Their reasoning might be wrong or irrelevant; '
            'prioritize the question and show your own reasoning. '
            'Put your final answer inside \\boxed{YOUR_FINAL_ANSWER}.'
        )

    parts = [f'Target Question:\n{question}', instruction]
    if role != 'planner':
        if media == 'kv':
            parts.append(
                'Earlier Agent work is available only as latent attention-cache state. '
                'Use it when helpful. It is not a textual transcript, so do not claim '
                'to quote or reproduce a previous Agent message.'
            )
        else:
            parts.append(
                'Previous Agent outputs (text; may contain errors):\n'
                + (context.strip() or '(No previous text received)')
            )
    return [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': '\n\n'.join(parts)},
    ]
