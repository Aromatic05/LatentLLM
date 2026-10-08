"""Execute a single LatentMAS role using either text context or latent KV memory."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

from latentllm.transport.codec import decode_cache, decode_tokens, encode_cache, encode_tokens
from latentllm.transport.schema import model as wire


class Runtime:
    def __init__(self, name: str | None, device: str = 'auto', *, latent_steps: int = 8, max_tokens: int = 192):
        self.name = name or ''
        self.device = device
        self.latent_steps = latent_steps
        self.max_tokens = max_tokens
        self.ready = False
        self.model: Any = None
        self.args = argparse.Namespace(model_name=self.name, task='gsm8k', prompt='sequential', think=False,
                                       text_mas_context_length=100000, latent_space_realign=False)

    def load(self) -> None:
        if not self.name:
            return
        import torch
        root = Path(__file__).resolve().parents[4]
        upstream = root / 'upstream/LatentMAS'
        if not upstream.is_dir():
            raise FileNotFoundError(f'LatentMAS submodule unavailable: {upstream}')
        if str(upstream) not in sys.path:
            sys.path.insert(0, str(upstream))
        from models import ModelWrapper
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ModelWrapper(self.name, torch.device(self.device), use_vllm=False, args=self.args)
        self.ready = True

    def step(self, request: wire.StepRequest, incoming: tuple[wire.Header, bytes] | None) -> tuple[dict, wire.Header, bytes]:
        if not self.ready:
            raise RuntimeError('Model is not loaded')
        import torch
        from prompts import build_agent_message_sequential_latent_mas, build_agent_messages_sequential_text_mas
        started = time.perf_counter_ns()
        header = wire.Header(run_id=request.run_id, model=self.name, media=request.media)
        upstream_cache = None
        context = ''
        if incoming:
            prev, data = incoming
            if prev.model != self.name or prev.media != request.media:
                raise ValueError('Incoming model or communication mode mismatch')
            if request.media == wire.KV_CACHE:
                upstream_cache = decode_cache(data, list(prev.tensors), self.device)
            elif request.media == wire.TOKEN:
                context = self.model.tokenizer.decode(decode_tokens(data), skip_special_tokens=False)
            else:
                raise ValueError('Invalid media')
        elif request.role != 'planner':
            raise ValueError('Missing previous agent state')
        if request.media == wire.TOKEN:
            messages = build_agent_messages_sequential_text_mas(
                role=request.role, question=request.question, context=context,
                method='text_mas', args=self.args)
            prompts, ids, mask, _ = self.model.prepare_chat_batch([messages], add_generation_prompt=True)
            generated, _ = self.model.generate_text_batch(
                ids, mask, max_new_tokens=self.max_tokens, temperature=0.6, top_p=0.95)
            output = generated[0].strip()
            if request.role != 'judger':
                context += f'[{request.role.title()}]:\n{output}\n\n'
            ids = self.model.tokenizer(context, add_special_tokens=False)['input_ids']
            payload = encode_tokens(ids)
            header.tokens = len(ids)
            header.positions = len(ids)
            count = len(self.model.tokenizer(output, add_special_tokens=False)['input_ids'])
        elif request.media == wire.KV_CACHE:
            messages = build_agent_message_sequential_latent_mas(
                role=request.role, question=request.question, context='',
                method='latent_mas', args=self.args)
            prompts, ids, mask, _ = self.model.prepare_chat_batch([messages], add_generation_prompt=True)
            if request.role != 'judger':
                with torch.no_grad():
                    cache = self.model.generate_latent_batch(
                        ids, attention_mask=mask, latent_steps=self.latent_steps,
                        past_key_values=upstream_cache)
                payload, tensor_specs, positions = encode_cache(cache)
                header.tensors.extend(tensor_specs)
                header.positions = positions
                header.tokens = int(mask.sum().item())
                output = ''
                count = 0
            else:
                generated, _ = self.model.generate_text_batch(
                    ids, mask, max_new_tokens=self.max_tokens, temperature=0.6,
                    top_p=0.95, past_key_values=upstream_cache)
                output = generated[0].strip()
                count = len(self.model.tokenizer(output, add_special_tokens=False)['input_ids'])
                payload = b''
        else:
            raise ValueError(f'Unsupported media: {request.media}')
        if self.device.startswith('cuda'):
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        return dict(output=output, tokens=count, positions=header.positions, duration_ms=elapsed), header, payload
