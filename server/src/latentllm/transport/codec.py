"""Lossless Token ID and Hugging Face KV tensor binary codecs."""
from __future__ import annotations

import struct
from typing import Any

import torch

from .schema import model

_DTYPE = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'float64': torch.float64,
}


def encode_tokens(tokens: list[int]) -> bytes:
    if any(token < 0 or token > 0xFFFFFFFF for token in tokens):
        raise ValueError('Token ids must fit into uint32')
    return struct.pack(f'<{len(tokens)}I', *tokens)


def decode_tokens(data: bytes) -> list[int]:
    if len(data) % 4:
        raise ValueError('Invalid Token ID payload length')
    return list(struct.unpack(f'<{len(data) // 4}I', data))


def _layers(cache: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, 'to_legacy_cache'):
        return list(cache.to_legacy_cache())
    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
        return list(zip(cache.key_cache, cache.value_cache, strict=True))
    if hasattr(cache, 'layers'):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if isinstance(cache, (tuple, list)):
        return list(cache)
    raise TypeError(f'Unsupported KV-Cache layout: {type(cache).__name__}')


def encode_cache(cache: Any) -> tuple[bytes, list[model.Tensor], int]:
    """Copy all K/V layers into contiguous CPU bytes, retaining dtype and layout."""
    layers = _layers(cache)
    if not layers:
        raise ValueError('Cannot send an empty KV-Cache')
    metadata: list[model.Tensor] = []
    buffer = bytearray()
    positions = -1
    for index, (key, value) in enumerate(layers):
        for name, tensor in (('key', key), ('value', value)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError('KV entries must be torch.Tensor')
            if tensor.dtype not in _DTYPE.values():
                raise TypeError(f'Unsupported tensor dtype: {tensor.dtype}')
            if tensor.ndim != 4:
                raise ValueError('KV tensors must have [batch, heads, sequence, dimension] shape')
            if positions < 0:
                positions = tensor.shape[-2]
            elif tensor.shape[-2] != positions:
                raise ValueError('KV tensor sequence positions must match')
            packed = tensor.detach().to('cpu').contiguous().view(torch.uint8).numpy().tobytes()
            metadata.append(model.Tensor(name=f'{index}.{name}', dtype=str(tensor.dtype).removeprefix('torch.'), shape=list(tensor.shape), offset=len(buffer), size=len(packed)))
            buffer.extend(packed)
    return bytes(buffer), metadata, positions


def decode_cache(payload: bytes, tensors: list[model.Tensor], device: str = 'cpu') -> Any:
    """Recover a Transformers DynamicCache without executing arbitrary pickle code."""
    if not tensors or len(tensors) % 2:
        raise ValueError('A KV-Cache requires matched key/value tensors')
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index in range(len(tensors) // 2):
        result: list[torch.Tensor] = []
        for kind in ('key', 'value'):
            spec = tensors[index * 2 + len(result)]
            if spec.name != f'{index}.{kind}':
                raise ValueError('Unexpected KV tensor order')
            dtype = _DTYPE.get(spec.dtype)
            if dtype is None:
                raise ValueError(f'Unknown KV tensor dtype: {spec.dtype}')
            shape = tuple(spec.shape)
            if len(shape) != 4 or any(dim <= 0 for dim in shape):
                raise ValueError('Invalid KV tensor dimensions')
            if spec.offset + spec.size > len(payload):
                raise ValueError('KV payload shorter than tensor metadata')
            expected = torch.empty((), dtype=dtype).element_size()
            for dim in shape:
                expected *= dim
            if spec.size != expected:
                raise ValueError('KV tensor size does not match shape and dtype')
            raw = bytearray(payload[spec.offset:spec.offset + spec.size])
            tensor = torch.frombuffer(raw, dtype=torch.uint8).view(dtype).reshape(shape).to(device).clone()
            result.append(tensor)
        layers.append((result[0], result[1]))
    try:
        from transformers import DynamicCache
        if hasattr(DynamicCache, 'from_legacy_cache'):
            return DynamicCache.from_legacy_cache(tuple(layers))
        cache = DynamicCache()
        for key, value in layers:
            cache.update(key, value, len(cache.layers))
        return cache
    except ImportError:
        return tuple(layers)
