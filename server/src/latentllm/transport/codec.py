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


def _span(spec: model.Tensor, data: bytes) -> tuple[int, int, int]:
    """Validate the packed [batch, heads, sequence, dimension] byte layout."""
    dtype = _DTYPE.get(spec.dtype)
    shape = list(spec.shape)
    if dtype is None or len(shape) != 4 or any(dim <= 0 for dim in shape):
        raise ValueError('Invalid KV tensor dtype or dimensions')
    element = torch.empty((), dtype=dtype).element_size()
    per_token = shape[-1] * element
    heads = shape[0] * shape[1]
    stride = shape[-2] * per_token
    if spec.size != heads * stride or spec.offset + spec.size > len(data):
        raise ValueError('KV tensor metadata does not match its packed bytes')
    return heads, stride, per_token


def slice_cache(base: model.Header, previous: bytes,
                current: model.Header, payload: bytes) -> tuple[bytes, list[model.Tensor]]:
    """Extract newly appended sequence positions; reject any modified prefix.

    Chunk boundaries are deliberately independent from tensor boundaries.
    Each [batch, head] has its own contiguous sequence, so a raw payload
    suffix is NOT a valid KV delta.
    """
    start, end = base.positions, current.positions
    if not 0 < start < end or not base.tensors or len(base.tensors) != len(current.tensors):
        raise ValueError('KV base or sequence interval is incompatible')
    output = bytearray()
    metadata: list[model.Tensor] = []
    for old, new in zip(base.tensors, current.tensors, strict=True):
        if old.name != new.name or old.dtype != new.dtype:
            raise ValueError('KV tensor identity changed')
        if list(old.shape[:2]) + list(old.shape[3:]) != list(new.shape[:2]) + list(new.shape[3:]):
            raise ValueError('KV tensor head/dimension layout changed')
        old_heads, old_stride, old_token = _span(old, previous)
        new_heads, new_stride, new_token = _span(new, payload)
        if old_heads != new_heads or old_token != new_token:
            raise ValueError('KV tensor geometry changed')
        if old.shape[2] != start or new.shape[2] != end:
            raise ValueError('KV sequence length differs from header')
        offset = len(output)
        old_view, new_view = memoryview(previous), memoryview(payload)
        for head in range(old_heads):
            a = old_view[old.offset + head * old_stride: old.offset + (head + 1) * old_stride]
            b = new_view[new.offset + head * new_stride: new.offset + head * new_stride + old_stride]
            if a != b:
                raise ValueError(f'KV prefix changed in {old.name}, head {head}')
            output.extend(new_view[new.offset + head * new_stride + old_stride:
                                   new.offset + (head + 1) * new_stride])
        metadata.append(model.Tensor(name=new.name, dtype=new.dtype,
                                     shape=[*new.shape[:2], end - start, new.shape[3]],
                                     offset=offset, size=len(output) - offset))
    return bytes(output), metadata


def merge_cache(base: model.Header, previous: bytes,
                delta: model.Header, payload: bytes) -> tuple[bytes, list[model.Tensor]]:
    """Reconstruct the exact complete layout from an existing base and APPEND.

    No pickle or object deserialization: just byte concatenation per head.
    The caller must also validate base identity and full-payload SHA-256.
    """
    start, end = delta.start_position, delta.end_position
    if start != base.positions or end != delta.positions or not 0 < start < end:
        raise ValueError('KV append interval does not match cached base')
    if len(base.tensors) != len(delta.tensors) or not base.tensors:
        raise ValueError('KV tensor counts do not match')
    output = bytearray()
    metadata: list[model.Tensor] = []
    for old, fragment in zip(base.tensors, delta.tensors, strict=True):
        if old.name != fragment.name or old.dtype != fragment.dtype:
            raise ValueError('KV tensor identity mismatch')
        if list(old.shape[:2]) + list(old.shape[3:]) != list(fragment.shape[:2]) + list(fragment.shape[3:]):
            raise ValueError('KV tensor head/dimension mismatch')
        heads, old_stride, old_token = _span(old, previous)
        fragment_heads, delta_stride, delta_token = _span(fragment, payload)
        if heads != fragment_heads or old_token != delta_token:
            raise ValueError('KV tensor geometry mismatch')
        if old.shape[2] != start or fragment.shape[2] != end - start:
            raise ValueError('KV tensor sequence mismatch')
        offset = len(output)
        old_view, delta_view = memoryview(previous), memoryview(payload)
        for head in range(heads):
            output.extend(old_view[old.offset + head * old_stride: old.offset + (head + 1) * old_stride])
            output.extend(delta_view[fragment.offset + head * delta_stride:
                                     fragment.offset + (head + 1) * delta_stride])
        metadata.append(model.Tensor(name=old.name, dtype=old.dtype,
                                     shape=[*old.shape[:2], end, old.shape[3]],
                                     offset=offset, size=len(output) - offset))
    return bytes(output), metadata
