"""Conservative extraction of an answer the agent actually emitted to the user."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_BOXED = re.compile(r'\\boxed\s*\{\s*([^{}]+?)\s*\}')
_NUMBER = r'([+-]?\d+(?:\.\d+)?|[A-D])'
_MARKER = re.compile(
    r'(?:'
    r'(?:最终(?:的)?|正确(?:的)?|标准)?答案\s*(?:应该|应当|应)?\s*(?:是|为|[:：=])'
    r'|(?:the\s+)?(?:final\s+|correct\s+)?answer\s*(?:is|equals|should\s+be|[:=])'
    r'|final\s+result\s*(?:is|equals|[:=])'
    r')\s*\$?\s*' + _NUMBER + r'(?!\d|\.\d)',
    re.IGNORECASE,
)
_BARE = re.compile(r'^\s*\$?\s*' + _NUMBER + r'\s*\$?\s*[.。!！]?\s*$')


def visible_answer(output: str) -> str:
    """Ignore `<think>` content unless a complete closing tag precedes the answer."""
    if '</think>' in output:
        return output.rsplit('</think>', 1)[-1].strip()
    if output.lstrip().startswith('<think>'):
        return ''
    return output.strip()


def extract_answer(output: str) -> str | None:
    """Do not silently use the final number from an arbitrary calculation."""
    visible = visible_answer(output)
    boxes = _BOXED.findall(visible)
    if boxes:
        return boxes[-1].strip()
    candidates = _MARKER.findall(visible)
    if candidates:
        return candidates[-1].strip()
    bare = _BARE.fullmatch(visible)
    return bare.group(1) if bare else None


def equivalent_answer(prediction: str, expected: str) -> bool:
    a, b = prediction.strip(), expected.strip()
    try:
        return Decimal(a) == Decimal(b)
    except InvalidOperation:
        return a.casefold() == b.casefold()
