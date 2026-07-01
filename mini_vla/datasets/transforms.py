"""Text tokenisation utilities for instruction encoding.

Uses a minimal hard-coded vocabulary — no ``transformers`` dependency.
"""

from __future__ import annotations

from typing import List, Optional

# Minimal vocabulary covering Stage 1 instructions.
# Extend as needed when new instructions are added.
VOCAB: dict[str, int] = {
    "<pad>": 0,
    "<unk>": 1,
    "move": 2,
    "red": 3,
    "object": 4,
    "to": 5,
    "target": 6,
    "left": 7,
    "right": 8,
    "up": 9,
    "down": 10,
}

# Reverse mapping for debugging / decoding
ID_TO_TOKEN: dict[int, str] = {v: k for k, v in VOCAB.items()}


def tokenize(text: str, max_len: Optional[int] = None) -> List[int]:
    """Convert a text instruction to a list of token IDs.

    Unknown words are mapped to ``<unk>`` (1).
    If *max_len* is given, the sequence is padded or truncated.
    """
    tokens = [VOCAB.get(word, VOCAB["<unk>"]) for word in text.strip().split()]
    if max_len is not None:
        if len(tokens) < max_len:
            tokens += [VOCAB["<pad>"]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
    return tokens


def decode(tokens: List[int]) -> str:
    """Convert token IDs back to a space-separated string (for debugging)."""
    return " ".join(ID_TO_TOKEN.get(t, "<unk>") for t in tokens)


__all__ = [
    "VOCAB",
    "ID_TO_TOKEN",
    "tokenize",
    "decode",
]
