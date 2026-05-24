"""Text augmentation generator.

Categories: ``typos``, ``case_change``, ``synonym_swap``, ``emoji_inject``,
``truncation``, ``padding``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import BaseGenerator

_CATEGORIES = (
    "typos",
    "case_change",
    "synonym_swap",
    "emoji_inject",
    "truncation",
    "padding",
)

# A tiny built-in synonym map. Users can extend it by passing their own.
_DEFAULT_SYNONYMS: dict[str, list[str]] = {
    "good": ["great", "fine", "decent", "solid"],
    "bad": ["poor", "lousy", "subpar", "rough"],
    "fast": ["quick", "rapid", "speedy"],
    "slow": ["sluggish", "lagging", "delayed"],
    "big": ["large", "huge", "sizable"],
    "small": ["tiny", "little", "compact"],
    "happy": ["glad", "pleased", "content"],
    "sad": ["unhappy", "down", "blue"],
    "buy": ["purchase", "acquire"],
    "sell": ["offload", "vend"],
}

_EMOJIS = ("😀", "🔥", "✨", "🚀", "💯", "👀", "🤔", "👍", "💀", "🎉")


class TextGenerator(BaseGenerator):
    """Generate perturbed strings.

    Parameters
    ----------
    random_state
        Seed for reproducible sampling and perturbation choices.
    typo_rate
        Fraction of characters mutated by the ``typos`` category.
    synonym_map
        Optional ``{word: [alternatives, ...]}`` map used by
        ``synonym_swap``. Falls back to a small built-in map.
    pad_token
        Token repeated by the ``padding`` category.
    """

    def __init__(
        self,
        random_state: int | None = None,
        typo_rate: float = 0.05,
        synonym_map: dict[str, list[str]] | None = None,
        pad_token: str = " <PAD>",
    ) -> None:
        super().__init__(random_state=random_state)
        if not 0.0 < typo_rate < 1.0:
            raise ValueError("typo_rate must be in (0, 1).")
        self.typo_rate = typo_rate
        self.synonym_map = synonym_map or _DEFAULT_SYNONYMS
        self.pad_token = pad_token

    def categories(self) -> Sequence[str]:
        return _CATEGORIES

    def _apply(self, category: str, item: Any) -> str:
        if not isinstance(item, str):
            raise TypeError("TextGenerator expects str inputs.")
        return getattr(self, f"_{category}")(item)

    # --- categories ------------------------------------------------------

    def _typos(self, text: str) -> str:
        if not text:
            return text
        chars = list(text)
        n_mutations = max(1, int(len(chars) * self.typo_rate))
        positions = self._rng.choice(len(chars), size=n_mutations, replace=False)
        for pos in positions:
            op = self._rng.integers(0, 4)
            if op == 0 and pos + 1 < len(chars):  # swap with neighbor
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            elif op == 1:  # delete
                chars[pos] = ""
            elif op == 2:  # duplicate
                chars[pos] = chars[pos] * 2
            else:  # replace with adjacent letter
                if chars[pos].isalpha():
                    shift = int(self._rng.integers(-1, 2))
                    chars[pos] = chr(ord(chars[pos]) + shift)
        return "".join(chars)

    def _case_change(self, text: str) -> str:
        ops = ("upper", "lower", "swap", "title")
        op = ops[int(self._rng.integers(0, len(ops)))]
        if op == "upper":
            return text.upper()
        if op == "lower":
            return text.lower()
        if op == "title":
            return text.title()
        return text.swapcase()

    def _synonym_swap(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        for i, word in enumerate(words):
            key = word.lower().strip(".,!?;:'\"")
            choices = self.synonym_map.get(key)
            if choices and self._rng.random() < 0.5:
                replacement = choices[int(self._rng.integers(0, len(choices)))]
                # Preserve simple capitalization.
                if word[:1].isupper():
                    replacement = replacement.capitalize()
                words[i] = word.replace(key, replacement) if key in word else replacement
        return " ".join(words)

    def _emoji_inject(self, text: str) -> str:
        n = int(self._rng.integers(1, 4))
        picks = [_EMOJIS[int(self._rng.integers(0, len(_EMOJIS)))] for _ in range(n)]
        if not text:
            return " ".join(picks)
        # Insert at random word boundaries.
        words = text.split()
        for emoji in picks:
            pos = int(self._rng.integers(0, len(words) + 1))
            words.insert(pos, emoji)
        return " ".join(words)

    def _truncation(self, text: str) -> str:
        if len(text) <= 1:
            return text
        keep = max(1, int(len(text) * float(self._rng.uniform(0.3, 0.8))))
        return text[:keep]

    def _padding(self, text: str) -> str:
        repeats = int(self._rng.integers(5, 25))
        return text + (self.pad_token * repeats)
