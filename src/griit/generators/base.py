"""Shared infrastructure for GRIIT generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, NamedTuple, Optional, Sequence

import numpy as np


class AugmentedSample(NamedTuple):
    """A single perturbed sample.

    Tuple-compatible (so it works with ``a, b, c = sample``) while still
    exposing named fields and a clear category label for downstream slicing.
    """

    item: Any
    label: Any
    category: str


class BaseGenerator(ABC):
    """Common scaffolding shared by every modality-specific generator.

    Subclasses implement :meth:`categories` (a list of category names) and
    :meth:`_apply` (one augmentation given a category + item).
    """

    def __init__(self, random_state: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(random_state)

    # --- public API -------------------------------------------------------

    @abstractmethod
    def categories(self) -> Sequence[str]:
        """Names of augmentation categories implemented by this generator."""

    @abstractmethod
    def _apply(self, category: str, item: Any) -> Any:
        """Apply a single augmentation category to a single item."""

    def generate(
        self,
        items: Sequence[Any],
        labels: Sequence[Any],
        n_per_category: int = 500,
        categories: Optional[Iterable[str]] = None,
    ) -> List[AugmentedSample]:
        """Generate perturbed samples grouped by category.

        For every category, ``n_per_category`` items are drawn from
        ``items`` (with replacement when there are fewer inputs than
        requested) and the matching transform is applied. The label is
        carried through unchanged.

        Returns a flat list of :class:`AugmentedSample` tuples
        ``(item, label, category)``.
        """

        if len(items) == 0:
            raise ValueError("`items` must contain at least one element.")
        if len(items) != len(labels):
            raise ValueError(
                f"len(items)={len(items)} does not match len(labels)={len(labels)}"
            )
        if n_per_category <= 0:
            raise ValueError("`n_per_category` must be positive.")

        active_categories = list(categories) if categories else list(self.categories())
        unknown = set(active_categories) - set(self.categories())
        if unknown:
            raise ValueError(
                f"Unknown categories: {sorted(unknown)}. "
                f"Available: {list(self.categories())}"
            )

        n = len(items)
        replace = n < n_per_category
        out: List[AugmentedSample] = []

        for category in active_categories:
            idx = self._rng.choice(n, size=n_per_category, replace=replace)
            for i in idx:
                augmented = self._apply(category, items[int(i)])
                out.append(AugmentedSample(augmented, labels[int(i)], category))

        return out
