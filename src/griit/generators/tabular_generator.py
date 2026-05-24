"""Tabular augmentation generator.

Categories: ``null_injection``, ``outlier_injection``, ``column_shuffle``,
``new_category``.

Each item passed to :meth:`generate` should be a ``pandas.DataFrame``
representing one row (or a small group of rows). Returning a DataFrame
keeps column names intact for downstream model adapters.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import BaseGenerator

_CATEGORIES = (
    "null_injection",
    "outlier_injection",
    "column_shuffle",
    "new_category",
)


class TabularGenerator(BaseGenerator):
    """Generate corrupted tabular rows.

    Parameters
    ----------
    null_rate
        Fraction of cells overwritten with NaN by ``null_injection``.
    outlier_sigma
        Multiplier on per-column std applied by ``outlier_injection``.
    new_category_token
        Sentinel string injected into object/categorical columns.
    """

    def __init__(
        self,
        random_state: int | None = None,
        null_rate: float = 0.2,
        outlier_sigma: float = 6.0,
        new_category_token: str = "__griit_unseen__",
    ) -> None:
        super().__init__(random_state=random_state)
        if not 0.0 < null_rate <= 1.0:
            raise ValueError("null_rate must be in (0, 1].")
        self.null_rate = null_rate
        self.outlier_sigma = outlier_sigma
        self.new_category_token = new_category_token

    def categories(self) -> Sequence[str]:
        return _CATEGORIES

    def _apply(self, category: str, item: Any) -> Any:
        try:
            import pandas as pd  # local import keeps pandas optional
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "TabularGenerator requires pandas; install with `pip install pandas`."
            ) from exc

        if not isinstance(item, pd.DataFrame):
            raise TypeError("TabularGenerator expects pandas.DataFrame inputs.")

        return getattr(self, f"_{category}")(item.copy())

    # --- categories ------------------------------------------------------

    def _null_injection(self, df):
        n_cells = df.size
        n_null = max(1, int(n_cells * self.null_rate))
        flat_idx = self._rng.choice(n_cells, size=n_null, replace=False)
        rows, cols = np.unravel_index(flat_idx, df.shape)
        for r, c in zip(rows, cols):
            df.iat[int(r), int(c)] = np.nan
        return df

    def _outlier_injection(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        for col in numeric_cols:
            std = float(df[col].std()) if df[col].std() > 0 else 1.0
            mean = float(df[col].mean())
            sign = 1 if self._rng.random() < 0.5 else -1
            df[col] = mean + sign * self.outlier_sigma * std
        return df

    def _column_shuffle(self, df):
        cols = list(df.columns)
        permuted = list(self._rng.permutation(cols))
        # Rename shuffled columns so the row's *values* end up under
        # different column labels, simulating a schema drift.
        return df.rename(columns=dict(zip(cols, permuted)))

    def _new_category(self, df):
        try:
            object_cols = df.select_dtypes(include=["object", "string", "category"]).columns
        except TypeError:  # pragma: no cover - older pandas
            object_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(object_cols) == 0:
            # Fall back to first column.
            object_cols = df.columns[:1]
        for col in object_cols:
            df[col] = self.new_category_token
        return df
