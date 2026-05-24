"""Stress testing harness for GRIIT.

A :class:`StressTester` runs a batch of perturbed samples through a wrapped
model and produces a :class:`TestReport` describing per-category failure
rates, calibration, and an overall robustness score (0-100).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .adapters import ModelAdapter, wrap_model
from .generators import AugmentedSample

__all__ = [
    "CategoryResult",
    "TestReport",
    "StressTester",
]


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CategoryResult:
    """Per-category outcome of a stress test."""

    category: str
    n_tested: int
    n_failed: int
    failure_rate: float
    failed_samples: List[AugmentedSample] = field(default_factory=list)


@dataclass
class TestReport:
    """Aggregate result of a :class:`StressTester` run."""

    baseline_accuracy: float
    results_by_category: Mapping[str, CategoryResult]
    overall_failure_rate: float
    robustness_score: float
    top_failure_categories: List[str]
    calibration_score: float = 0.5
    n_total: int = 0
    n_failed: int = 0


# ---------------------------------------------------------------------------
# Stress tester
# ---------------------------------------------------------------------------


class StressTester:
    """Run perturbed samples through a model and score robustness.

    Parameters
    ----------
    model
        A :class:`griit.adapters.ModelAdapter` or any object the adapters
        module knows how to wrap (sklearn estimator, torch ``nn.Module``,
        ``keras.Model``).
    baseline_data
        Optional ``(X, y)`` tuple of clean inputs and labels used to
        compute :attr:`TestReport.baseline_accuracy` before stress testing.
        When omitted the baseline term defaults to ``1.0`` (i.e. it does
        not penalize the score).
    batch_size
        Number of samples passed to ``adapter.predict`` per call.
    max_failed_examples
        Maximum failed samples retained per category in the report.
    """

    def __init__(
        self,
        model: Any,
        baseline_data: Optional[Tuple[Sequence[Any], Sequence[Any]]] = None,
        batch_size: int = 32,
        max_failed_examples: int = 50,
    ) -> None:
        self.adapter: ModelAdapter = (
            model if isinstance(model, ModelAdapter) else wrap_model(model)
        )
        self.baseline_data = baseline_data
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.max_failed_examples = int(max_failed_examples)
        self._last_report: Optional[TestReport] = None

    # --- public API -------------------------------------------------------

    def run(self, samples: Sequence[AugmentedSample]) -> TestReport:
        """Score ``samples`` and return a :class:`TestReport`."""

        if len(samples) == 0:
            raise ValueError("`samples` must contain at least one element.")

        baseline_accuracy = self._compute_baseline_accuracy()

        grouped: dict[str, List[AugmentedSample]] = {}
        for s in samples:
            grouped.setdefault(s.category, []).append(s)

        results_by_category: dict[str, CategoryResult] = {}
        all_correctness: List[bool] = []
        all_max_proba: List[float] = []
        proba_supported = True

        for category, group in grouped.items():
            preds, max_proba = self._predict_group(group, want_proba=proba_supported)
            if max_proba is None:
                proba_supported = False

            failed_samples: List[AugmentedSample] = []
            n_failed = 0
            for sample, pred in zip(group, preds):
                correct = self._labels_match(pred, sample.label)
                if not correct:
                    n_failed += 1
                    if len(failed_samples) < self.max_failed_examples:
                        failed_samples.append(sample)
                all_correctness.append(correct)

            if max_proba is not None:
                all_max_proba.extend(float(p) for p in max_proba)

            n_tested = len(group)
            results_by_category[category] = CategoryResult(
                category=category,
                n_tested=n_tested,
                n_failed=n_failed,
                failure_rate=n_failed / n_tested if n_tested else 0.0,
                failed_samples=failed_samples,
            )

        n_total = len(samples)
        n_failed_total = sum(r.n_failed for r in results_by_category.values())
        overall_failure_rate = n_failed_total / n_total

        if proba_supported and all_max_proba:
            calibration_score = self._calibration(all_max_proba, all_correctness)
        else:
            calibration_score = 0.5

        top_failure_categories = sorted(
            results_by_category.keys(),
            key=lambda c: results_by_category[c].failure_rate,
            reverse=True,
        )
        top_failure_rate = (
            results_by_category[top_failure_categories[0]].failure_rate
            if top_failure_categories
            else 0.0
        )

        robustness_score = 100.0 * (
            baseline_accuracy * 0.30
            + (1.0 - overall_failure_rate) * 0.40
            + calibration_score * 0.15
            + (1.0 - top_failure_rate) * 0.15
        )

        report = TestReport(
            baseline_accuracy=baseline_accuracy,
            results_by_category=results_by_category,
            overall_failure_rate=overall_failure_rate,
            robustness_score=robustness_score,
            top_failure_categories=top_failure_categories,
            calibration_score=calibration_score,
            n_total=n_total,
            n_failed=n_failed_total,
        )
        self._last_report = report
        return report

    def summary(self, report: Optional[TestReport] = None) -> str:
        """Return (and print) a human-readable summary string."""

        report = report or self._last_report
        if report is None:
            raise RuntimeError("No report available. Call `run(...)` first.")

        lines: List[str] = []
        lines.append("GRIIT Stress Test Summary")
        lines.append("=" * 25)
        lines.append(f"Baseline accuracy   : {report.baseline_accuracy * 100:6.2f}%")
        lines.append(f"Calibration score   : {report.calibration_score * 100:6.2f}%")
        lines.append(f"Cases run           : {report.n_total}")
        lines.append(
            f"⚠️  Failed on {report.n_failed}/{report.n_total} cases "
            f"({report.overall_failure_rate * 100:.2f}%)"
        )
        lines.append(f"Robustness score    : {report.robustness_score:6.2f} / 100")
        lines.append("")
        lines.append("Top failure categories:")

        ranked = report.top_failure_categories[:5]
        if not ranked:
            lines.append("  (none)")
        else:
            width = max(len(c) for c in ranked)
            for category in ranked:
                cr = report.results_by_category[category]
                lines.append(
                    f"  {category.ljust(width)}   "
                    f"{cr.failure_rate * 100:6.2f}%  "
                    f"({cr.n_failed}/{cr.n_tested})"
                )

        text = "\n".join(lines)
        print(text)
        return text

    # --- internals --------------------------------------------------------

    def _compute_baseline_accuracy(self) -> float:
        if self.baseline_data is None:
            return 1.0
        X, y = self.baseline_data
        if len(X) == 0:
            return 1.0
        preds = self.adapter.predict(X)
        correct = sum(
            1 for p, t in zip(np.asarray(preds).tolist(), list(y))
            if self._labels_match(p, t)
        )
        return correct / len(X)

    def _predict_group(
        self,
        group: Sequence[AugmentedSample],
        want_proba: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Run predict (and optionally predict_proba) over a category group in batches."""

        preds_chunks: List[np.ndarray] = []
        proba_chunks: List[np.ndarray] = []

        for chunk in _chunked(group, self.batch_size):
            batch = _stack_items([s.item for s in chunk])
            preds = np.asarray(self.adapter.predict(batch))
            if preds.ndim == 0:
                preds = preds.reshape(1)
            preds_chunks.append(preds)

            if want_proba:
                try:
                    proba = np.asarray(self.adapter.predict_proba(batch))
                except NotImplementedError:
                    want_proba = False
                else:
                    if proba.ndim == 1:
                        proba = proba.reshape(-1, 1)
                    proba_chunks.append(proba.max(axis=1))

        all_preds = np.concatenate(preds_chunks, axis=0)
        max_proba = (
            np.concatenate(proba_chunks, axis=0) if proba_chunks else None
        )
        return all_preds, max_proba

    @staticmethod
    def _calibration(max_proba: Sequence[float], correctness: Sequence[bool]) -> float:
        proba = np.asarray(max_proba, dtype=np.float64)
        correct = np.asarray(correctness, dtype=np.float64)
        if proba.size == 0:
            return 0.5
        return float(1.0 - np.mean(np.abs(proba - correct)))

    @staticmethod
    def _labels_match(pred: Any, label: Any) -> bool:
        try:
            if isinstance(pred, np.ndarray) or isinstance(label, np.ndarray):
                return bool(np.array_equal(pred, label))
            if isinstance(pred, float) and isinstance(label, float):
                return math.isclose(pred, label, rel_tol=1e-7, abs_tol=1e-9)
            return pred == label
        except Exception:  # pragma: no cover - defensive
            return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _stack_items(items: Sequence[Any]) -> Any:
    """Pack a list of items into the shape downstream adapters prefer."""

    if len(items) == 0:
        return items
    first = items[0]

    if isinstance(first, np.ndarray) and all(
        isinstance(x, np.ndarray) and x.shape == first.shape for x in items
    ):
        return np.stack(items, axis=0)

    try:
        import pandas as pd

        if isinstance(first, pd.DataFrame) and all(
            isinstance(x, pd.DataFrame) for x in items
        ):
            return pd.concat(items, ignore_index=True)
    except ImportError:  # pragma: no cover
        pass

    return list(items)
