"""Failure-case retraining loop for GRIIT.

A :class:`Retrainer` takes a :class:`griit.adapters.ModelAdapter`, a list of
:class:`griit.generators.AugmentedSample` failure cases, and clean baseline
data. It checkpoints the model, fine-tunes on the failures with a chosen
strategy, and rolls back if the baseline accuracy regresses below a
threshold. The outcome is summarized in a :class:`RetrainingResult`.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .adapters import ModelAdapter
from .generators import AugmentedSample
from .tester import _stack_items

__all__ = ["Retrainer", "RetrainingResult"]

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = {"adaptive", "curriculum", "adversarial"}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class RetrainingResult:
    """Summary of a single :meth:`Retrainer.retrain` call."""

    strategy: str
    n_failure_cases_used: int
    baseline_before: float
    baseline_after: float
    edge_case_accuracy_before: float
    edge_case_accuracy_after: float
    rolled_back: bool
    checkpoint_path: Optional[str]
    epochs_trained: int


# ---------------------------------------------------------------------------
# Retrainer
# ---------------------------------------------------------------------------


class Retrainer:
    """Fine-tune a wrapped model on failure cases with safe rollback.

    Parameters
    ----------
    adapter
        A :class:`griit.adapters.ModelAdapter` wrapping the model to update.
    strategy
        One of ``"adaptive"`` (cluster by category, interleave), ``"curriculum"``
        (lowest-confidence failures first), or ``"adversarial"`` (highest-
        confidence wrong predictions first).
    min_baseline_accuracy
        Lower bound on baseline accuracy after retraining. If the post-fit
        model dips below this on the clean baseline set, the checkpoint is
        restored and ``rolled_back=True``.
    max_epochs
        Maximum number of fine-tuning epochs (PyTorch / Keras). For sklearn
        models with ``partial_fit``, each epoch is one full pass.
    learning_rate
        Optimizer learning rate for PyTorch / Keras paths.
    checkpoint_dir
        Directory where pre-retraining weights are stashed.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        strategy: str = "adaptive",
        min_baseline_accuracy: float = 0.80,
        max_epochs: int = 10,
        learning_rate: float = 1e-4,
        checkpoint_dir: str = ".griit_checkpoints",
    ) -> None:
        if not isinstance(adapter, ModelAdapter):
            raise TypeError("adapter must be a griit.adapters.ModelAdapter instance.")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)}; got {strategy!r}."
            )
        if not 0.0 <= min_baseline_accuracy <= 1.0:
            raise ValueError("min_baseline_accuracy must be in [0, 1].")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        self.adapter = adapter
        self.strategy = strategy
        self.min_baseline_accuracy = float(min_baseline_accuracy)
        self.max_epochs = int(max_epochs)
        self.learning_rate = float(learning_rate)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # --- public API -------------------------------------------------------

    def retrain(
        self,
        failure_cases: Sequence[AugmentedSample],
        baseline_data: Tuple[Sequence[Any], Sequence[Any]],
    ) -> RetrainingResult:
        """Fine-tune on ``failure_cases`` and roll back on regression."""

        if len(failure_cases) == 0:
            raise ValueError("`failure_cases` must contain at least one sample.")
        X_base, y_base = baseline_data
        if len(X_base) != len(y_base):
            raise ValueError("baseline_data X and y must have matching lengths.")
        if len(X_base) == 0:
            raise ValueError("baseline_data must contain at least one example.")

        # 1) Snapshot weights before any fitting.
        checkpoint_path = self._save_checkpoint()

        # 2) Measure pre-retraining accuracy on baseline + edge cases.
        baseline_before = self._accuracy(X_base, list(y_base))
        edge_before = self._edge_case_accuracy(failure_cases)

        # 3) Order failure cases according to the chosen strategy.
        ordered = self._sort_failure_cases(failure_cases, self.strategy)

        # 4) Fine-tune.
        epochs_trained, fit_skipped = self._fit(ordered)

        # 5) Re-measure and decide on rollback.
        if fit_skipped:
            # No update happened. Honor the spec by flagging rollback=True so
            # callers know the model is unchanged.
            baseline_after = baseline_before
            edge_after = edge_before
            rolled_back = True
            kept_checkpoint = checkpoint_path
        else:
            baseline_after = self._accuracy(X_base, list(y_base))
            edge_after = self._edge_case_accuracy(failure_cases)

            if baseline_after < self.min_baseline_accuracy:
                self._restore_checkpoint(checkpoint_path)
                rolled_back = True
                # Re-measure post-rollback to confirm restoration.
                baseline_after = self._accuracy(X_base, list(y_base))
                edge_after = self._edge_case_accuracy(failure_cases)
                kept_checkpoint = checkpoint_path
            else:
                rolled_back = False
                # Successful retrain: stash file is no longer useful.
                self._cleanup_checkpoint(checkpoint_path)
                kept_checkpoint = None

        return RetrainingResult(
            strategy=self.strategy,
            n_failure_cases_used=len(ordered),
            baseline_before=baseline_before,
            baseline_after=baseline_after,
            edge_case_accuracy_before=edge_before,
            edge_case_accuracy_after=edge_after,
            rolled_back=rolled_back,
            checkpoint_path=kept_checkpoint,
            epochs_trained=epochs_trained,
        )

    # --- ordering ---------------------------------------------------------

    def _sort_failure_cases(
        self,
        failure_cases: Sequence[AugmentedSample],
        strategy: str,
    ) -> List[AugmentedSample]:
        """Public-ish helper exposed for tests."""

        cases = list(failure_cases)
        if strategy == "adaptive":
            return self._adaptive_order(cases)

        # curriculum / adversarial both depend on max-prob confidence.
        confidences = self._confidences(cases)
        if confidences is None:
            # Probabilities not available -> keep insertion order.
            return cases

        order = list(range(len(cases)))
        order.sort(key=lambda i: confidences[i], reverse=(strategy == "adversarial"))
        return [cases[i] for i in order]

    @staticmethod
    def _adaptive_order(cases: List[AugmentedSample]) -> List[AugmentedSample]:
        """Cluster by category, then interleave round-robin."""
        buckets: dict[str, list[AugmentedSample]] = defaultdict(list)
        for sample in cases:
            buckets[sample.category].append(sample)

        # Stable round-robin draw across categories.
        ordered: List[AugmentedSample] = []
        keys = list(buckets.keys())
        while any(buckets[k] for k in keys):
            for k in keys:
                if buckets[k]:
                    ordered.append(buckets[k].pop(0))
        return ordered

    def _confidences(self, cases: Sequence[AugmentedSample]) -> Optional[List[float]]:
        try:
            batch = _stack_items([s.item for s in cases])
            proba = np.asarray(self.adapter.predict_proba(batch))
        except NotImplementedError:
            return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("predict_proba failed during ordering: %s", exc)
            return None
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        return proba.max(axis=1).astype(float).tolist()

    # --- accuracy ---------------------------------------------------------

    def _accuracy(self, X: Sequence[Any], y: Sequence[Any]) -> float:
        if len(X) == 0:
            return 1.0
        preds = np.asarray(self.adapter.predict(X)).tolist()
        correct = sum(1 for p, t in zip(preds, y) if _labels_match(p, t))
        return correct / len(X)

    def _edge_case_accuracy(self, cases: Sequence[AugmentedSample]) -> float:
        items = [s.item for s in cases]
        labels = [s.label for s in cases]
        return self._accuracy(items, labels)

    # --- checkpointing ----------------------------------------------------

    def _save_checkpoint(self) -> str:
        framework = self.adapter.framework
        stamp = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
        path = os.path.join(self.checkpoint_dir, f"griit_ckpt_{stamp}")

        if framework == "pytorch":
            import torch

            full = path + ".pt"
            torch.save(self.adapter.model.state_dict(), full)
            return full
        if framework == "keras":
            full = path + ".weights.h5"
            self.adapter.model.save_weights(full)
            return full
        if framework == "sklearn":
            import joblib

            full = path + ".joblib"
            joblib.dump(self.adapter.model, full)
            return full
        raise NotImplementedError(
            f"Checkpointing is not supported for framework={framework!r}."
        )

    def _restore_checkpoint(self, path: str) -> None:
        framework = self.adapter.framework
        if framework == "pytorch":
            import torch

            state = torch.load(path, weights_only=True)
            self.adapter.model.load_state_dict(state)
            return
        if framework == "keras":
            self.adapter.model.load_weights(path)
            return
        if framework == "sklearn":
            import joblib

            # IMPORTANT: joblib.load returns a fresh object. We must replace
            # the adapter's `.model` attribute so subsequent adapter.predict()
            # calls hit the restored estimator, not the fine-tuned one.
            self.adapter.model = joblib.load(path)
            return
        raise NotImplementedError(
            f"Restoring is not supported for framework={framework!r}."
        )

    @staticmethod
    def _cleanup_checkpoint(path: str) -> None:
        try:
            os.remove(path)
        except OSError as exc:
            logger.warning("Failed to remove checkpoint %s: %s", path, exc)

    # --- training ---------------------------------------------------------

    def _fit(self, cases: Sequence[AugmentedSample]) -> Tuple[int, bool]:
        """Run the framework-specific fine-tune loop.

        Returns
        -------
        epochs_trained : int
            How many epochs actually ran.
        skipped : bool
            ``True`` when no parameter update occurred (e.g. sklearn without
            ``partial_fit``).
        """

        framework = self.adapter.framework
        if framework == "pytorch":
            return self._fit_pytorch(cases), False
        if framework == "keras":
            return self._fit_keras(cases), False
        if framework == "sklearn":
            return self._fit_sklearn(cases)
        raise NotImplementedError(
            f"Training is not supported for framework={framework!r}."
        )

    def _fit_pytorch(self, cases: Sequence[AugmentedSample]) -> int:
        import torch
        from torch import nn, optim

        model = self.adapter.model
        device = next(model.parameters()).device if any(True for _ in model.parameters()) else None

        items = _stack_items([s.item for s in cases])
        if isinstance(items, np.ndarray):
            X = torch.as_tensor(items, dtype=torch.float32)
        elif isinstance(items, torch.Tensor):
            X = items.float()
        else:
            X = torch.as_tensor(np.asarray(items), dtype=torch.float32)

        labels = np.asarray([s.label for s in cases])
        if labels.dtype.kind not in ("i", "u"):
            try:
                labels = labels.astype(np.int64)
            except ValueError as exc:
                raise ValueError(
                    "PyTorch retraining requires integer labels; got "
                    f"dtype={labels.dtype}."
                ) from exc
        y = torch.as_tensor(labels, dtype=torch.long)

        if device is not None:
            X = X.to(device)
            y = y.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        was_training = bool(getattr(model, "training", False))
        model.train()
        try:
            for epoch in range(self.max_epochs):
                optimizer.zero_grad()
                logits = model(X)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
        finally:
            if not was_training:
                model.eval()
        return self.max_epochs

    def _fit_keras(self, cases: Sequence[AugmentedSample]) -> int:
        items = _stack_items([s.item for s in cases])
        labels = np.asarray([s.label for s in cases])
        if isinstance(items, list):
            items = np.asarray(items)
        self.adapter.model.fit(
            items,
            labels,
            epochs=self.max_epochs,
            verbose=0,
        )
        return self.max_epochs

    def _fit_sklearn(self, cases: Sequence[AugmentedSample]) -> Tuple[int, bool]:
        model = self.adapter.model
        if not hasattr(model, "partial_fit"):
            warnings.warn(
                f"{type(model).__name__} does not support partial_fit; "
                "skipping retraining and flagging rolled_back=True.",
                stacklevel=2,
            )
            return 0, True

        items = _stack_items([s.item for s in cases])
        if isinstance(items, list):
            items = np.asarray(items)
        labels = np.asarray([s.label for s in cases])

        # `partial_fit` requires the full class set on first call.
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = np.unique(labels)
            for _ in range(self.max_epochs):
                model.partial_fit(items, labels, classes=classes)
        else:
            for _ in range(self.max_epochs):
                model.partial_fit(items, labels)
        return self.max_epochs, False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _labels_match(pred: Any, label: Any) -> bool:
    if isinstance(pred, np.ndarray) or isinstance(label, np.ndarray):
        return bool(np.array_equal(pred, label))
    return pred == label
