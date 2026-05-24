"""Tests for griit.trainer."""

from __future__ import annotations

import os
import warnings
from typing import List

import numpy as np
import pytest

from griit import AugmentedSample, Retrainer, RetrainingResult
from griit.adapters import ModelAdapter, PyTorchAdapter, SKLearnAdapter


# ---------------------------------------------------------------------------
# Confidence-routing fakes
# ---------------------------------------------------------------------------


class ScriptedAdapter(ModelAdapter):
    """Adapter whose predictions and probabilities are scripted in advance."""

    framework = "sklearn"  # so checkpoints route via joblib path

    def __init__(self, predictions, proba=None):
        super().__init__(model=object())
        self._predictions = list(predictions)
        self._proba = proba

    def predict(self, X):
        n = _length(X)
        # Cycle through the canned list so repeated calls keep working.
        return np.asarray(
            [self._predictions[i % len(self._predictions)] for i in range(n)]
        )

    def predict_proba(self, X):
        n = _length(X)
        if self._proba is None:
            return np.tile([0.9, 0.1], (n, 1))
        return np.asarray(self._proba[:n])


def _length(X):
    if hasattr(X, "shape") and getattr(X, "ndim", 0) >= 1:
        return int(X.shape[0])
    return len(X)


# ---------------------------------------------------------------------------
# Sample + baseline helpers
# ---------------------------------------------------------------------------


def _make_failures(n=4, category="blur", label=0):
    return [
        AugmentedSample(item=np.zeros((2, 2, 3), dtype=np.uint8), label=label, category=category)
        for _ in range(n)
    ]


def _baseline(n=4, label=0):
    X = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(n)]
    y = [label] * n
    return X, y


# ---------------------------------------------------------------------------
# Strategy ordering
# ---------------------------------------------------------------------------


def test_curriculum_orders_lowest_confidence_first():
    # Tag each sample with its index as the label so we can read off the order.
    cases = [
        AugmentedSample(item=np.zeros((2, 2, 3), dtype=np.uint8), label=i, category="blur")
        for i in range(4)
    ]
    # Confidence on each sample (max proba): 0.95, 0.55, 0.70, 0.60
    proba = np.array([
        [0.95, 0.05],
        [0.55, 0.45],
        [0.70, 0.30],
        [0.40, 0.60],
    ])
    adapter = ScriptedAdapter(predictions=[0, 0, 0, 0], proba=proba)
    trainer = Retrainer(adapter, strategy="curriculum")
    ordered = trainer._sort_failure_cases(cases, "curriculum")
    # Ascending confidence: 0.55, 0.60, 0.70, 0.95 -> indices 1, 3, 2, 0
    assert [s.label for s in ordered] == [1, 3, 2, 0]


def test_adversarial_orders_highest_confidence_first():
    cases = [
        AugmentedSample(item=np.zeros((2, 2, 3), dtype=np.uint8), label=i, category="blur")
        for i in range(4)
    ]
    proba = np.array([
        [0.95, 0.05],
        [0.55, 0.45],
        [0.70, 0.30],
        [0.40, 0.60],
    ])
    adapter = ScriptedAdapter(predictions=[0, 0, 0, 0], proba=proba)
    trainer = Retrainer(adapter, strategy="adversarial")
    ordered = trainer._sort_failure_cases(cases, "adversarial")
    # Descending confidence: 0.95, 0.70, 0.60, 0.55 -> indices 0, 2, 3, 1
    assert [s.label for s in ordered] == [0, 2, 3, 1]


def test_adaptive_interleaves_categories():
    cases = (
        _make_failures(n=3, category="blur")
        + _make_failures(n=2, category="noise")
        + _make_failures(n=1, category="fog")
    )
    adapter = ScriptedAdapter(predictions=[0])
    trainer = Retrainer(adapter, strategy="adaptive")
    ordered = trainer._sort_failure_cases(cases, "adaptive")
    cats = [s.category for s in ordered]
    # Round-robin: blur, noise, fog, blur, noise, blur
    assert cats == ["blur", "noise", "fog", "blur", "noise", "blur"]


# ---------------------------------------------------------------------------
# Sklearn checkpoint + rollback path (the tricky one)
# ---------------------------------------------------------------------------


class _DegradingSklearn:
    """Mimics a sklearn estimator with state we can degrade via partial_fit."""

    def __init__(self, prediction_value=0):
        self.prediction_value = prediction_value
        self.classes_ = np.array([0, 1])

    def predict(self, X):
        n = _length(X)
        return np.full(n, self.prediction_value, dtype=int)

    def predict_proba(self, X):
        n = _length(X)
        # Confidence depends on prediction_value so curriculum sorting works.
        prob_correct = 0.9
        cols = (
            [prob_correct, 1 - prob_correct]
            if self.prediction_value == 0
            else [1 - prob_correct, prob_correct]
        )
        return np.tile(cols, (n, 1))

    def partial_fit(self, X, y, classes=None):
        # Each call flips the prediction so baseline accuracy collapses
        # whenever the true labels are 0.
        self.prediction_value = 1
        return self


def test_sklearn_rollback_restores_predictions(tmp_path):
    model = _DegradingSklearn(prediction_value=0)
    adapter = SKLearnAdapter(model)
    trainer = Retrainer(
        adapter,
        strategy="curriculum",
        min_baseline_accuracy=0.99,
        max_epochs=1,
        checkpoint_dir=str(tmp_path),
    )

    X_base, y_base = _baseline(n=5, label=0)
    failures = _make_failures(n=4, label=0)

    # Snapshot predictions BEFORE retraining.
    preds_before = adapter.predict(X_base).tolist()
    assert preds_before == [0, 0, 0, 0, 0]

    result = trainer.retrain(failures, (X_base, y_base))

    assert result.rolled_back is True
    assert result.checkpoint_path is not None
    assert os.path.exists(result.checkpoint_path)
    # Critical check: rolled-back predictions must match pre-retraining ones.
    preds_after = adapter.predict(X_base).tolist()
    assert preds_after == preds_before
    # And the adapter's `.model` attribute must have been replaced by the
    # joblib.load result, not just a local variable.
    assert adapter.model is not model or adapter.model.prediction_value == 0


def test_successful_sklearn_retrain_cleans_up_checkpoint(tmp_path):
    """When baseline holds, the checkpoint file should be removed."""
    from sklearn.linear_model import SGDClassifier

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(40, 4))
    y_train = (X_train[:, 0] > 0).astype(int)
    model = SGDClassifier(loss="log_loss", random_state=0)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    adapter = SKLearnAdapter(model)
    trainer = Retrainer(
        adapter,
        strategy="adaptive",
        min_baseline_accuracy=0.0,  # accept any post-fit baseline
        max_epochs=1,
        checkpoint_dir=str(tmp_path),
    )

    failures = [
        AugmentedSample(item=X_train[i], label=int(y_train[i]), category="noise")
        for i in range(8)
    ]
    result = trainer.retrain(failures, (X_train, y_train.tolist()))
    assert result.rolled_back is False
    assert result.checkpoint_path is None
    # Nothing left lying around in the checkpoint directory.
    assert os.listdir(tmp_path) == []


# ---------------------------------------------------------------------------
# Sklearn partial_fit fallback
# ---------------------------------------------------------------------------


class _NoPartialFit:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(_length(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.9, 0.1], (_length(X), 1))


def test_partial_fit_missing_skips_with_warning(tmp_path):
    adapter = SKLearnAdapter(_NoPartialFit())
    trainer = Retrainer(
        adapter,
        strategy="adaptive",
        min_baseline_accuracy=0.0,
        max_epochs=2,
        checkpoint_dir=str(tmp_path),
    )
    X_base, y_base = _baseline(n=3, label=0)
    failures = _make_failures(n=2, label=0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = trainer.retrain(failures, (X_base, y_base))

    assert result.rolled_back is True
    assert result.epochs_trained == 0
    assert any("partial_fit" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# RetrainingResult fields
# ---------------------------------------------------------------------------


def test_retraining_result_fields_populated(tmp_path):
    from sklearn.linear_model import SGDClassifier

    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(20, 3))
    y_train = (X_train[:, 0] > 0).astype(int)
    model = SGDClassifier(loss="log_loss", random_state=1)
    model.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    adapter = SKLearnAdapter(model)
    trainer = Retrainer(
        adapter,
        strategy="adversarial",
        min_baseline_accuracy=0.0,
        max_epochs=2,
        checkpoint_dir=str(tmp_path),
    )

    failures = [
        AugmentedSample(item=X_train[i], label=int(y_train[i]), category="blur")
        for i in range(5)
    ]
    result = trainer.retrain(failures, (X_train, y_train.tolist()))
    assert isinstance(result, RetrainingResult)
    assert result.strategy == "adversarial"
    assert result.n_failure_cases_used == 5
    assert 0.0 <= result.baseline_before <= 1.0
    assert 0.0 <= result.baseline_after <= 1.0
    assert 0.0 <= result.edge_case_accuracy_before <= 1.0
    assert 0.0 <= result.edge_case_accuracy_after <= 1.0
    assert result.epochs_trained == 2


# ---------------------------------------------------------------------------
# PyTorch path: checkpoint is created
# ---------------------------------------------------------------------------


def test_pytorch_checkpoint_created_and_cleaned(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(16, 4)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)

    model = TinyNet()
    adapter = PyTorchAdapter(model)
    trainer = Retrainer(
        adapter,
        strategy="curriculum",
        min_baseline_accuracy=0.0,
        max_epochs=1,
        learning_rate=1e-3,
        checkpoint_dir=str(tmp_path),
    )

    failures = [
        AugmentedSample(item=X[i], label=int(y[i]), category="blur") for i in range(8)
    ]
    result = trainer.retrain(failures, (X, y.tolist()))
    # Successful retrain (min_baseline_accuracy=0) -> no checkpoint kept.
    assert result.rolled_back is False
    assert result.checkpoint_path is None
    assert os.listdir(tmp_path) == []


def test_pytorch_rollback_restores_weights(tmp_path):
    torch = pytest.importorskip("torch")
    from torch import nn

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

        def forward(self, x):
            return self.fc(x)

    # Manually pin weights so we can verify the rollback restores them exactly.
    torch.manual_seed(0)
    model = TinyNet()
    adapter = PyTorchAdapter(model)
    trainer = Retrainer(
        adapter,
        strategy="adaptive",
        min_baseline_accuracy=1.0,  # require perfection -> always rollback
        max_epochs=2,
        learning_rate=1e-1,
        checkpoint_dir=str(tmp_path),
    )

    # Use diverse baseline labels that an untrained 2-class linear layer
    # almost never matches perfectly, guaranteeing a sub-1.0 baseline_after.
    X = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [-1.0, -1.0]],
        dtype=np.float32,
    )
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

    failures = [
        AugmentedSample(item=X[i], label=int(y[i]), category="blur")
        for i in range(len(X))
    ]

    weights_before = model.fc.weight.detach().clone()
    bias_before = model.fc.bias.detach().clone()

    result = trainer.retrain(failures, (X, y.tolist()))

    weights_after = model.fc.weight.detach()
    bias_after = model.fc.bias.detach()

    assert result.rolled_back is True
    assert result.checkpoint_path is not None
    assert os.path.exists(result.checkpoint_path)
    # Rollback should restore weights bit-for-bit.
    torch.testing.assert_close(weights_before, weights_after)
    torch.testing.assert_close(bias_before, bias_after)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_strategy_rejected():
    adapter = SKLearnAdapter(_NoPartialFit())
    with pytest.raises(ValueError):
        Retrainer(adapter, strategy="random")


def test_empty_failure_cases_rejected(tmp_path):
    adapter = SKLearnAdapter(_NoPartialFit())
    trainer = Retrainer(adapter, checkpoint_dir=str(tmp_path))
    with pytest.raises(ValueError):
        trainer.retrain([], ([np.zeros(2)], [0]))
