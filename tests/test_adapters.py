"""Tests for adapter detection and behavior."""

from __future__ import annotations

import numpy as np
import pytest

from griit import Griit
from griit.adapters import (
    KerasAdapter,
    ModelAdapter,
    PyTorchAdapter,
    SKLearnAdapter,
    wrap_model,
)


# ---------------------------------------------------------------------------
# Fakes that mimic each framework's duck-typed surface.
# ---------------------------------------------------------------------------


class FakeSklearnModel:
    """Looks like a sklearn classifier (predict + predict_proba)."""

    def predict(self, X):
        X = np.asarray(X)
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        X = np.asarray(X)
        probs = np.zeros((len(X), 3))
        probs[:, 0] = 1.0
        return probs


class FakeSklearnDecisionFn:
    """Sklearn-like model exposing only decision_function (binary)."""

    def predict(self, X):
        X = np.asarray(X)
        return (self.decision_function(X) > 0).astype(int)

    def decision_function(self, X):
        X = np.asarray(X)
        return np.linspace(-1.0, 1.0, num=len(X))


class FakeTorchModule:
    """Mimics torch.nn.Module enough for duck-typed detection."""

    def __init__(self):
        self.training = False

    # required for the duck-typed branch
    def forward(self, x):  # pragma: no cover - unused by adapter
        return x

    def __call__(self, x):
        # Pretend to return logits for 3 classes.
        import torch

        batch = x.shape[0]
        logits = torch.zeros((batch, 3))
        logits[:, 1] = 5.0  # class 1 wins
        return logits

    def eval(self):
        self.training = False
        return self

    def train(self, mode: bool = True):
        self.training = mode
        return self


class FakeKerasModel:
    """Mimics a Keras model: has both `call` and `predict`."""

    def call(self, x):  # pragma: no cover - unused by adapter
        return x

    def predict(self, X, verbose=0):
        X = np.asarray(X)
        out = np.zeros((len(X), 4))
        out[:, 2] = 1.0  # class 2 wins
        return out


# ---------------------------------------------------------------------------
# Detection routing
# ---------------------------------------------------------------------------


def test_wrap_sklearn_model_returns_sklearn_adapter():
    adapter = wrap_model(FakeSklearnModel())
    assert isinstance(adapter, SKLearnAdapter)
    assert adapter.framework == "sklearn"


def test_wrap_keras_model_picks_keras_over_sklearn():
    # Keras models have .predict too; ensure we don't misroute to sklearn.
    adapter = wrap_model(FakeKerasModel())
    assert isinstance(adapter, KerasAdapter)


def test_wrap_torch_module_picks_pytorch():
    pytest.importorskip("torch")
    adapter = wrap_model(FakeTorchModule())
    assert isinstance(adapter, PyTorchAdapter)


def test_wrap_unknown_model_raises():
    class Mystery:
        pass

    with pytest.raises(ValueError):
        wrap_model(Mystery())


def test_wrap_existing_adapter_is_passthrough():
    inner = wrap_model(FakeSklearnModel())
    assert wrap_model(inner) is inner


# ---------------------------------------------------------------------------
# Behavior of each concrete adapter
# ---------------------------------------------------------------------------


def test_sklearn_adapter_predict_and_proba():
    adapter = SKLearnAdapter(FakeSklearnModel())
    X = np.zeros((5, 2))
    assert adapter.predict(X).tolist() == [0] * 5
    proba = adapter.predict_proba(X)
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(5))


def test_sklearn_adapter_decision_function_fallback_binary():
    adapter = SKLearnAdapter(FakeSklearnDecisionFn())
    X = np.zeros((4, 1))
    proba = adapter.predict_proba(X)
    assert proba.shape == (4, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(4))


def test_sklearn_adapter_raises_without_proba_or_decision():
    class Bare:
        def predict(self, X):
            return np.zeros(len(X))

    adapter = SKLearnAdapter(Bare())
    with pytest.raises(NotImplementedError):
        adapter.predict_proba(np.zeros((2, 2)))


def test_keras_adapter_argmax_predict():
    adapter = KerasAdapter(FakeKerasModel())
    X = np.zeros((6, 3))
    proba = adapter.predict_proba(X)
    assert proba.shape == (6, 4)
    np.testing.assert_array_equal(adapter.predict(X), np.full(6, 2))


def test_keras_adapter_binary_single_column_output():
    class BinaryKeras:
        def call(self, x):  # pragma: no cover
            return x

        def predict(self, X, verbose=0):
            X = np.asarray(X)
            return np.full((len(X), 1), 0.8)

    adapter = KerasAdapter(BinaryKeras())
    proba = adapter.predict_proba(np.zeros((3, 2)))
    assert proba.shape == (3, 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(3))
    np.testing.assert_allclose(proba[:, 1], 0.8)


def test_pytorch_adapter_predict_uses_argmax():
    pytest.importorskip("torch")
    adapter = PyTorchAdapter(FakeTorchModule())
    X = np.zeros((7, 2), dtype=np.float32)
    proba = adapter.predict_proba(X)
    assert proba.shape == (7, 3)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(7), rtol=1e-5)
    np.testing.assert_array_equal(adapter.predict(X), np.full(7, 1))


# ---------------------------------------------------------------------------
# Griit facade
# ---------------------------------------------------------------------------


def test_griit_wraps_and_delegates():
    g = Griit(FakeSklearnModel())
    assert g.framework == "sklearn"
    assert isinstance(g.adapter, ModelAdapter)
    X = np.zeros((3, 2))
    assert g.predict(X).shape == (3,)
    assert g.predict_proba(X).shape == (3, 3)


def test_griit_accepts_existing_adapter():
    inner = SKLearnAdapter(FakeSklearnModel())
    g = Griit(inner)
    assert g.adapter is inner


def test_griit_rejects_unknown_model():
    with pytest.raises(ValueError):
        Griit(object())
