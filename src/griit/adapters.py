"""Model adapters that give GRIIT a uniform interface across frameworks.

GRIIT needs to work with any trained model. Rather than special-casing
PyTorch / scikit-learn / Keras everywhere, we wrap the underlying model in a
:class:`ModelAdapter` that exposes a small, framework-agnostic API:

* :meth:`ModelAdapter.predict` returns hard predictions as a NumPy array.
* :meth:`ModelAdapter.predict_proba` returns class probabilities as a NumPy
  array of shape ``(n_samples, n_classes)``.

Use :func:`wrap_model` (or :class:`griit.core.Griit`) to auto-detect the
framework and pick the right adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

__all__ = [
    "ModelAdapter",
    "PyTorchAdapter",
    "SKLearnAdapter",
    "KerasAdapter",
    "wrap_model",
]


class ModelAdapter(ABC):
    """Abstract framework-agnostic wrapper around a trained model."""

    #: Short framework identifier, overridden by subclasses.
    framework: str = "unknown"

    def __init__(self, model: Any) -> None:
        self.model = model

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Return hard predictions for ``X`` as a NumPy array."""

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Return class probabilities for ``X`` as a NumPy array."""

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"{type(self).__name__}(framework={self.framework!r}, "
            f"model={type(self.model).__name__})"
        )


# ---------------------------------------------------------------------------
# scikit-learn
# ---------------------------------------------------------------------------


class SKLearnAdapter(ModelAdapter):
    """Adapter for scikit-learn estimators (anything implementing ``predict``)."""

    framework = "sklearn"

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        if hasattr(self.model, "decision_function"):
            scores = np.asarray(self.model.decision_function(X))
            if scores.ndim == 1:
                # Binary classifier: turn margin into a 2-column probability.
                probs = _sigmoid(scores).reshape(-1, 1)
                return np.hstack([1.0 - probs, probs])
            return _softmax(scores)
        raise NotImplementedError(
            f"{type(self.model).__name__} does not expose predict_proba "
            "or decision_function; cannot produce class probabilities."
        )


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------


class PyTorchAdapter(ModelAdapter):
    """Adapter for ``torch.nn.Module`` instances.

    Inputs may be NumPy arrays, lists, or ``torch.Tensor`` objects. The model
    is switched to ``eval()`` mode and inference runs under ``torch.no_grad``.
    """

    framework = "pytorch"

    def __init__(self, model: Any, device: Optional[str] = None) -> None:
        super().__init__(model)
        self.device = device

    def _forward(self, X: Any) -> np.ndarray:
        import torch  # local import keeps torch optional at install time

        was_training = bool(getattr(self.model, "training", False))
        self.model.eval()
        try:
            if isinstance(X, torch.Tensor):
                tensor = X
            else:
                tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
            if self.device is not None:
                tensor = tensor.to(self.device)
            with torch.no_grad():
                out = self.model(tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out.detach().cpu().numpy()
        finally:
            if was_training:
                self.model.train()

    def predict_proba(self, X: Any) -> np.ndarray:
        logits = self._forward(X)
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            # Single-logit binary output -> sigmoid into a 2-column matrix.
            probs = _sigmoid(logits.reshape(-1, 1))
            return np.hstack([1.0 - probs, probs])
        return _softmax(logits)

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1)


# ---------------------------------------------------------------------------
# Keras / TensorFlow
# ---------------------------------------------------------------------------


class KerasAdapter(ModelAdapter):
    """Adapter for ``keras.Model`` / ``tf.keras.Model`` instances."""

    framework = "keras"

    def predict_proba(self, X: Any) -> np.ndarray:
        out = np.asarray(self.model.predict(X, verbose=0))
        if out.ndim == 1 or (out.ndim == 2 and out.shape[1] == 1):
            probs = out.reshape(-1, 1)
            return np.hstack([1.0 - probs, probs])
        return out

    def predict(self, X: Any) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=-1)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _is_pytorch_module(model: Any) -> bool:
    try:
        import torch.nn as nn
    except ImportError:
        return False
    return isinstance(model, nn.Module)


def _is_sklearn_estimator(model: Any) -> bool:
    try:
        from sklearn.base import BaseEstimator
    except ImportError:
        return False
    return isinstance(model, BaseEstimator)


def _is_keras_model(model: Any) -> bool:
    try:
        import keras  # standalone Keras 3+

        if isinstance(model, keras.Model):
            return True
    except ImportError:
        pass
    try:
        from tensorflow import keras as tf_keras  # type: ignore

        if isinstance(model, tf_keras.Model):
            return True
    except ImportError:
        pass
    return False


def wrap_model(model: Any) -> ModelAdapter:
    """Detect the framework of ``model`` and wrap it in the matching adapter.

    Detection order is PyTorch -> Keras -> scikit-learn. Keras is checked
    before scikit-learn because ``keras.Model`` also exposes ``predict``,
    which would otherwise route it to :class:`SKLearnAdapter`.
    """

    if isinstance(model, ModelAdapter):
        return model

    # 1) Strong isinstance checks when the framework is importable.
    if _is_pytorch_module(model):
        return PyTorchAdapter(model)
    if _is_keras_model(model):
        return KerasAdapter(model)
    if _is_sklearn_estimator(model):
        return SKLearnAdapter(model)

    # 2) Duck-typed fallback for objects whose framework is not importable
    #    (e.g. mocks, custom wrappers). Keep the same priority order.
    if hasattr(model, "forward") and callable(getattr(model, "forward")):
        return PyTorchAdapter(model)
    if (
        hasattr(model, "call")
        and callable(getattr(model, "call"))
        and hasattr(model, "predict")
    ):
        return KerasAdapter(model)
    if hasattr(model, "predict") and callable(getattr(model, "predict")):
        return SKLearnAdapter(model)

    raise ValueError(
        f"Unsupported model type: {type(model).__name__!r}. "
        "Expected a PyTorch nn.Module, a scikit-learn estimator, "
        "or a Keras / tf.keras Model."
    )


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _softmax(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    shifted = arr - np.max(arr, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-arr))
