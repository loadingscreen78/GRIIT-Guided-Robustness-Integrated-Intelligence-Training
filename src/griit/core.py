"""Core GRIIT object that wraps any supported model."""

from __future__ import annotations

from typing import Any

import numpy as np

from .adapters import (
    KerasAdapter,
    ModelAdapter,
    PyTorchAdapter,
    SKLearnAdapter,
    wrap_model,
)

__all__ = ["Griit"]


class Griit:
    """Framework-agnostic handle around a trained model.

    Parameters
    ----------
    model
        A PyTorch ``nn.Module``, scikit-learn estimator, Keras / tf.keras
        ``Model``, or an object that already implements
        :class:`griit.adapters.ModelAdapter`.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from griit import Griit
    >>> g = Griit(LogisticRegression())            # doctest: +SKIP
    >>> g.framework                                # doctest: +SKIP
    'sklearn'
    """

    def __init__(self, model: Any) -> None:
        self.adapter: ModelAdapter = self._wrap_model(model)

    # --- public API -------------------------------------------------------

    @property
    def model(self) -> Any:
        """The underlying framework-native model object."""
        return self.adapter.model

    @property
    def framework(self) -> str:
        """Short identifier of the detected framework."""
        return self.adapter.framework

    def predict(self, X: Any) -> np.ndarray:
        """Hard predictions for ``X``."""
        return self.adapter.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Class probabilities for ``X`` as ``(n_samples, n_classes)``."""
        return self.adapter.predict_proba(X)

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"Griit(framework={self.framework!r}, "
            f"model={type(self.model).__name__})"
        )

    # --- internals --------------------------------------------------------

    @staticmethod
    def _wrap_model(model: Any) -> ModelAdapter:
        """Auto-detect ``model``'s framework and return the matching adapter.

        Detection priority:

        1. If ``model`` is already a :class:`ModelAdapter`, use it as-is.
        2. ``torch.nn.Module``  -> :class:`PyTorchAdapter`
           (``hasattr(model, "forward")`` as fallback).
        3. ``keras.Model`` / ``tf.keras.Model``  -> :class:`KerasAdapter`
           (``hasattr(model, "call")`` as fallback). Checked before sklearn
           because Keras models also expose ``predict``.
        4. ``sklearn.base.BaseEstimator``  -> :class:`SKLearnAdapter`
           (``hasattr(model, "predict")`` as fallback).
        5. Otherwise raise :class:`ValueError`.
        """

        if isinstance(model, ModelAdapter):
            return model
        return wrap_model(model)


# Re-export adapters so callers can do `from griit.core import PyTorchAdapter`.
__all__ += ["ModelAdapter", "PyTorchAdapter", "SKLearnAdapter", "KerasAdapter"]
