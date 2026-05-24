"""Tests for griit.tester."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from griit import AugmentedSample, StressTester, TestReport
from griit.adapters import ModelAdapter, SKLearnAdapter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class RecordingAdapter(ModelAdapter):
    """Adapter that records every batch it sees and emits scripted outputs."""

    framework = "fake"

    def __init__(
        self,
        predictions: List[int] | None = None,
        proba: np.ndarray | None = None,
    ) -> None:
        # Keep model attribute unused but valid.
        super().__init__(model=object())
        self._predictions = predictions
        self._proba = proba
        self._idx = 0
        self._proba_idx = 0
        self.batch_sizes: List[int] = []
        self.proba_calls: int = 0

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else int(np.asarray(X).shape[0])
        self.batch_sizes.append(n)
        if self._predictions is None:
            # Default: predict 0 for everything.
            return np.zeros(n, dtype=int)
        out = np.asarray(self._predictions[self._idx : self._idx + n])
        self._idx += n
        return out

    def predict_proba(self, X):
        n = len(X) if hasattr(X, "__len__") else int(np.asarray(X).shape[0])
        self.proba_calls += 1
        if self._proba is None:
            # Default: 90% confidence on class 0.
            out = np.tile([0.9, 0.1], (n, 1))
            return out
        out = self._proba[self._proba_idx : self._proba_idx + n]
        self._proba_idx += n
        return out


class NoProbaAdapter(ModelAdapter):
    framework = "fake"

    def __init__(self, predictions):
        super().__init__(model=object())
        self._predictions = list(predictions)
        self._idx = 0

    def predict(self, X):
        n = len(X) if hasattr(X, "__len__") else int(np.asarray(X).shape[0])
        out = np.asarray(self._predictions[self._idx : self._idx + n])
        self._idx += n
        return out

    def predict_proba(self, X):
        raise NotImplementedError("not supported")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _samples(items_per_category):
    """Build a flat list of AugmentedSample from {category: [(item, label), ...]}."""
    out = []
    for category, pairs in items_per_category.items():
        for item, label in pairs:
            out.append(AugmentedSample(item=item, label=label, category=category))
    return out


# ---------------------------------------------------------------------------
# Grouping + batching
# ---------------------------------------------------------------------------


def test_run_groups_by_category_and_counts():
    samples = _samples({
        "blur":  [(np.zeros((4, 4, 3), dtype=np.uint8), 0) for _ in range(5)],
        "noise": [(np.zeros((4, 4, 3), dtype=np.uint8), 0) for _ in range(3)],
    })
    # Predict 0 -> all correct; calibration uses default 0.9 confidence.
    adapter = RecordingAdapter()
    tester = StressTester(adapter)
    report = tester.run(samples)

    assert isinstance(report, TestReport)
    assert set(report.results_by_category.keys()) == {"blur", "noise"}
    assert report.results_by_category["blur"].n_tested == 5
    assert report.results_by_category["noise"].n_tested == 3
    assert report.results_by_category["blur"].n_failed == 0
    assert report.overall_failure_rate == 0.0
    assert report.n_total == 8
    assert report.n_failed == 0


def test_run_batches_at_configured_size():
    samples = _samples({
        "blur": [(np.zeros((4, 4, 3), dtype=np.uint8), 0) for _ in range(70)],
    })
    adapter = RecordingAdapter()
    tester = StressTester(adapter, batch_size=32)
    tester.run(samples)
    # 70 samples in groups of 32 -> 32, 32, 6.
    assert adapter.batch_sizes == [32, 32, 6]


# ---------------------------------------------------------------------------
# Failure rate
# ---------------------------------------------------------------------------


def test_failure_rates_per_category():
    # 4 blur samples, all labelled 0; predict 0,1,1,1 -> 3 failures.
    # 4 noise samples, labelled 0; predict 0,0,0,1 -> 1 failure.
    samples = _samples({
        "blur":  [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(4)],
        "noise": [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(4)],
    })
    adapter = RecordingAdapter(predictions=[0, 1, 1, 1, 0, 0, 0, 1])
    tester = StressTester(adapter)
    report = tester.run(samples)

    assert report.results_by_category["blur"].failure_rate == pytest.approx(0.75)
    assert report.results_by_category["noise"].failure_rate == pytest.approx(0.25)
    assert report.overall_failure_rate == pytest.approx(0.5)
    assert report.top_failure_categories[0] == "blur"
    # Failed samples retained:
    assert len(report.results_by_category["blur"].failed_samples) == 3
    assert len(report.results_by_category["noise"].failed_samples) == 1


# ---------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------


def test_robustness_score_matches_formula():
    samples = _samples({
        "blur":  [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(4)],
        "noise": [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(4)],
    })
    # blur: 3/4 fail. noise: 1/4 fail. Overall 4/8 = 0.5.
    adapter = RecordingAdapter(predictions=[0, 1, 1, 1, 0, 0, 0, 1])
    tester = StressTester(adapter)
    report = tester.run(samples)

    expected = 100.0 * (
        report.baseline_accuracy * 0.30
        + (1.0 - report.overall_failure_rate) * 0.40
        + report.calibration_score * 0.15
        + (1.0 - report.results_by_category["blur"].failure_rate) * 0.15
    )
    assert report.robustness_score == pytest.approx(expected)


def test_baseline_accuracy_used_when_provided():
    samples = _samples({
        "blur": [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(4)],
    })
    # Baseline: predict 0 for all 4 inputs, but only 2 of the labels are 0.
    adapter = RecordingAdapter()
    X_baseline = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)]
    y_baseline = [0, 0, 1, 1]
    tester = StressTester(adapter, baseline_data=(X_baseline, y_baseline))
    report = tester.run(samples)
    assert report.baseline_accuracy == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibration_score_default_when_predict_proba_missing():
    samples = _samples({
        "blur": [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(3)],
    })
    adapter = NoProbaAdapter(predictions=[0, 0, 0])
    tester = StressTester(adapter)
    report = tester.run(samples)
    assert report.calibration_score == 0.5


def test_calibration_score_perfect_when_confidence_matches_correctness():
    # 2 correct (label=0, predicted=0) at 1.0 confidence
    # 2 wrong   (label=0, predicted=1) at 0.0 confidence
    # |max_proba - correctness| = 0 everywhere -> calibration = 1.0
    samples = _samples({
        "blur": [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(4)],
    })
    proba = np.array(
        [
            [1.0, 0.0],  # pred 0, correct, max=1.0
            [1.0, 0.0],  # pred 0, correct, max=1.0
            [1.0, 0.0],  # pred 0... but we'll override predictions below
            [1.0, 0.0],
        ]
    )
    # Force predictions: first two correct, last two wrong.
    adapter = RecordingAdapter(
        predictions=[0, 0, 1, 1],
        proba=np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    )
    # max([0.0, 0.0]) = 0.0 -> abs(0 - 0) = 0; calibration becomes 1.0.
    tester = StressTester(adapter)
    report = tester.run(samples)
    assert report.calibration_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_format(capsys):
    samples = _samples({
        "blur":  [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(4)],
        "noise": [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(4)],
    })
    adapter = RecordingAdapter(predictions=[0, 1, 1, 1, 0, 0, 0, 1])
    tester = StressTester(adapter)
    tester.run(samples)
    text = tester.summary()
    captured = capsys.readouterr().out

    assert text == captured.rstrip("\n")
    assert "GRIIT Stress Test Summary" in text
    assert "Failed on 4/8 cases" in text
    assert "Robustness score" in text
    assert "Top failure categories" in text
    assert "blur" in text
    # blur (75%) ranked above noise (25%):
    assert text.index("blur") < text.index("noise")


def test_summary_without_run_raises():
    adapter = RecordingAdapter()
    with pytest.raises(RuntimeError):
        StressTester(adapter).summary()


# ---------------------------------------------------------------------------
# Wrapping a non-adapter model
# ---------------------------------------------------------------------------


def test_stress_tester_accepts_raw_model():
    class Bare:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

        def predict_proba(self, X):
            return np.tile([0.9, 0.1], (len(X), 1))

    samples = _samples({
        "blur": [(np.zeros((2, 2, 3), dtype=np.uint8), 0) for _ in range(2)],
    })
    tester = StressTester(Bare())
    assert isinstance(tester.adapter, SKLearnAdapter)
    report = tester.run(samples)
    assert report.overall_failure_rate == 0.0


# ---------------------------------------------------------------------------
# End-to-end with the image generator
# ---------------------------------------------------------------------------


def test_end_to_end_with_image_generator():
    from griit.generators import ImageGenerator

    images = [np.full((8, 8, 3), 128, dtype=np.uint8) for _ in range(3)]
    labels = [0, 0, 0]
    gen = ImageGenerator(random_state=0, severity="light")
    samples = gen.generate(images, labels, n_per_category=2)

    adapter = RecordingAdapter()  # predicts 0 -> all correct
    tester = StressTester(adapter, batch_size=4)
    report = tester.run(samples)

    assert report.n_total == 2 * len(gen.categories())
    assert report.overall_failure_rate == 0.0
    assert report.robustness_score > 90
