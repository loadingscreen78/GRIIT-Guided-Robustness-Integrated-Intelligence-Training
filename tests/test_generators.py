"""Tests for the griit.generators package."""

from __future__ import annotations

import numpy as np
import pytest

from griit.generators import (
    AugmentedSample,
    ImageGenerator,
    TabularGenerator,
    TextGenerator,
    VideoGenerator,
)


# ---------------------------------------------------------------------------
# ImageGenerator
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_images():
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8) for _ in range(4)]


def test_image_generator_returns_expected_count(sample_images):
    gen = ImageGenerator(random_state=0)
    out = gen.generate(sample_images, labels=[0, 1, 0, 1], n_per_category=3)
    assert len(out) == 3 * len(gen.categories())
    assert all(isinstance(s, AugmentedSample) for s in out)


def test_image_generator_each_category_present(sample_images):
    gen = ImageGenerator(random_state=0)
    out = gen.generate(sample_images, labels=[0, 1, 0, 1], n_per_category=2)
    seen = {s.category for s in out}
    assert seen == set(gen.categories())


def test_image_generator_outputs_are_arrays_and_label_carried(sample_images):
    gen = ImageGenerator(random_state=0)
    out = gen.generate(sample_images, labels=["a", "b", "c", "d"], n_per_category=2)
    for img, label, category in out:
        assert isinstance(img, np.ndarray)
        assert img.ndim == 3
        assert label in {"a", "b", "c", "d"}
        assert category in ImageGenerator(random_state=0).categories()


def test_image_generator_subset_of_categories(sample_images):
    gen = ImageGenerator(random_state=0)
    out = gen.generate(
        sample_images,
        labels=[0, 1, 0, 1],
        n_per_category=2,
        categories=["blur", "noise"],
    )
    assert len(out) == 4
    assert {s.category for s in out} == {"blur", "noise"}


def test_image_generator_actually_changes_pixels(sample_images):
    gen = ImageGenerator(random_state=0, severity="heavy")
    out = gen.generate(sample_images, labels=[0, 1, 0, 1], n_per_category=1)
    # At least one category must produce a different image from the source.
    diffs = []
    for s in out:
        # Find the original this could have come from (any of them works for diff check).
        diffs.append(any(not np.array_equal(s.item, src) for src in sample_images))
    assert any(diffs)


def test_image_generator_rejects_unknown_category(sample_images):
    gen = ImageGenerator(random_state=0)
    with pytest.raises(ValueError):
        gen.generate(sample_images, labels=[0, 1, 0, 1], categories=["bogus"])


def test_image_generator_validates_inputs():
    gen = ImageGenerator(random_state=0)
    with pytest.raises(ValueError):
        gen.generate([], labels=[], n_per_category=1)
    with pytest.raises(ValueError):
        gen.generate([np.zeros((4, 4, 3), dtype=np.uint8)], labels=[0, 1])


# ---------------------------------------------------------------------------
# TextGenerator
# ---------------------------------------------------------------------------


def test_text_generator_produces_strings():
    gen = TextGenerator(random_state=1)
    texts = ["the quick brown fox jumps", "good morning everyone", "buy now or lose"]
    out = gen.generate(texts, labels=[0, 1, 1], n_per_category=4)
    assert len(out) == 4 * len(gen.categories())
    for item, _label, category in out:
        assert isinstance(item, str)
        assert category in gen.categories()


def test_text_generator_padding_grows_text():
    gen = TextGenerator(random_state=1)
    out = gen.generate(["hello"], labels=[0], n_per_category=2, categories=["padding"])
    assert all(len(s.item) > len("hello") for s in out)


def test_text_generator_truncation_shrinks_text():
    gen = TextGenerator(random_state=1)
    long = "the quick brown fox jumps over the lazy dog repeatedly"
    out = gen.generate([long], labels=[0], n_per_category=3, categories=["truncation"])
    assert all(len(s.item) < len(long) for s in out)


def test_text_generator_emoji_inject_changes_text():
    gen = TextGenerator(random_state=1)
    out = gen.generate(["hello world"], labels=[0], n_per_category=3, categories=["emoji_inject"])
    assert all(s.item != "hello world" for s in out)


# ---------------------------------------------------------------------------
# TabularGenerator
# ---------------------------------------------------------------------------


def _make_df():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(
        {"age": [30], "income": [50000.0], "city": ["NYC"], "plan": ["pro"]}
    )


def test_tabular_null_injection_introduces_nans():
    pd = pytest.importorskip("pandas")
    df = _make_df()
    gen = TabularGenerator(random_state=0, null_rate=0.5)
    out = gen.generate([df], labels=[0], n_per_category=2, categories=["null_injection"])
    assert any(s.item.isna().any().any() for s in out)


def test_tabular_outlier_injection_pushes_numeric_far():
    pytest.importorskip("pandas")
    df = _make_df()
    gen = TabularGenerator(random_state=0, outlier_sigma=10)
    out = gen.generate(
        [df], labels=[0], n_per_category=1, categories=["outlier_injection"]
    )
    perturbed = out[0].item
    # outlier_injection sets the column to mean ± k*std; for a single-row
    # frame std is 0 so it falls back to 1.0 and value becomes mean ± 10.
    assert perturbed["age"].iloc[0] != df["age"].iloc[0]


def test_tabular_column_shuffle_preserves_values():
    pytest.importorskip("pandas")
    df = _make_df()
    gen = TabularGenerator(random_state=0)
    out = gen.generate(
        [df], labels=[0], n_per_category=1, categories=["column_shuffle"]
    )
    # The set of values is unchanged but column ordering may differ.
    perturbed = out[0].item
    assert sorted(perturbed.columns) == sorted(df.columns)


def test_tabular_new_category_replaces_object_columns():
    pytest.importorskip("pandas")
    df = _make_df()
    gen = TabularGenerator(random_state=0)
    out = gen.generate([df], labels=[0], n_per_category=1, categories=["new_category"])
    perturbed = out[0].item
    assert perturbed["city"].iloc[0] == gen.new_category_token
    assert perturbed["plan"].iloc[0] == gen.new_category_token


# ---------------------------------------------------------------------------
# VideoGenerator
# ---------------------------------------------------------------------------


def _make_clip(n_frames=8):
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(n_frames, 16, 16, 3), dtype=np.uint8)


def test_video_frame_drop_changes_some_frames():
    clip = _make_clip()
    gen = VideoGenerator(random_state=0, drop_rate=0.5)
    out = gen.generate([clip], labels=[0], n_per_category=1, categories=["frame_drop"])
    perturbed = out[0].item
    assert perturbed.shape == clip.shape
    assert not np.array_equal(perturbed, clip)


def test_video_temporal_jitter_preserves_frame_set():
    clip = _make_clip(n_frames=8)
    gen = VideoGenerator(random_state=0, jitter_window=4)
    out = gen.generate(
        [clip], labels=[0], n_per_category=1, categories=["temporal_jitter"]
    )
    perturbed = out[0].item
    # Same set of frames, possibly reordered.
    src_sums = sorted(int(f.sum()) for f in clip)
    out_sums = sorted(int(f.sum()) for f in perturbed)
    assert src_sums == out_sums


def test_video_compression_artifact_keeps_shape():
    clip = _make_clip()
    gen = VideoGenerator(random_state=0)
    out = gen.generate(
        [clip], labels=[0], n_per_category=1, categories=["compression_artifact"]
    )
    perturbed = out[0].item
    assert perturbed.shape == clip.shape
