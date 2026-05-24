"""Video augmentation generator.

Each item is a 4D NumPy array shaped ``(T, H, W, C)`` (or 3D ``(T, H, W)``
for grayscale) representing one video clip. Categories:

* ``frame_drop``           - randomly drop frames and pad-repeat the previous one
* ``compression_artifact`` - JPEG-encode each frame to add codec-style artifacts
* ``temporal_jitter``      - reorder frames within small windows
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import BaseGenerator
from .image_generator import ImageGenerator

_CATEGORIES = (
    "frame_drop",
    "compression_artifact",
    "temporal_jitter",
)


class VideoGenerator(BaseGenerator):
    """Generate corrupted video clips.

    Parameters
    ----------
    drop_rate
        Fraction of frames replaced by their predecessor for ``frame_drop``.
    jitter_window
        Window size used for ``temporal_jitter`` shuffling.
    """

    def __init__(
        self,
        random_state: int | None = None,
        drop_rate: float = 0.2,
        jitter_window: int = 4,
    ) -> None:
        super().__init__(random_state=random_state)
        if not 0.0 < drop_rate < 1.0:
            raise ValueError("drop_rate must be in (0, 1).")
        if jitter_window < 2:
            raise ValueError("jitter_window must be >= 2.")
        self.drop_rate = drop_rate
        self.jitter_window = jitter_window
        self._image_helper = ImageGenerator(random_state=random_state)

    def categories(self) -> Sequence[str]:
        return _CATEGORIES

    def _apply(self, category: str, item: Any) -> np.ndarray:
        clip = np.asarray(item)
        if clip.ndim not in (3, 4):
            raise ValueError(
                "VideoGenerator expects 3D (T,H,W) or 4D (T,H,W,C) arrays; "
                f"got shape {clip.shape}."
            )
        return getattr(self, f"_{category}")(clip)

    # --- categories ------------------------------------------------------

    def _frame_drop(self, clip: np.ndarray) -> np.ndarray:
        n_frames = clip.shape[0]
        if n_frames < 2:
            return clip
        n_drop = max(1, int(n_frames * self.drop_rate))
        drop_idx = self._rng.choice(n_frames - 1, size=n_drop, replace=False) + 1
        out = clip.copy()
        for i in drop_idx:
            out[i] = out[i - 1]
        return out

    def _compression_artifact(self, clip: np.ndarray) -> np.ndarray:
        transform = self._image_helper._jpeg_compression()
        out = np.empty_like(clip)
        for t in range(clip.shape[0]):
            out[t] = transform(image=clip[t])["image"]
        return out

    def _temporal_jitter(self, clip: np.ndarray) -> np.ndarray:
        n_frames = clip.shape[0]
        out = clip.copy()
        for start in range(0, n_frames, self.jitter_window):
            end = min(start + self.jitter_window, n_frames)
            window = list(range(start, end))
            shuffled = list(self._rng.permutation(window))
            out[start:end] = clip[shuffled]
        return out
