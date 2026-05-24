"""Image augmentation generator built on top of albumentations.

Each augmentation category is exposed as a private method that returns an
albumentations ``Compose`` (or a single transform) with ``p=1.0`` so the
perturbation always fires when sampled. The categories cover the common
robustness failure modes:

* ``low_light``         - brightness / contrast drops
* ``blur``              - motion + Gaussian blur
* ``noise``             - Gaussian noise
* ``fog``               - fog / rain weather simulation
* ``rotation``          - random rotation
* ``occlusion``         - coarse dropout patches
* ``jpeg_compression``  - lossy JPEG re-encoding artefacts
* ``color_shift``       - hue / saturation / RGB shifts
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Sequence

import numpy as np

from .base import BaseGenerator

_CATEGORIES = (
    "low_light",
    "blur",
    "noise",
    "fog",
    "rotation",
    "occlusion",
    "jpeg_compression",
    "color_shift",
)


class ImageGenerator(BaseGenerator):
    """Generate perturbed images for robustness evaluation.

    Parameters
    ----------
    random_state
        Seed for sample selection. Albumentations transforms use their own
        global RNG; pass ``random_state`` to reproduce the *sampling* of
        which input image gets perturbed for which category.
    severity
        ``"light"``, ``"medium"`` (default) or ``"heavy"`` - scales the
        intensity of every augmentation.
    """

    def __init__(
        self,
        random_state: int | None = None,
        severity: str = "medium",
    ) -> None:
        super().__init__(random_state=random_state)
        if severity not in {"light", "medium", "heavy"}:
            raise ValueError("severity must be one of 'light', 'medium', 'heavy'.")
        self.severity = severity
        self._transforms: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    # --- BaseGenerator hooks ---------------------------------------------

    def categories(self) -> Sequence[str]:
        return _CATEGORIES

    def _apply(self, category: str, item: Any) -> np.ndarray:
        image = self._coerce_image(item)
        transform = self._get_transform(category)
        return transform(image=image)["image"]

    # --- transform registry ----------------------------------------------

    def _get_transform(self, category: str):
        """Lazily build (and cache) the albumentations transform per category."""

        if category not in self._transforms:
            builder = getattr(self, f"_{category}")
            self._transforms[category] = builder()
        return self._transforms[category]

    # --- private builders, one per category ------------------------------

    def _low_light(self):
        import albumentations as A

        ranges = {
            "light":  (-0.3, -0.1),
            "medium": (-0.5, -0.2),
            "heavy":  (-0.8, -0.4),
        }[self.severity]
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=ranges,
                contrast_limit=(-0.3, 0.0),
                p=1.0,
            ),
        ])

    def _blur(self):
        import albumentations as A

        kernel = {"light": 5, "medium": 9, "heavy": 15}[self.severity]
        return A.Compose([
            A.OneOf([
                A.MotionBlur(blur_limit=kernel, p=1.0),
                A.GaussianBlur(blur_limit=(3, kernel), p=1.0),
            ], p=1.0),
        ])

    def _noise(self):
        import albumentations as A

        std = {
            "light":  (0.05, 0.10),
            "medium": (0.10, 0.20),
            "heavy":  (0.20, 0.35),
        }[self.severity]
        return A.Compose([
            A.GaussNoise(std_range=std, mean_range=(0.0, 0.0), p=1.0),
        ])

    def _fog(self):
        import albumentations as A

        coef = {
            "light":  (0.1, 0.3),
            "medium": (0.3, 0.6),
            "heavy":  (0.5, 0.9),
        }[self.severity]
        return A.Compose([
            A.OneOf([
                A.RandomFog(fog_coef_range=coef, alpha_coef=0.1, p=1.0),
                A.RandomRain(blur_value=3, brightness_coefficient=0.9, p=1.0),
            ], p=1.0),
        ])

    def _rotation(self):
        import albumentations as A

        limit = {"light": 10, "medium": 25, "heavy": 45}[self.severity]
        return A.Compose([
            A.Rotate(limit=limit, border_mode=0, p=1.0),
        ])

    def _occlusion(self):
        import albumentations as A

        holes = {"light": (1, 2), "medium": (2, 4), "heavy": (3, 6)}[self.severity]
        size = {"light": (8, 16), "medium": (16, 32), "heavy": (24, 48)}[self.severity]
        return A.Compose([
            A.CoarseDropout(
                num_holes_range=holes,
                hole_height_range=size,
                hole_width_range=size,
                fill=0,
                p=1.0,
            ),
        ])

    def _jpeg_compression(self):
        import albumentations as A

        quality = {
            "light":  (60, 85),
            "medium": (30, 60),
            "heavy":  (10, 30),
        }[self.severity]
        return A.Compose([
            A.ImageCompression(quality_range=quality, compression_type="jpeg", p=1.0),
        ])

    def _color_shift(self):
        import albumentations as A

        hue = {"light": 10, "medium": 20, "heavy": 35}[self.severity]
        sat = {"light": 15, "medium": 30, "heavy": 50}[self.severity]
        rgb = {"light": 10, "medium": 20, "heavy": 35}[self.severity]
        return A.Compose([
            A.OneOf([
                A.HueSaturationValue(
                    hue_shift_limit=hue,
                    sat_shift_limit=sat,
                    val_shift_limit=10,
                    p=1.0,
                ),
                A.RGBShift(
                    r_shift_limit=rgb,
                    g_shift_limit=rgb,
                    b_shift_limit=rgb,
                    p=1.0,
                ),
            ], p=1.0),
        ])

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _coerce_image(item: Any) -> np.ndarray:
        """Albumentations expects an ``HxW`` or ``HxWxC`` uint8/float NumPy array."""
        arr = np.asarray(item)
        if arr.ndim == 2 or arr.ndim == 3:
            return arr
        raise ValueError(
            "ImageGenerator expects 2D (grayscale) or 3D (HxWxC) arrays; "
            f"got shape {arr.shape}."
        )
