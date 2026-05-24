"""Augmented-sample generators for GRIIT.

Each generator takes a small set of clean inputs and produces a much larger
set of perturbed (item, label, category) samples that exercise different
robustness failure modes.
"""

from __future__ import annotations

from .base import AugmentedSample, BaseGenerator
from .image_generator import ImageGenerator
from .tabular_generator import TabularGenerator
from .text_generator import TextGenerator
from .video_generator import VideoGenerator

__all__ = [
    "AugmentedSample",
    "BaseGenerator",
    "ImageGenerator",
    "TabularGenerator",
    "TextGenerator",
    "VideoGenerator",
]
