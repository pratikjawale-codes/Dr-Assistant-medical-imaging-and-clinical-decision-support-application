"""
Medical Image Window/Level (Contrast/Brightness) Engine
Implements standard DICOM VOI (Value of Interest) LUT transformations.
"""

import base64
import io
from typing import Dict, Tuple
import numpy as np
from PIL import Image
from ..config import WINDOW_PRESETS


class WindowLevelEngine:
    """Applies clinical windowing to pixel arrays and encodes to web formats."""

    @staticmethod
    def apply_window_level(
        pixel_array: np.ndarray,
        center: float,
        width: float,
        invert: bool = False
    ) -> np.ndarray:
        """
        Maps pixel values using linear DICOM VOI windowing:
        Lower bound = center - 0.5 - (width - 1) / 2
        Upper bound = center - 0.5 + (width - 1) / 2
        """
        img = pixel_array.astype(np.float32)
        c = float(center)
        w = float(max(width, 1.0))

        if w <= 1.0:
            output = np.where(img <= c - 0.5, 0, 255).astype(np.uint8)
            return 255 - output if invert else output

        lower = c - 0.5 - (w - 1.0) / 2.0
        upper = c - 0.5 + (w - 1.0) / 2.0

        # Linear ramp clamp
        clamped = np.clip(img, lower, upper)
        normalized = ((clamped - lower) / (upper - lower)) * 255.0
        output = normalized.astype(np.uint8)

        if invert:
            output = 255 - output

        return output

    @classmethod
    def apply_preset(
        cls,
        pixel_array: np.ndarray,
        preset_name: str,
        invert: bool = False
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Applies named medical preset (lung, bone, soft_tissue, brain, default)."""
        preset = WINDOW_PRESETS.get(preset_name.lower(), WINDOW_PRESETS["default"])
        windowed = cls.apply_window_level(pixel_array, preset["center"], preset["width"], invert=invert)
        return windowed, preset

    @staticmethod
    def array_to_base64_png(image_array: np.ndarray) -> str:
        """Converts an 8-bit 2D or 3D numpy array to a data URL PNG string."""
        if len(image_array.shape) == 2:
            img = Image.fromarray(image_array, mode="L")
        elif len(image_array.shape) == 3 and image_array.shape[2] == 3:
            img = Image.fromarray(image_array, mode="RGB")
        elif len(image_array.shape) == 3 and image_array.shape[2] == 4:
            img = Image.fromarray(image_array, mode="RGBA")
        else:
            img = Image.fromarray(image_array.astype(np.uint8))

        buffered = io.BytesIO()
        img.save(buffered, format="PNG", optimize=True)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
