# -*- coding: utf-8 -*-
"""
KrakenPresetFromImage: produce a resolution string (e.g., "3456x2304") from an input IMAGE.
Place this file as: custom_nodes/kraken_tools/kraken_preset_from_image.py
Register it from __init__.py with:
    _safe_register("kraken_preset_from_image", "KrakenPresetFromImage", "🦑 Kraken Preset From Image")
"""
from typing import Tuple

class KrakenPresetFromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (["factor", "long_side", "short_side", "megapixels"], {"default": "factor"}),
                "factor": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.05}),
                "target_long_side": ("INT", {"default": 2048, "min": 64, "max": 65536}),
                "target_short_side": ("INT", {"default": 1024, "min": 64, "max": 65536}),
                "target_megapixels": ("FLOAT", {"default": 4.0, "min": 0.1, "max": 4096.0, "step": 0.1}),
                "align_multiple": ("INT", {"default": 8, "min": 1, "max": 256}),
                "max_width": ("INT", {"default": 0, "min": 0, "max": 65536}),
                "max_height": ("INT", {"default": 0, "min": 0, "max": 65536}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("preset_out",)
    FUNCTION = "make"
    CATEGORY = "🦑 Kraken Tools/Upscaling"

    # --- helpers ---
    @staticmethod
    def _shape(image) -> Tuple[int, int]:
        """
        Return (W, H) for the first image in the batch.
        Handles numpy arrays and torch tensors with shape [B,H,W,C] or [H,W,C].
        """
        # Try numpy-like
        shape = getattr(image, "shape", None)
        if shape is None:
            raise ValueError("Unsupported image type: no shape attribute")
        if len(shape) == 4:
            _, H, W, _ = shape
        elif len(shape) == 3:
            H, W, _ = shape
        else:
            raise ValueError(f"Unsupported IMAGE shape {shape}. Expected [B,H,W,C] or [H,W,C].")
        return int(W), int(H)

    @staticmethod
    def _ceil_to_multiple(x: int, m: int) -> int:
        if m <= 1:
            return max(1, int(x))
        return int(((x + m - 1) // m) * m)

    def make(self, image, mode="factor", factor=2.0, target_long_side=2048,
             target_short_side=1024, target_megapixels=4.0, align_multiple=8,
             max_width=0, max_height=0):
        Wi, Hi = self._shape(image)
        Wt, Ht = Wi, Hi

        # compute scale based on mode
        if mode == "factor":
            s = max(1.0, float(factor))
        elif mode == "long_side":
            long_in = max(Wi, Hi)
            s = max(1.0, float(target_long_side) / max(1, long_in))
        elif mode == "short_side":
            short_in = min(Wi, Hi)
            s = max(1.0, float(target_short_side) / max(1, short_in))
        elif mode == "megapixels":
            cur_px = float(Wi) * float(Hi)
            target_px = max(1.0, float(target_megapixels) * 1e6)
            # avoid division by zero; ensure scale >= 1
            from math import sqrt
            s = max(1.0, sqrt(target_px / max(1.0, cur_px)))
        else:
            s = 1.0

        # preliminary target
        Wt = int(round(Wi * s))
        Ht = int(round(Hi * s))

        # clamp to max dims if provided (0 = no clamp)
        if max_width > 0 and Wt > max_width:
            s2 = float(max_width) / max(1, Wi)
            s2 = max(1.0, s2)
            Wt, Ht = int(round(Wi * s2)), int(round(Hi * s2))
        if max_height > 0 and Ht > max_height:
            s2 = float(max_height) / max(1, Hi)
            s2 = max(1.0, s2)
            Wt, Ht = int(round(Wi * s2)), int(round(Hi * s2))

        # align to model-friendly multiples
        Wt = max(1, self._ceil_to_multiple(Wt, int(align_multiple)))
        Ht = max(1, self._ceil_to_multiple(Ht, int(align_multiple)))

        return (f"{Wt}x{Ht}",)
