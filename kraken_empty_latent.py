import torch
import math
import os
import time
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import folder_paths

counter = 0  # Global counter for unique filenames

def _ratio_label_from_preset_or_size(preset: str, w: int, h: int) -> str:
    m = re.search(r'\(([^)]+)\)', preset)
    if m and ':' in m.group(1):
        return m.group(1).strip()
    g = math.gcd(int(w), int(h))
    g = g if g > 0 else 1
    return f"{int(w)//g}:{int(h)//g}"

class KrakenEmptyLatentImage:
    @classmethod
    def INPUT_TYPES(cls):
        megapixel_options = [
            "0.5", "0.75", "1.0", "1.25", "1.5", "1.75",
            "2.0", "2.25", "2.5", "2.75", "3.0", "Custom"
        ]

        aspect_options = [
            "Custom",

            # --- Vertical / Portrait ---
            "9:20 (Android Vertical)",
            "Screen: Android Portrait - 1080x2400 (9:20)",
            "Screen: iPhone Portrait - 1170x2532 (9:19)",
            "9:16 (Vertical)",
            "Screen: Full HD Portrait - 1080x1920 (9:16)",
            "Screen: QHD Portrait - 1440x2560 (9:16)",
            "5:8 (Golden Portrait)",
            "2:3 (Portrait)",
            "3:4 (Portrait)",
            "4:5 (Portrait)",
            "Social: Instagram Portrait - 1080x1350 (4:5)",

            # --- Square ---
            "1:1 (Square)",
            "AI: Square - 1024x1024 (1:1)",
            "Social: Instagram Square - 1080x1080 (1:1)",

            # --- Landscape ---
            "5:4 (Classic Landscape)",
            "4:3 (Landscape)",
            "3:2 (Landscape)",
            "16:10 (Widescreen Laptop)",
            "Screen: WUXGA - 1920x1200 (16:10)",
            "5:3 (Wide Landscape)",
            "7:4 (Wide Video)",
            "Video: KlingAI Wide - 1344x768 (7:4)",
            "16:9 (HD)",
            "Video: HD - 1920x1080 (16:9)",
            "Screen: QHD - 2560x1440 (16:9)",
            "Screen: 4K UHD - 3840x2160 (16:9)",
            "Social: YouTube Thumbnail - 1280x720 (16:9)",

            # --- Cinematic / Panorama ---
            "1.85:1 (Film Flat)",
            "1.91:1 (Social Banner)",
            "Social: Facebook Cover - 1200x628 (1.91:1)",

            # ⭐ Explicit 2:1 for panoramic / 360 LoRAs
            "2:1 (Panoramic / 360)",

            "21:9 (Ultrawide)",
            "Screen: UWQHD - 3440x1440 (21:9)",
            "2.39:1 (Cinemascope)",
            "Video: DCI 4K Scope - 4096x1716 (2.39:1)",
            "32:9 (Super Ultrawide)",
            "Screen: Dual QHD - 5120x1440 (32:9)",
            "3:1 (Wide Banner)"
        ]

        divisible_options = ["Auto", 8, 16, 32, 64, 128, 256]

        return {
            "required": {
                "megapixel": (megapixel_options, {"default": "1.0"}),
                "preset": (aspect_options, {"default": "Custom"}),
                "use_custom_aspect": ("BOOLEAN", {"default": False}),
                "custom_aspect": ("STRING", {"default": "1:1"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8}),
                "divisible_by": (divisible_options, {"default": "Auto"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "enable_preview": ("BOOLEAN", {"default": True}),
                "max_preview_size": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "width_out", "height_out", "resolution", "preset_out")
    FUNCTION = "generate"
    CATEGORY = "kraken/latent"

    def generate(
        self, megapixel, preset, use_custom_aspect, custom_aspect,
        width, height, divisible_by, batch_size,
        enable_preview=True, max_preview_size=512
    ):
        ar = 1.0
        ratio_display = f"{width}x{height}"
        target_preset_str = None

        if use_custom_aspect:
            ratio_display = custom_aspect
            w_part, h_part = map(float, custom_aspect.split(":"))
            ar = w_part / h_part

        elif preset != "Custom":
            if " - " in preset:
                size_str = preset.split(" - ")[1].split(" (")[0]
                width, height = map(int, size_str.split("x"))
                ar = width / height
                ratio_display = _ratio_label_from_preset_or_size(preset, width, height)
                target_preset_str = size_str
            else:
                ratio_str = preset.split(" (")[0]
                w_part, h_part = map(float, ratio_str.split(":"))
                ar = w_part / h_part
                ratio_display = f"{w_part}:{h_part}"

        else:
            ar = width / height

        if megapixel == "Custom":
            mp = (width * height) / 1_000_000
        else:
            mp = float(megapixel)
            base_width = math.sqrt(mp * 1_000_000 * ar)
            base_height = math.sqrt(mp * 1_000_000 / ar)
            width = int(round(base_width / 64) * 64)
            height = int(round(base_height / 64) * 64)

        latent = torch.zeros([batch_size, 16, height // 8, width // 8])
        resolution = f"{width} x {height}"
        preset_out = target_preset_str or resolution

        return ({"samples": latent}, width, height, resolution, preset_out)
