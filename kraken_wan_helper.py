# kraken_wan_helper.py (v11 - Slider UI)
#
# Reverts the duration input to a slider with a 1-10 second range
# as requested. This is the final, feature-complete version.
#
# Category: ⚓ Kraken/WAN

import math
import re
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image as PILImage

# (Helper functions remain unchanged)

def _tensor_to_pil(img: torch.Tensor) -> PILImage.Image:
    if not isinstance(img, torch.Tensor) or img.ndim != 4:
        raise TypeError("Expected torch.Tensor [B,H,W,C]")
    t = img[0].clamp(0.0, 1.0).detach().cpu()
    if t.shape[-1] == 1: t = t.repeat(1, 1, 3)
    if t.shape[-1] == 4: t = t[..., :3]
    arr = (t.numpy() * 255.0 + 0.5).astype(np.uint8)
    return PILImage.fromarray(arr, mode="RGB")

def _pil_to_tensor(pil: PILImage.Image) -> torch.Tensor:
    a = np.array(pil).astype(np.float32) / 255.0
    if a.ndim == 2: a = np.stack([a, a, a], axis=-1)
    if a.shape[-1] == 4: a = a[..., :3]
    return torch.from_numpy(a).unsqueeze(0)

def _preprocess_image(pil_in: PILImage.Image, max_long_side: int) -> PILImage.Image:
    Wi, Hi = pil_in.size
    if max(Wi, Hi) <= max_long_side:
        return pil_in
    scale = max_long_side / max(Wi, Hi)
    new_w = max(1, int(round(Wi * scale)))
    new_h = max(1, int(round(Hi * scale)))
    return pil_in.resize((new_w, new_h), PILImage.LANCZOS)

def _nearest_common_aspect_label(w, h):
    if w <= 0 or h <= 0: return "9:16"
    r = float(w) / float(h)
    candidates = [("16:9", 16/9), ("9:16", 9/16), ("1:1", 1.0), ("4:5", 4/5), ("3:4", 3/4)]
    return min(candidates, key=lambda c: abs(r - c[1]))[0]

def _snap_down(v, m):
    if m <= 1: return int(v)
    return max(m, (int(v) // m) * m)

def _size_by_ar_and_group(ar_label: str, group: str):
    if group == "720":
        table = {"16:9":(1280,720),"9:16":(720,1280),"1:1":(960,960),"4:5":(864,1080),"3:4":(816,1088)}
    else: # 480 group
        table = {"16:9":(832,480),"9:16":(480,832),"1:1":(624,624),"4:5":(384,480),"3:4":(368,496)}
    return table.get(ar_label, table["9:16"])


# ---------- The Node Class ----------

class KrakenWanHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "performance_tier": (["720p Quality Tier", "480p Speed Tier"], {"default": "720p Quality Tier"}),
                "max_preprocess_long_side": ("INT", {"default": 1280, "min": 256, "max": 2048, "step": 64}),
                "multiple_of": ("INT", {"default": 16, "min": 1, "max": 128, "step": 1}),
                # MODIFIED: Changed duration back to a slider with a 1-10 second range
                "duration_seconds": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1}),
                "fps": ([16, 24, 30, 8, 12], {"default": 16}),
                "add_bookend_frame": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("processed_first_frame", "processed_last_frame", "width", "height", "frames(length)", "fps", "resolution_text")
    FUNCTION = "generate"
    CATEGORY = "⚓ Kraken/WAN"

    def generate(self, performance_tier, max_preprocess_long_side, multiple_of, 
                 duration_seconds, fps, add_bookend_frame, first_frame=None, last_frame=None):
        
        # (All processing logic remains the same)
        W, H = -1, -1
        processed_pil_first = None
        if first_frame is not None:
            input_pil_first = _tensor_to_pil(first_frame)
            processed_pil_first = _preprocess_image(input_pil_first, max_preprocess_long_side)
            W, H = processed_pil_first.size
        if processed_pil_first is None:
            processed_pil_first = PILImage.new("RGB", (64, 64), (0, 0, 0))
        processed_tensor_first_out = _pil_to_tensor(processed_pil_first)

        processed_pil_last = None
        if last_frame is not None:
            input_pil_last = _tensor_to_pil(last_frame)
            processed_pil_last = _preprocess_image(input_pil_last, max_preprocess_long_side)
        if processed_pil_last is None:
            processed_pil_last = PILImage.new("RGB", (64, 64), (0, 0, 0))
        processed_tensor_last_out = _pil_to_tensor(processed_pil_last)
        
        group = "720" if "720p" in performance_tier else "480"
        if W > 0 and H > 0:
            ar_label = _nearest_common_aspect_label(W, H)
            width, height = _size_by_ar_and_group(ar_label, group)
        else:
            width, height = _size_by_ar_and_group("9:16", group)

        frames = int(duration_seconds) * int(fps)
        if add_bookend_frame:
            frames += 1

        width = _snap_down(width, multiple_of)
        height = _snap_down(height, multiple_of)
        resolution_text = f"{width} x {height}"

        return (processed_tensor_first_out, processed_tensor_last_out, width, height, frames, float(fps), resolution_text)

NODE_CLASS_MAPPINGS = {"KrakenWanHelper": KrakenWanHelper}
NODE_DISPLAY_NAME_MAPPINGS = {"KrakenWanHelper": "🦑 Kraken WAN Helper"}