
# 🦑 Kraken Resolution Helper (Exact) — BATCH-SAFE
# Processes every frame in a batch [B,H,W,C]; preserves batch size so downstream video nodes (e.g., RIFE) see >=2 frames.

import math
import re
from typing import Optional, Tuple, List

import numpy as np
import torch
from PIL import Image as PILImage

# ---------- parsing ----------

def _parse_wxH(s: str) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    s = str(s).strip().lower()
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", s)
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        return None
    return w, h

# ---------- tensor <-> PIL ----------

def _tensor_to_pil_list(img: torch.Tensor) -> List[PILImage.Image]:
    """
    Accepts ComfyUI IMAGE (torch tensor [B,H,W,C], float 0..1) and returns list of RGB PIL.Image for all frames.
    """
    if not isinstance(img, torch.Tensor) or img.ndim != 4:
        raise TypeError("Expected torch.Tensor [B,H,W,C]")
    t = img.clamp(0.0, 1.0).detach().cpu()  # [B,H,W,C]
    if t.shape[-1] == 1:
        t = t.repeat(1, 1, 1, 3)
    if t.shape[-1] == 4:
        t = t[..., :3]
    arr = (t.numpy() * 255.0 + 0.5).astype(np.uint8)  # [B,H,W,3]
    frames = [PILImage.fromarray(arr[i], mode="RGB") for i in range(arr.shape[0])]
    return frames

def _pil_list_to_tensor(frames: List[PILImage.Image]) -> torch.Tensor:
    arrs = []
    for im in frames:
        a = np.array(im).astype(np.float32) / 255.0
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        if a.shape[-1] == 4:
            a = a[..., :3]
        arrs.append(a)
    arr = np.stack(arrs, axis=0)  # [B,H,W,3]
    return torch.from_numpy(arr)  # [B,H,W,3]

# ---------- math helpers ----------

def _cover_scale(wi: int, hi: int, wt: int, ht: int) -> float:
    return max(wt / wi, ht / hi)

def _contain_scale(wi: int, hi: int, wt: int, ht: int) -> float:
    return min(wt / wi, ht / hi)

def _round_up(x: float, step: float) -> float:
    return math.ceil(x / step) * step

def _ceil_to_multiple(x: int, m: int) -> int:
    if m <= 1:
        return x
    return int(math.ceil(x / m) * m)

# ---------- core resize/crop ops (single frame) ----------

def _process_frame(
    pil: PILImage.Image,
    Wi: int, Hi: int,
    Wt: int, Ht: int,
    mode: str, interp: int,
    allow_upscale: bool,
    anchor_x: str, anchor_y: str,
    crop_margin_px: int,
    min_overshoot_px: int,
    scale_step: float,
    multiple_of: int,
):
    def _anchor_offsets(big_w: int, big_h: int, small_w: int, small_h: int, ax: str, ay: str):
        off_x = 0 if ax == "left" else (big_w - small_w if ax == "right" else (big_w - small_w) // 2)
        off_y = 0 if ay == "top" else (big_h - small_h if ay == "bottom" else (big_h - small_h) // 2)
        return max(0, off_x), max(0, off_y)

    if mode == "stretch":
        return pil.resize((Wt, Ht), interp)

    elif mode == "keep proportion":
        s = _contain_scale(Wi, Hi, Wt, Ht)
        if s > 1.0 and not allow_upscale:
            s = 1.0
        new_w = max(1, int(round(Wi * s)))
        new_h = max(1, int(round(Hi * s)))
        if multiple_of > 1:
            new_w = _ceil_to_multiple(new_w, multiple_of)
            new_h = _ceil_to_multiple(new_h, multiple_of)
        if not allow_upscale:
            new_w = min(new_w, Wt)
            new_h = min(new_h, Ht)
        scaled = pil.resize((new_w, new_h), interp)
        canvas = PILImage.new("RGB", (Wt, Ht), (0, 0, 0))
        off_x, off_y = _anchor_offsets(Wt, Ht, new_w, new_h, "center", "center")
        canvas.paste(scaled, (off_x, off_y))
        return canvas

    elif mode == "pad":
        s = _contain_scale(Wi, Hi, Wt, Ht)
        if s > 1.0 and not allow_upscale:
            s = 1.0
        new_w = max(1, int(round(Wi * s)))
        new_h = max(1, int(round(Hi * s)))
        if multiple_of > 1:
            new_w = _ceil_to_multiple(new_w, multiple_of)
            new_h = _ceil_to_multiple(new_h, multiple_of)
        if not allow_upscale:
            new_w = min(new_w, Wt)
            new_h = min(new_h, Ht)
        scaled = pil.resize((new_w, new_h), interp)
        canvas = PILImage.new("RGB", (Wt, Ht), (0, 0, 0))
        off_x, off_y = _anchor_offsets(Wt, Ht, new_w, new_h, anchor_x, anchor_y)
        canvas.paste(scaled, (off_x, off_y))
        return canvas

    else:  # "fill / crop"
        margin = max(0, int(crop_margin_px))
        min_over = max(0, int(min_overshoot_px))

        want_w = Wt + 2 * margin + 2 * min_over
        want_h = Ht + 2 * margin + 2 * min_over
        s_req = _cover_scale(Wi, Hi, want_w, want_h)
        s = _round_up(s_req, max(0.001, float(scale_step)))

        if s > 1.0 and not allow_upscale:
            s = 1.0
            new_w = max(1, int(round(Wi * s)))
            new_h = max(1, int(round(Hi * s)))
            if multiple_of > 1:
                new_w = _ceil_to_multiple(new_w, multiple_of)
                new_h = _ceil_to_multiple(new_h, multiple_of)
            scaled = pil.resize((new_w, new_h), interp)
            canvas = PILImage.new("RGB", (Wt, Ht), (0, 0, 0))
            off_x, off_y = _anchor_offsets(Wt, Ht, new_w, new_h, "center", "center")
            canvas.paste(scaled, (off_x, off_y))
            return canvas
        else:
            pred_w = int(math.ceil(Wi * s))
            pred_h = int(math.ceil(Hi * s))
            if multiple_of > 1:
                pred_w = _ceil_to_multiple(pred_w, multiple_of)
                pred_h = _ceil_to_multiple(pred_h, multiple_of)

            guard = 0
            while (pred_w < Wt + min_over or pred_h < Ht + min_over) and guard < 6:
                s = round(s + scale_step, 6)
                pred_w = int(math.ceil(Wi * s))
                pred_h = int(math.ceil(Hi * s))
                if multiple_of > 1:
                    pred_w = _ceil_to_multiple(pred_w, multiple_of)
                    pred_h = _ceil_to_multiple(pred_h, multiple_of)
                guard += 1

            scaled = pil.resize((pred_w, pred_h), interp)
            left, top = _anchor_offsets(pred_w, pred_h, Wt, Ht, anchor_x, anchor_y)
            return scaled.crop((left, top, left + Wt, top + Ht))

# ---------- node ----------

class KrakenResolutionHelper:
    @classmethod
    def INPUT_TYPES(cls):
        mode_options = ["fill / crop", "pad", "keep proportion", "stretch"]
        interp_options = ["lanczos", "bicubic", "bilinear", "nearest"]
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (mode_options, {"default": "fill / crop"}),
                "interpolation": (interp_options, {"default": "lanczos"}),
                "allow_upscale": ("BOOLEAN", {"default": False}),
                "target_resolution_in": ("STRING", {"default": ""}),
            },
            "optional": {
                "preset_in": ("STRING", {"default": ""}),
                "anchor_x": (["left", "center", "right"], {"default": "center"}),
                "anchor_y": (["top", "center", "bottom"], {"default": "center"}),
                "crop_margin_px": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "min_overshoot_px": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "scale_step": ("FLOAT", {"default": 0.01, "min": 0.001, "max": 0.25, "step": 0.001}),
                "multiple_of": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "size")
    FUNCTION = "apply"
    CATEGORY = "kraken/resize"

    def _resolve_target(self, preset_in: str, target_resolution_in: str, Wi: int, Hi: int):
        r = _parse_wxH(target_resolution_in) or _parse_wxH(preset_in)
        if r:
            return r
        return Wi, Hi

    def apply(
        self,
        image,
        mode="fill / crop",
        interpolation="lanczos",
        allow_upscale=False,
        target_resolution_in="",
        preset_in="",
        anchor_x="center",
        anchor_y="center",
        crop_margin_px=0,
        min_overshoot_px=2,
        scale_step=0.01,
        multiple_of=0,
    ):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise TypeError("Expected IMAGE tensor [B,H,W,C]")

        B, Hi, Wi, C = image.shape
        # Resolve target once from strings / incoming size of the first frame
        Wt, Ht = self._resolve_target(preset_in, target_resolution_in, Wi, Hi)

        # Prepare interp
        interp_map = {
            "lanczos": PILImage.LANCZOS,
            "bicubic": PILImage.BICUBIC,
            "bilinear": PILImage.BILINEAR,
            "nearest": PILImage.NEAREST,
        }
        interp = interp_map.get(interpolation, PILImage.LANCZOS)

        # Convert all frames to PIL
        frames_in = _tensor_to_pil_list(image)
        frames_out = []
        for pil in frames_in:
            out_pil = _process_frame(
                pil, Wi, Hi, Wt, Ht, mode, interp,
                allow_upscale, anchor_x, anchor_y,
                crop_margin_px, min_overshoot_px,
                scale_step, multiple_of,
            )
            frames_out.append(out_pil)

        out_tensor = _pil_list_to_tensor(frames_out)  # [B,Ht,Wt,3]
        size_str = f"{Wt} x {Ht}"
        return (out_tensor, size_str)

NODE_CLASS_MAPPINGS = {"KrakenResolutionHelper": KrakenResolutionHelper}
NODE_DISPLAY_NAME_MAPPINGS = {"KrakenResolutionHelper": "🦑 Kraken Resolution Helper (Exact) — Batch Safe"}
