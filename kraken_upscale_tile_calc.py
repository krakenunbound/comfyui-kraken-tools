"""
Kraken Upscale & Tile Calc
MIT License © 2025 The Kraken (@KrakenUnbound)

Computes a working upscale + tiles/overlap for USDU with strict cover.
`min_overshoot_px` guarantees the predicted canvas is at least that many pixels
larger than the final target on BOTH dimensions (after alignment).
"""

import math, os, re, time, logging
import numpy as np, torch
from PIL import Image, ImageDraw
import folder_paths

counter = 0

def _round_to(x, base): return int(round(x / base) * base)
def _ceil_to(x, base): return int(math.ceil(x / base) * base)
def _clamp(v, lo, hi): return max(lo, min(hi, v))
def _blank_image_tensor(w=8, h=8): return torch.zeros((1, h, w, 3), dtype=torch.float32)

class KrakenUpscaleTileCalc:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolution_in": ("STRING", {"default": "1024 x 1024"}),
            },
            "optional": {
                "preset_in": ("STRING", {"default": "Custom"}),
                "image": ("IMAGE",),
                "scale_step": ("FLOAT", {"default": 0.01, "min": 0.005, "max": 0.25, "step": 0.005}),
                "crop_margin_px": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "min_overshoot_px": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1}),
                "tile_padding": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1}),
                "seam_fix_width": ("INT", {"default": 64, "min": 0, "max": 256, "step": 1}),
                "seam_fix_mask_blur": ("INT", {"default": 16, "min": 0, "max": 64, "step": 1}),
                "seam_fix_padding": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
                "min_upscale_policy": (["auto","off","1.5x","2.0x","custom"], {"default": "auto"}),
                "min_upscale_custom": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 4.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = (
        "IMAGE","FLOAT","INT","INT","INT","INT","INT","INT","INT","STRING","STRING","STRING"
    )
    RETURN_NAMES = (
        "image_out","upscale_by","tile_width","tile_height","mask_blur","tile_padding",
        "seam_fix_width","seam_fix_mask_blur","seam_fix_padding",
        "predicted_resolution","target_resolution","target_preset"
    )
    FUNCTION = "calculate"
    CATEGORY = "kraken/upscale"

    def _parse_resolution_ws(self, s):
        s = (s or "").lower().replace(" ", "")
        if "x" in s:
            try:
                w, h = s.split("x")
                return (int(w), int(h))
            except Exception:
                return None
        return None

    def _parse_resolution_from_label(self, label):
        s = str(label or "")
        m = re.search(r"(\d+)\s*[xX]\s*(\d+)", s)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _image_size(self, image, resolution_in):
        if image is not None:
            if isinstance(image, torch.Tensor):
                if image.ndim == 4: _, h, w, _ = image.shape
                else: h, w = image.shape[0], image.shape[1]
                return (w, h)
            if isinstance(image, np.ndarray):
                if image.ndim == 4: _, h, w, _ = image.shape
                else: h, w = image.shape[0], image.shape[1]
                return (w, h)
        return self._parse_resolution_ws(resolution_in) or (1024, 1024)

    def _choose_min_upscale(self, wi, hi, Wt, Ht, policy, custom):
        policy = (policy or "auto").lower()
        if policy == "off": return 1.0
        if policy == "1.5x": return 1.5
        if policy == "2.0x": return 2.0
        if policy == "custom":
            try: return max(1.0, min(4.0, float(custom)))
            except Exception: return 1.5
        src_mp = (wi*hi)/1_000_000.0
        tgt_mp = (Wt*Ht)/1_000_000.0
        if max(Wt,Ht) <= 1024 or tgt_mp <= 1.0: return 2.0
        if src_mp <= 2.0: return 2.0
        return 1.5

    def _make_preview(self, pred_w, pred_h, Wt, Ht):
        w, h = max(pred_w, Wt), max(pred_h, Ht)
        scale = max(1, min(512 / w, 512 / h))
        canvas = Image.new("RGB", (int(w*scale), int(h*scale)), (24,24,24))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0,0,canvas.size[0]-1,canvas.size[1]-1], outline=(160,160,160), width=2)
        tw, th = int(Wt*scale), int(Ht*scale)
        x0, y0 = (canvas.size[0]-tw)//2, (canvas.size[1]-th)//2
        draw.rectangle([x0,y0,x0+tw-1,y0+th-1], outline=(0,200,120), width=2)
        draw.text((6,6), f"pred {pred_w}x{pred_h}", fill=(220,220,220))
        draw.text((6,24), f"target {Wt}x{Ht}", fill=(0,200,120))
        return canvas

    def calculate(
        self,
        resolution_in,
        preset_in="Custom",
        image=None,
        scale_step=0.01,
        crop_margin_px=0,
        min_overshoot_px=2,
        mask_blur=16,
        tile_padding=16,
        seam_fix_width=64,
        seam_fix_mask_blur=16,
        seam_fix_padding=32,
        min_upscale_policy="auto",
        min_upscale_custom=1.5,
    ):
        wi, hi = self._image_size(image, resolution_in)
        tgt = self._parse_resolution_ws(preset_in) or self._parse_resolution_from_label(preset_in) or self._parse_resolution_ws(resolution_in) or (wi, hi)
        Wt, Ht = tgt

        margin = _clamp(int(crop_margin_px), 0, max(Wt, Ht)//10)
        min_over = max(0, int(min_overshoot_px))

        # Required cover scale includes both margin & overshoot
        want_w = Wt + 2*margin + 2*min_over
        want_h = Ht + 2*margin + 2*min_over
        required = max(want_w/wi, want_h/hi)

        s = max(1.0, round(math.ceil(required / scale_step) * scale_step, 3))
        # Enforce minimum upscale policy so USDU actually does work
        min_s = self._choose_min_upscale(wi, hi, Wt, Ht, min_upscale_policy, min_upscale_custom)
        if min_s > 1.0:
            s = max(s, round(min_s / scale_step) * scale_step)
        s = round(s, 3)

        pred_align = 8
        pred_w = _ceil_to(wi * s, pred_align)
        pred_h = _ceil_to(hi * s, pred_align)

        # Strict cover: ensure >= target + min_over after alignment
        guard = 0
        while (pred_w < Wt + min_over or pred_h < Ht + min_over) and guard < 6:
            s = round(s + scale_step, 3)
            pred_w = _ceil_to(wi * s, pred_align)
            pred_h = _ceil_to(hi * s, pred_align)
            guard += 1

        # Tiles & overlap
        overlap = _clamp(int(round(min(pred_w, pred_h) * 0.015)), 32, 96)
        tile_w = _round_to(_clamp(pred_w // 2 + overlap, 512, 2048), 64)
        tile_h = _round_to(_clamp(pred_h // 2 + overlap, 512, 2048), 64)

        predicted_resolution = f"{pred_w} x {pred_h}"
        target_resolution = f"{Wt} x {Ht}"
        target_preset = f"{Wt}x{Ht}"

        global counter; counter += 1
        full_output_folder = folder_paths.get_temp_directory()
        file = f"kraken_tilecalc_{time.time()}_{counter:05}.png"
        try:
            preview_img = self._make_preview(pred_w, pred_h, Wt, Ht)
            preview_img.save(os.path.join(full_output_folder, file), compress_level=4)
            images = [{"filename": file, "subfolder": "", "type": "temp"}]
        except Exception as e:
            logging.warning(f"Preview generation failed: {e}"); images = []

        image_out = image if image is not None else _blank_image_tensor()
        return {
            "ui": {"images": images},
            "result": (image_out, s, tile_w, tile_h, mask_blur, tile_padding, seam_fix_width,
                       seam_fix_mask_blur, seam_fix_padding, predicted_resolution, target_resolution, target_preset),
        }
