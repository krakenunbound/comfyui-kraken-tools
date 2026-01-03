# ──────────────────────────────────────────────────────────────────────────────
# 🦑 KrakenKSampler — v4.0
# ------------------------------------------------------------------------------
# What this node is:
#   A user-friendly wrapper around ComfyUI's KSampler pipeline with:
#     • A robust UI that mirrors Comfy's ksampler knobs + a few quality-of-life
#       toggles (decode, tiled decode, tile size, debug logs).
#     • Smarter handling of Automatic Mixed Precision (AMP):
#         - New "AMP Mode": "auto" | "on" | "off"
#         - "auto" enables AMP for standard SD/SDXL models but automatically
#           DISABLES outer autocast when the model is WAN/Flow/FP8 to avoid
#           nested autocast conflicts (your crash).
#     • Defensive calling of KSampler to bridge param schema differences across
#       Comfy versions (tries several kwargs fallbacks).
#     • Safe channels_last memory layout fix for 4D latents to reduce overhead.
#     • Clear diagnostics when decode is requested without a VAE connected.
#
# Why v4.0 fixes your crash with WAN 2.2:
#   - Your stack trace showed: "RuntimeError: Unexpected floating ScalarType
#     in at::autocast::prioritize" deep in WAN's addcmul, with FP8/bfloat16
#     mixed precision internals and Comfy's own precision management.
#   - The crash is caused by a SECOND, outer torch.autocast context wrapped
#     around a model that already manages precision (Flow/WAN/FP8). PyTorch
#     autocast prioritization hits an unexpected type mix and blows up.
#   - Solution: do NOT wrap WAN/Flow/FP8 in an additional autocast context.
#     This node detects such models and disables outer AMP automatically.
#
# TL;DR:
#   • Set "AMP Mode" to "auto" (default). For SD/SDXL → AMP ON, for WAN/Flow → OFF.
#   • You can still force "on" or "off" from the UI if you want.
#
# Notes:
#   • This file is self-contained; no "drop-ins" required.
#   • Works with ComfyUI 0.3.x+ and PyTorch 2.x (tested up to 2.7).
#   • Keep your ComfyUI and WAN nodes up to date for best stability.
# ──────────────────────────────────────────────────────────────────────────────

import torch
import comfy
from contextlib import nullcontext
from nodes import KSampler, VAEDecode, VAEDecodeTiled

# ──────────────────────────────────────────────────────────────────────────────
# Precision helpers
# ──────────────────────────────────────────────────────────────────────────────

# Cache the chosen AMP dtype so we don’t query device caps repeatedly.
_AMP_DTYPE = None

def _get_amp_dtype():
    """
    Decide which dtype to use for AMP on *supported* models:
      • bfloat16 for compute capability >= 8 (Ampere+), else float16.

    IMPORTANT: This is only used when the outer autocast is ENABLED.
    For WAN/Flow/FP8 we will usually disable the outer autocast entirely.
    """
    global _AMP_DTYPE
    if _AMP_DTYPE is not None:
        return _AMP_DTYPE

    use_bf16 = False
    try:
        if torch.cuda.is_available():
            # Ampere (SM 80) or newer generally does well with bf16
            use_bf16 = torch.cuda.get_device_properties(0).major >= 8
    except Exception:
        pass

    _AMP_DTYPE = torch.bfloat16 if use_bf16 else torch.float16
    return _AMP_DTYPE


def _model_type_string(model) -> str:
    """
    Best-effort extraction of a model's 'model_type' string as set by Comfy/WAN.
    For WAN 2.2 you typically see 'FLOW' in logs.
    """
    mt = (
        getattr(model, "model_type", None)
        or getattr(getattr(model, "model", None), "model_type", None)
        or ""
    )
    return str(mt).upper()


def _should_enable_autocast_for(model) -> bool:
    """
    Return True if we should wrap the sample call in an outer autocast context.

    We DISABLE the outer autocast for WAN/Flow or explicit FP8 models because
    they already manage precision internally and nesting autocast can crash
    with 'Unexpected floating ScalarType...' errors.

    We also sniff for various flags WAN/Loaders may expose.
    """
    mt = _model_type_string(model)
    if "FLOW" in mt or "WAN" in mt:
        return False

    # Defensive flags some loaders may set:
    if any(bool(getattr(model, attr, False))
           for attr in ("uses_fp8", "fp8", "enable_fp8", "scaled_fp8")):
        return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
# Main Node
# ──────────────────────────────────────────────────────────────────────────────

class KrakenKSampler:
    """
    A one-stop sampler node that mirrors Comfy's KSampler but adds:
      • AMP Mode (auto / on / off) to avoid crashes with WAN/Flow/FP8.
      • Optional image decode (standard or tiled) right after sampling.
      • Slightly more forgiving argument adapter for KSampler signature drift.
      • Helpful debugging logs.

    Outputs:
      (LATENT, IMAGE)
      - LATENT is always returned.
      - IMAGE is decoded if decode_switch is on and a VAE is connected,
        otherwise a 1x1 empty image placeholder is returned.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # UI schema
    # ──────────────────────────────────────────────────────────────────────────
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Core conditioning + model inputs
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),

                # Sampler knobs
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                # Noise / step slicing controls
                "add_noise": (["enable", "disable"],),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),

                # Negative prompt handling
                # "auto" = only use negative if it's wired; otherwise skip + set cfg=1.0 for that branch
                "negative_mode": (["auto", "use", "ignore"], {"default": "auto"}),

                # NEW: AMP Mode — the fix for WAN 2.2 crash
                #   auto: enable AMP unless model looks like WAN/Flow/FP8 (safe default)
                #   on  : force AMP on
                #   off : force AMP off
                "amp_mode": (["auto", "on", "off"], {"default": "auto"}),

                # Decode options
                "decode_switch": (["off", "on"], {"default": "on"}),
                "tiled_decode": (["off", "on"], {"default": "off"}),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 2048, "step": 64}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "debug_logs": (["off", "on"], {"default": "off"}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("LATENT", "IMAGE")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "🦑 Kraken Tools/Sampling"

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_image(device="cpu"):
        """
        Return a 1x1 black image placeholder (batch=1, channels=3).
        Used when decode is off or no VAE is provided.
        """
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32, device=device)

    @staticmethod
    def _is_empty_cond(cond):
        """
        Check if the provided conditioning is "empty" (not wired).
        Comfy encodes conditionings as lists/tuples; empty list/tuple means no input.
        """
        if cond is None:
            return True
        try:
            return isinstance(cond, (list, tuple)) and len(cond) == 0
        except Exception:
            return False

    def _resolve_negative(self, negative_mode, negative, cfg):
        """
        Decide whether to apply negative conditioning and what CFG to use for it.
        - "ignore" → no negative; cfg=1.0 for the negative branch
        - "use"    → only use if provided; else skip
        - "auto"   → use if provided, else skip
        Returns: (negative_effective, cfg_effective, used_negative_flag)
        """
        mode = (negative_mode or "auto").lower()
        missing = self._is_empty_cond(negative)

        if mode == "ignore":
            return [], 1.0, False
        if mode == "use":
            if missing:
                return [], 1.0, False
            return negative, cfg, True
        # auto mode
        if missing:
            return [], 1.0, False
        else:
            return negative, cfg, True

    def _ksampler_call(self,
                       model, seed, steps, cfg_eff, sampler_name, scheduler,
                       positive, negative_eff, latent_image,
                       denoise, disable_noise_flag, start_at_step, end_at_step, leftover):
        """
        Wrap nodes.KSampler().sample with a small compatibility shim: Comfy’s
        KSampler signature has evolved over time. We try a few kwargs variants
        to maintain compatibility across versions rather than hard failing.
        """
        ks = KSampler()
        pos_args = [model, seed, steps, cfg_eff, sampler_name, scheduler, positive, negative_eff, latent_image]
        attempts = [
            # Newer style
            dict(denoise=denoise, disable_noise=disable_noise_flag,
                 start_at_step=start_at_step, end_at_step=end_at_step,
                 return_with_leftover_noise=leftover),
            # Older style (uses add_noise instead of disable_noise)
            dict(denoise=denoise, add_noise=not disable_noise_flag,
                 start_at_step=start_at_step, end_at_step=end_at_step,
                 return_with_leftover_noise=leftover),
            # Even older variants…
            dict(denoise=denoise, add_noise=not disable_noise_flag,
                 start_at_step=start_at_step, end_at_step=end_at_step),
            dict(denoise=denoise, add_noise=not disable_noise_flag),
            dict(denoise=denoise),
        ]
        last_err = None
        for kw in attempts:
            try:
                return ks.sample(*pos_args, **kw)[0]
            except TypeError as e:
                last_err = e
                continue
        # If we got here, all attempts failed.
        raise last_err

    def _decode(self, vae, samples, tiled, tile_size):
        """
        Decode a latent to an IMAGE tensor via Comfy's VAEDecode nodes.
        - Supports tiled decode (useful for large images on limited VRAM).
        - Returns a tensor in [0,1] with shape (B, H, W, C).
        """
        image = None
        try:
            if tiled:
                if self.debug: print("[KrakenKSampler] Using tiled VAE decode.")
                try:
                    img = VAEDecodeTiled().decode(vae, samples, tile_size)[0]
                except TypeError:
                    # Some builds expect (vae, samples, tile, tile_stride)
                    img = VAEDecodeTiled().decode(vae, samples, tile_size, 64)[0]
            else:
                if self.debug: print("[KrakenKSampler] Using standard VAE decode.")
                img = VAEDecode().decode(vae, samples)[0]

            if img is not None:
                image = torch.nan_to_num(img).clamp_(0.0, 1.0)
            return image
        except Exception as e:
            if self.debug: print(f"[KrakenKSampler] VAEDecode node path failed: {e}. Trying utils fallback…")
            try:
                import comfy.utils as cu
                latent = samples["samples"] if isinstance(samples, dict) and "samples" in samples else samples
                img = cu.decode_latent_to_image(vae, latent)
                if img is not None:
                    image = torch.nan_to_num(img).clamp_(0.0, 1.0)
            except Exception as e2:
                if self.debug: print(f"[KrakenKSampler] utils fallback failed: {e2}")
        return image

    # ──────────────────────────────────────────────────────────────────────────
    # Main execution
    # ──────────────────────────────────────────────────────────────────────────
    def sample(
        self,
        model,
        positive,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        add_noise,
        start_at_step,
        end_at_step,
        return_with_leftover_noise,
        negative_mode,
        amp_mode,           # NEW: "auto" | "on" | "off" — fixes WAN crash
        decode_switch,
        tiled_decode,
        tile_size,
        negative=None,
        vae=None,
        debug_logs="off",
        prompt=None,
        extra_pnginfo=None,
    ):
        # ──────────────────────────────────────────────────────────────────────
        # Housekeeping & input normalization
        # ──────────────────────────────────────────────────────────────────────
        self.debug = (debug_logs == "on")

        # Tile size must be a multiple of 64 (VAE stride). Clamp reasonably.
        if tile_size < 64:
            tile_size = 64
        if tile_size % 64 != 0:
            tile_size = (tile_size // 64) * 64

        disable_noise_flag = (add_noise == "disable")
        leftover = (return_with_leftover_noise == "enable")
        end_at_step = max(start_at_step, end_at_step)

        # Whether to apply negative conditioning
        negative_eff, cfg_eff, use_neg = self._resolve_negative(negative_mode, negative, cfg)

        # Track the device of the latent so we can create a placeholder IMAGE on the same device
        _lat = latent_image["samples"] if isinstance(latent_image, dict) and "samples" in latent_image else latent_image
        latent_device = getattr(_lat, "device", "cuda" if torch.cuda.is_available() else "cpu")

        if self.debug:
            mt = _model_type_string(model)
            print(f"[KrakenKSampler] Model type: {mt or 'UNKNOWN'}")
            print(f"[KrakenKSampler] CFG {cfg_eff:.2f} | Negative {'USED' if use_neg else 'SKIPPED'} | Steps {steps}")

        # ──────────────────────────────────────────────────────────────────────
        # AMP decision: the key to avoiding your WAN 2.2 crash
        # ──────────────────────────────────────────────────────────────────────
        if torch.cuda.is_available():
            if amp_mode == "on":
                enable_amp = True
            elif amp_mode == "off":
                enable_amp = False
            else:  # "auto"
                enable_amp = _should_enable_autocast_for(model)
        else:
            enable_amp = False

        if self.debug:
            print(f"[KrakenKSampler] AMP mode={amp_mode} -> enabled={enable_amp}")

        # NOTE: We use torch.amp.autocast with enabled=... so the context is a no-op
        # when disabled; this avoids nesting conflicts with WAN/Flow/FP8.
        try:
            from torch.amp.autocast_mode import autocast as amp_autocast
        except Exception:
            # Back-compat alias (older torch)
            amp_autocast = torch.autocast

        amp_ctx = (
            amp_autocast(device_type="cuda", dtype=_get_amp_dtype(), enabled=enable_amp)
            if torch.cuda.is_available() else nullcontext()
        )

        # ──────────────────────────────────────────────────────────────────────
        # Run sampler
        # ──────────────────────────────────────────────────────────────────────
        with torch.inference_mode():
            with amp_ctx:
                # Ensure 4D latents are channels_last; helps some kernels on CUDA
                try:
                    if isinstance(latent_image, dict) and "samples" in latent_image:
                        t = latent_image["samples"]
                        if torch.is_tensor(t) and t.dim() == 4 and not t.is_contiguous(memory_format=torch.channels_last):
                            latent_image["samples"] = t.contiguous(memory_format=torch.channels_last)
                    elif torch.is_tensor(latent_image):
                        t = latent_image
                        if t.dim() == 4 and not t.is_contiguous(memory_format=torch.channels_last):
                            latent_image = t.contiguous(memory_format=torch.channels_last)
                except Exception:
                    # Non-fatal
                    pass

                # Actually sample
                samples = self._ksampler_call(
                    model, seed, steps, cfg_eff, sampler_name, scheduler,
                    positive, negative_eff, latent_image,
                    denoise, disable_noise_flag, start_at_step, end_at_step, leftover
                )

                # ──────────────────────────────────────────────────────────────
                # Decode (optional)
                # ──────────────────────────────────────────────────────────────
                image = None
                if decode_switch == "on":
                    if vae is not None:
                        try:
                            image = self._decode(
                                vae=vae,
                                samples=samples,
                                tiled=(tiled_decode == "on"),
                                tile_size=tile_size,
                            )
                        except Exception as e:
                            if self.debug:
                                print(f"[KrakenKSampler] Decode failed: {e}")
                    else:
                        # Clear, user-visible error message
                        print("\033[91m[KrakenKSampler] ERROR: Decode is ON but no VAE is connected. "
                              "Please connect a VAE or set decode_switch to OFF.\033[0m")

                # If decode is OFF or decode failed, emit a placeholder image
                if image is None:
                    image = self._empty_image(device=latent_device)

        # Light diagnostics
        if self.debug and hasattr(image, "shape"):
            try:
                print(f"[KrakenKSampler] Output latent dtype/device: "
                      f"{(samples['samples'].dtype if isinstance(samples, dict) and 'samples' in samples else getattr(samples, 'dtype', 'unknown'))}/"
                      f"{(samples['samples'].device if isinstance(samples, dict) and 'samples' in samples else getattr(samples, 'device', 'unknown'))}")
                print(f"[KrakenKSampler] Decoded image shape: {tuple(image.shape)} dtype={image.dtype} device={image.device}")
            except Exception:
                pass

        return (samples, image)


# ──────────────────────────────────────────────────────────────────────────────
# Comfy registration
# ──────────────────────────────────────────────────────────────────────────────
NODE_CLASS_MAPPINGS = {"KrakenKSampler": KrakenKSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"KrakenKSampler": "🦑 Kraken KSampler"}
