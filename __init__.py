# -*- coding: utf-8 -*-
"""
Kraken custom nodes - Release the Kraken! 🐙
"""
import torch

# --- Performance Backend Hints (Windows & Torch 2.x friendly) ---
try:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel.enable_flash_sdp(True)
    sdp_kernel.enable_mem_efficient_sdp(True)
    sdp_kernel.enable_math_sdp(True)
except Exception:
    pass

import pathlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _safe_register(pyname, clsname, display):
    """
    Safely import and register a node class from a sibling module.
    pyname: python module filename without .py
    clsname: class name inside that module
    display: string used by ComF_UI for the node display name
    """
    try:
        mod = __import__(f"{__name__}.{pyname}", fromlist=[clsname])
        cls = getattr(mod, clsname)
        NODE_CLASS_MAPPINGS[clsname] = cls
        NODE_DISPLAY_NAME_MAPPINGS[clsname] = display
    except Exception as e:
        print(f"[kraken_tools] Skipped {clsname}: {e}")

# --- Existing nodes you already use ---
_safe_register("kraken_loras3", "KrakenLoras3", "🐙 Kraken LoRA Loader (3)")
_safe_register("kraken_empty_latent", "KrakenEmptyLatentImage", "🐙 Kraken Empty Latent Image")
_safe_register("kraken_resolution_helper", "KrakenResolutionHelper", "🐙 Kraken Resolution Helper")
_safe_register("kraken_upscale_tile_calc", "KrakenUpscaleTileCalc", "🐙 Kraken Upscale & Tile Calc")
_safe_register("kraken_wan_helper", "KrakenWanHelper", "🐙 Kraken WAN Helper")
_safe_register("kraken_ksampler", "KrakenKSampler", "🐙 Kraken KSampler")
_safe_register("kraken_dual_clip_loader", "KrakenDualCLIPLoader", "🐙 Kraken Dual CLIP Loader")
_safe_register("kraken_image_resize", "KrakenImageResize", "🐙 Kraken Image Resize")
_safe_register("kraken_image_processor", "KrakenImageProcessor", "🐙 Kraken Image Processor")
_safe_register("kraken_preset_from_image", "KrakenPresetFromImage", "🐙 Kraken Preset From Image")
_safe_register("kraken_ollama_chat", "KrakenOllamaPromptChat", "🐙 Kraken Ollama Prompt Chat")
_safe_register("kraken_last_frame_meta", "KrakenLastFrameMeta", "🐙 Kraken Last Frame + Meta")
_safe_register("kraken_wan_prompt", "KrakenWanPrompt", "🐙 Kraken WAN Prompt Splitter")

# --- Your custom checkpoint loader (already present) ---
_safe_register("kraken_checkpoint_loader", "KrakenCheckpointLoader", "🐙 Kraken Checkpoint Loader")

# --- Kraken Unbound Prompt generator ---
_safe_register("kraken_unbound_prompt", "KrakenUnboundPrompt", "🐙 Kraken Unbound Prompt")

# --- NOTE: All-in-One Discord Bot moved to standalone package: kraken-discord-bot ---
# See: https://github.com/krakenunbound/kraken-discord-bot

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
