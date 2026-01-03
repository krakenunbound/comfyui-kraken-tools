
import os
from typing import List, Tuple, Optional

import torch

# Comfy internals
import folder_paths
from comfy import utils as comfy_utils
from comfy import sd as comfy_sd
from comfy import model_management


class KrakenDualCLIPLoader:
    """
    Pure custom dual text-encoder loader (no fallback to stock nodes).
    Builds a single CLIP object (works with CLIPTextEncode), supports CLIP-L + T5 pairing,
    warmup, precision control, and device placement. Order-independent.
    """

    @classmethod
    def INPUT_TYPES(cls):
        te_files = folder_paths.get_filename_list("text_encoders")
        devices = ["auto", "cuda", "cpu"]
        try:
            if torch.backends.mps.is_available():
                devices.append("mps")
        except Exception:
            pass
        try:
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                devices.append("xpu")
        except Exception:
            pass

        return {
            "required": {
                "type": (["flux", "sdxl", "sd3", "hunyuan_dit"], {"default": "flux"}),
                "clip_path": (te_files, {"default": "clip_l.safetensors"}),
                "t5_path": (te_files, {"default": os.path.join("t5", "t5xxl_fp16.safetensors")}),
                "mode": (["dual", "t5-only", "clip-only"], {"default": "dual"}),
                "device": (devices, {"default": "auto"}),
                "precision_override": (["auto", "fp16", "bf16", "fp32"], {"default": "auto"}),
                "warmup": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CLIP", "STRING",)
    RETURN_NAMES = ("clip", "diagnostics",)
    OUTPUT_NODE = False
    FUNCTION = "load"
    CATEGORY = "Kraken/Loaders"

    # ---------------- helpers ----------------

    def _dtype_from_override(self, override: str) -> Optional[torch.dtype]:
        if override == "fp16":
            return torch.float16
        if override == "bf16":
            return torch.bfloat16
        if override == "fp32":
            return torch.float32
        return None

    def _device_from_choice(self, choice: str):
        if choice == "auto":
            return model_management.text_encoder_device()
        try:
            return torch.device(choice)
        except Exception:
            return model_management.text_encoder_device()

    def _cliptype_from_model_type(self, model_type: str):
        mt = model_type.lower()
        # These enums exist in recent Comfy; for older, STABLE_DIFFUSION covers SDXL
        if hasattr(comfy_sd, "CLIPType"):
            if mt == "flux" and hasattr(comfy_sd.CLIPType, "FLUX"):
                return comfy_sd.CLIPType.FLUX
            if mt == "sd3" and hasattr(comfy_sd.CLIPType, "SD3"):
                return comfy_sd.CLIPType.SD3
            if mt == "hunyuan_dit" and hasattr(comfy_sd.CLIPType, "HUNYUAN_DIT"):
                return comfy_sd.CLIPType.HUNYUAN_DIT
            # SDXL/SD1.5 route through the generic path
            return comfy_sd.CLIPType.STABLE_DIFFUSION
        # Extremely old builds: fall back to a generic path
        return 0  # generic

    def _sanitize_rel(self, name: str) -> str:
        # Some UIs export "t5\t5xxl_fp16.safetensors" (TAB). Normalize that.
        if "\t" in name:
            name = name.replace("\t", "\\t")
        return name

    def _load_state(self, filename: str) -> Tuple[dict, str]:
        filename = self._sanitize_rel(filename)
        full_path = folder_paths.get_full_path("text_encoders", filename)
        if full_path is None:
            # Try searching by basename across known folders
            base = os.path.basename(filename)
            for root in folder_paths.get_folder_paths("text_encoders"):
                candidate = os.path.join(root, base)
                if os.path.exists(candidate):
                    full_path = candidate
                    break
        sd = comfy_utils.load_torch_file(full_path, safe_load=True)
        return sd, full_path

    def _maybe_warmup(self, clip_obj):
        try:
            tokens = clip_obj.tokenize("warmup")
            _ = clip_obj.encode_from_tokens(tokens, return_pooled=True)
        except Exception:
            pass

    # ---------------- main ----------------

    def load(self, type, clip_path, t5_path, mode, device, precision_override, warmup=False):
        diag: List[str] = []
        clip_type = self._cliptype_from_model_type(type)
        load_dev = self._device_from_choice(device)
        offload_dev = model_management.text_encoder_offload_device()
        dtype_override = self._dtype_from_override(precision_override)

        diag.append(f"Model type : {type}")
        diag.append(f"Device     : {str(load_dev)} (offload {str(offload_dev)})")
        diag.append(f"Precision  : {precision_override}")

        names_to_load: List[str] = []
        if mode in ("dual", "clip-only"):
            names_to_load.append(clip_path)
        if mode in ("dual", "t5-only"):
            names_to_load.append(t5_path)

        state_dicts = []
        file_paths = []
        for name in names_to_load:
            try:
                sd, fp = self._load_state(name)
                state_dicts.append(sd)
                file_paths.append(fp)
            except Exception as e:
                diag.append(f"ERROR loading {name}: {repr(e)}")

        if not state_dicts:
            raise RuntimeError("Failed to load any text encoders.\n" + "\n".join(diag))

        model_options = {
            "load_device": load_dev,
            "offload_device": offload_dev,
            "initial_device": load_dev,
        }
        if dtype_override is not None:
            model_options["dtype"] = dtype_override

        try:
            clip_obj = comfy_sd.load_text_encoder_state_dicts(
                state_dicts=state_dicts,
                embedding_directory=(folder_paths.get_folder_paths("embeddings")[0]
                                     if folder_paths.get_folder_paths("embeddings") else None),
                clip_type=clip_type,
                model_options=model_options,
            )
        except TypeError:
            # Older signature compatibility
            clip_obj = comfy_sd.load_text_encoder_state_dicts(
                state_dicts,
                folder_paths.get_folder_paths("embeddings")[0]
                if folder_paths.get_folder_paths("embeddings") else None,
                clip_type,
                model_options,
            )

        if warmup:
            self._maybe_warmup(clip_obj)
            diag.append("Warmup     : done")

        diag.append("Files      : " + " | ".join(file_paths))
        return (clip_obj, "\n".join(diag))


NODE_CLASS_MAPPINGS = {
    "KrakenDualCLIPLoader": KrakenDualCLIPLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KrakenDualCLIPLoader": "Kraken Dual CLIP Loader",
}
