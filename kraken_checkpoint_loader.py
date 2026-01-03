# Place this file in your custom_nodes package alongside your other nodes.

import os
import hashlib
import torch
import folder_paths
import comfy.sd

# simple cache so multi-GB files aren’t re-hashed every run
_HASH_CACHE = {}  # key: (path, mtime) -> sha256[:10]

def _sha256_first10(path: str) -> str:
    try:
        st = os.stat(path)
        k = (path, st.st_mtime)
        if k in _HASH_CACHE:
            return _HASH_CACHE[k]
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(chunk)
        short = h.hexdigest()[:10]
        _HASH_CACHE[k] = short
        return short
    except Exception:
        return ""  # never block the graph just because hashing failed

class KrakenCheckpointLoader:
    """
    Loads MODEL/CLIP/VAE from a chosen checkpoint and also outputs:
      - model_name: filename as selected in the UI
      - ckpt_sha10: first 10 chars of the file's SHA-256
    """
    CATEGORY = "Kraken/Loaders"
    FUNCTION = "load"

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING", "STRING")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "model_name", "ckpt_sha10")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "weight_dtype": (["default", "fp16", "bf16", "fp32"],),
                "compute_hash": ("BOOLEAN", {"default": True}),
            }
        }

    @staticmethod
    def _dtype(choice: str):
        return {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }.get(choice, None)  # None => let Comfy choose best dtype

    def load(self, ckpt_name, weight_dtype="default", compute_hash=True):
        # Resolve to an absolute path (raises if missing)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)

        model_options = {}
        dt = self._dtype(weight_dtype)
        if dt is not None:
            model_options["dtype"] = dt

        # Let Comfy figure out config and wire CLIP/VAE
        outputs = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            model_options=model_options,
        )
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
            raise RuntimeError("Unexpected output from load_checkpoint_guess_config")

        model, clip, vae = outputs[0], outputs[1], outputs[2]
        sha10 = _sha256_first10(ckpt_path) if compute_hash else ""
        return (model, clip, vae, ckpt_name, sha10)

NODE_CLASS_MAPPINGS = {
    "kraken_checkpoint_loader": KrakenCheckpointLoader,
}
