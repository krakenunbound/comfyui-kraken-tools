# KrakenLoras3 — 3-slot LoRA loader with readable labels, CivitAI trigger words,
# strength parsing, background API key loading, and PER-LORA CLIP Skip.

from __future__ import annotations
import os
import json
import re
import hashlib
from typing import List, Tuple

import requests
import folder_paths
# We now import CLIPSetLastLayer to handle the skipping internally
from nodes import LoraLoader, CLIPSetLastLayer

# -------- utilities ---------------------------------------------------------

def _safe_choices_loras() -> List[str]:
    try:
        return ["None"] + sorted(folder_paths.get_filename_list("loras"))
    except Exception:
        return ["None"]

def _stem(fname: str) -> str:
    name = os.path.splitext(os.path.basename(fname))[0]
    return re.sub(r"[_-]+", " ", name).strip()

def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in items:
        s2 = s.strip().strip(",")
        if s2 and s2.lower() not in seen:
            seen.add(s2.lower())
            out.append(s2)
    return out

def calculate_sha256(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def get_metadata(filepath: str, type: str) -> dict:
    filepath = folder_paths.get_full_path(type, filepath)
    with open(filepath, "rb") as file:
        header_size = int.from_bytes(file.read(8), "little", signed=False)
        if header_size <= 0:
            return None
        header = file.read(header_size)
        header_json = json.loads(header)
        return header_json.get("__metadata__", None)

def sort_tags_by_frequency(meta_tags: dict) -> List[str]:
    if not meta_tags or "ss_tag_frequency" not in meta_tags:
        return []
    meta_tags = json.loads(meta_tags["ss_tag_frequency"])
    sorted_tags = {}
    for dataset in meta_tags.values():
        for tag, count in dataset.items():
            tag = str(tag).strip()
            sorted_tags[tag] = sorted_tags.get(tag, 0) + count
    sorted_tags = dict(sorted(sorted_tags.items(), key=lambda item: item[1], reverse=True))
    return list(sorted_tags.keys())

# -------- API key storage ---------------------------------------------------

CONFIG_PATH = os.path.join(os.path.expanduser("~"), "Documents", "ComfyUI", "user", "kraken_config.json")

def load_api_key() -> str:
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
                return config.get("civitai_api_key", "")
        return ""
    except Exception as e:
        print(f"[KrakenLoras3] Failed to load API key from {CONFIG_PATH}: {str(e)}")
        return ""

# ---- CivitAI triggers and strengths (best-effort; works without API key for public data) --

CIVITAI_MODEL_VERSION_BY_HASH = "https://civitai.com/api/v1/model-versions/by-hash"
TAGS_CACHE_PATH = os.path.join(folder_paths.get_folder_paths("loras")[0], "loras_tags.json")

def civitai_fetch_triggers(lora_name: str, api_key: str | None = None, force_fetch: bool = False) -> Tuple[List[str], List[str], float, float]:
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path:
        return [], [], 1.0, 1.0
    try:
        with open(TAGS_CACHE_PATH, 'r') as f:
            lora_tags = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        lora_tags = {}
    cached_data = lora_tags.get(lora_name, None)
    if not force_fetch and cached_data:
        return (cached_data.get("trigger_words", []), cached_data.get("tags", []), cached_data.get("model_strength", 1.0), cached_data.get("clip_strength", 1.0))
    
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        lora_hash = calculate_sha256(lora_path)
        r = requests.get(f"{CIVITAI_MODEL_VERSION_BY_HASH}/{lora_hash}", headers=headers, timeout=10)
        r.raise_for_status()
        model_info = r.json()
    except Exception as e:
        model_info = None
    
    if model_info:
        trigger_words = model_info.get("trainedWords", [])
        tags = model_info.get("tags", [])
        description = model_info.get("description", "") or ""
        model_strength, clip_strength = 1.0, 1.0
        weight_match = re.search(r"(?:weight|strength)\s*[:=]\s*(\d*\.?\d+)(?:-(\d*\.?\d+))?", description, re.IGNORECASE)
        if weight_match:
            model_strength = float(weight_match.group(1))
            clip_strength = float(weight_match.group(2) or model_strength)
        lora_tags[lora_name] = {"trigger_words": trigger_words, "tags": tags, "model_strength": model_strength, "clip_strength": clip_strength}
        with open(TAGS_CACHE_PATH, 'w') as f:
            json.dump(lora_tags, f, indent=4)
    else:
        meta_tags = get_metadata(lora_name, "loras")
        trigger_words = sort_tags_by_frequency(meta_tags)
        tags = []
        model_strength, clip_strength = 1.0, 1.0
        
    return trigger_words, tags, model_strength, clip_strength

# ------------ Node ----------------------------------------------------------

class KrakenLoras3:
    @classmethod
    def INPUT_TYPES(cls):
        lora_choices = _safe_choices_loras()

        if load_api_key():
            api_key_status = "✅ Key Loaded from File"
        else:
            api_key_status = "⚠️ Key Not Found"

        required = {
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "lora_1_enabled": ("BOOLEAN", {"default": True, "label": "LoRA 1 — Enabled"}),
            "lora_1_file": (lora_choices, {"default": "None", "label": "LoRA 1 — File"}),
            "lora_1_clip_skip": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1, "label": "LoRA 1 — CLIP Skip (0=Off)"}),
            "lora_1_model_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05, "label": "LoRA 1 — Model Strength"}),
            "lora_1_clip_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05, "label": "LoRA 1 — CLIP Strength"}),
            "lora_1_force_fetch": ("BOOLEAN", {"default": False, "label": "LoRA 1 — Force Fetch"}),
            
            "lora_2_enabled": ("BOOLEAN", {"default": False, "label": "LoRA 2 — Enabled"}),
            "lora_2_file": (lora_choices, {"default": "None", "label": "LoRA 2 — File"}),
            "lora_2_clip_skip": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1, "label": "LoRA 2 — CLIP Skip (0=Off)"}),
            "lora_2_model_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05, "label": "LoRA 2 — Model Strength"}),
            "lora_2_clip_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05, "label": "LoRA 2 — CLIP Strength"}),
            "lora_2_force_fetch": ("BOOLEAN", {"default": False, "label": "LoRA 2 — Force Fetch"}),
            
            "lora_3_enabled": ("BOOLEAN", {"default": False, "label": "LoRA 3 — Enabled"}),
            "lora_3_file": (lora_choices, {"default": "None", "label": "LoRA 3 — File"}),
            "lora_3_clip_skip": ("INT", {"default": 0, "min": 0, "max": 4, "step": 1, "label": "LoRA 3 — CLIP Skip (0=Off)"}),
            "lora_3_model_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05, "label": "LoRA 3 — Model Strength"}),
            "lora_3_clip_strength": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.05, "label": "LoRA 3 — CLIP Strength"}),
            "lora_3_force_fetch": ("BOOLEAN", {"default": False, "label": "LoRA 3 — Force Fetch"}),
            
            "placement": (["prepend", "append", "replace"], {"default": "prepend"}),
            # --- CORRECTED SECTION ---
            # The definitions for separator and api_key_status are now correct.
            "separator": ("STRING", {"default": ", ", "label": "Separator"}),
            "fetch_triggers": ("BOOLEAN", {"default": True, "label": "Fetch Trigger Words"}),
            "api_key_status": ("STRING", {"default": api_key_status, "label": "CivitAI API Key Status"}),
        }
        
        optional = {
            "positive_prompt": ("STRING", {"forceInput": True}),
        }
        
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING_LIST", "STRING_LIST", "STRING_LIST")
    RETURN_NAMES = ("model", "clip", "prompt", "lora_names", "triggers", "tags")
    FUNCTION = "apply"
    CATEGORY = "Kraken/LoRA"
    
    def _apply_one(self, model, clip, file: str, sm: float, sc: float):
        if not file or file == "None":
            return model, clip
        loader = LoraLoader()
        model, clip = loader.load_lora(model, clip, file, sm, sc)
        return model, clip

    def apply(self, model, clip, 
              lora_1_enabled, lora_1_file, lora_1_clip_skip, lora_1_model_strength, lora_1_clip_strength, lora_1_force_fetch, 
              lora_2_enabled, lora_2_file, lora_2_clip_skip, lora_2_model_strength, lora_2_clip_strength, lora_2_force_fetch, 
              lora_3_enabled, lora_3_file, lora_3_clip_skip, lora_3_model_strength, lora_3_clip_strength, lora_3_force_fetch, 
              placement, separator, fetch_triggers, api_key_status, positive_prompt=""):
        
        civitai_api_key = load_api_key()
        
        triggers, tags, lora_names = [], [], []
        
        lora_slots = [
            (lora_1_enabled, lora_1_file, lora_1_clip_skip, lora_1_model_strength, lora_1_clip_strength, lora_1_force_fetch),
            (lora_2_enabled, lora_2_file, lora_2_clip_skip, lora_2_model_strength, lora_2_clip_strength, lora_2_force_fetch),
            (lora_3_enabled, lora_3_file, lora_3_clip_skip, lora_3_model_strength, lora_3_clip_strength, lora_3_force_fetch),
        ]
        
        for en, fn, cs, sm, sc, ff in lora_slots:
            if en and fn and fn != "None":
                if fetch_triggers:
                    trigger_words, tag_list, rec_model_strength, rec_clip_strength = civitai_fetch_triggers(fn, api_key=civitai_api_key or None, force_fetch=ff)
                    triggers.extend(trigger_words)
                    tags.extend(tag_list)
                    if sm == 1.0: sm = rec_model_strength
                    if sc == 1.0: sc = rec_clip_strength
                
                model, clip = self._apply_one(model, clip, fn, sm, sc)
                lora_names.append(_stem(fn))
                
                if cs > 0:
                    clip = CLIPSetLastLayer().set_last_layer(clip, -cs)[0]
        
        trigger_str = separator.join([t for t in _dedupe_keep_order(triggers) if t])
        positive_prompt = positive_prompt.strip().strip(",")
        if not trigger_str:
            final_prompt = positive_prompt
        elif placement == "replace":
            final_prompt = trigger_str
        elif placement == "append":
            final_prompt = f"{positive_prompt}{separator if positive_prompt else ''}{trigger_str}".strip().strip(",")
        else: # prepend
            final_prompt = f"{trigger_str}{separator if positive_prompt else ''}{positive_prompt}".strip().strip(",")
        
        return (model, clip, final_prompt, lora_names, triggers, tags)

# ---- node registration (for ComfyUI automatic discovery) -------------------

NODE_CLASS_MAPPINGS = {
    "KrakenLoras3": KrakenLoras3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KrakenLoras3": "KrakenLoras3",
}