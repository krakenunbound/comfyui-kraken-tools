# -*- coding: utf-8 -*-
"""
Kraken Unbound Prompt — v3.2
WHAT'S NEW (v3.2)
- **VRAM controls:** keep_alive_minutes (0–10) + force_unload after run.
- **Character budget:** optional enforce_char_limit + char_limit with custom StoppingCriteria and post-trim on sentence boundary.
- **Vision mode (optional):** accept an image input and use Qwen2-VL-2B to caption the image, then feed that text into the stylistic rewriter.
- **Prepend/Append:** Added position dropdown to place style, lighting, lens, f-stop, bokeh, DSLR at start or end of prompt.
- **Streamlined lists:** Reduced style, lighting, lens, f-stop options; added manga, isometric, noir, black & white; removed film_grain, hdr.
DESIGN GOALS
- Minimal VRAM footprint on 8GB GPUs (e.g., 3060 Ti): on-demand loading, quick unloads, no quantization required.
- Support short, modern NL prompts **or** compact tag-lines for SD1.5/anime.
- Easy to swap models without code edits.
"""
from typing import List
import warnings
import torch
from transformers import (
    pipeline,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
    Qwen2VLForConditionalGeneration,
)
import time, gc
from PIL import Image
import re

def _join(parts: List[str], sep: str) -> str:
    parts = [p.strip() for p in parts if isinstance(p, str) and p.strip()]
    return (sep if sep is not None else ", ").join(parts)

class CharLimitCriteria(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len_tokens: int, char_limit: int):
        self.tokenizer = tokenizer
        self.prompt_len_tokens = prompt_len_tokens
        self.char_limit = int(max(0, char_limit))

    def __call__(self, input_ids, scores, **kwargs):
        if self.char_limit <= 0:
            return False
        gen_ids = input_ids[0][self.prompt_len_tokens:]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return len(text) >= self.char_limit

class KrakenUnboundPrompt:
    """
    ComfyUI node that builds a well-structured **text prompt** for T2I.
    Uses Qwen2-VL-2B-Instruct for both:
      1) Image-to-text captioning (if use_image_as_source=True).
      2) Text prompt enhancement (if enhance_prompt=True).
    Memory policy: models load **on demand**, and can be unloaded immediately
    after a run (keep_alive_minutes=0) or cached for a short time window.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "use_image_as_source": ("BOOLEAN", {"default": False}),
                "input_image": ("IMAGE", {}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "model_name": (["Qwen/Qwen2-VL-2B-Instruct"], {"default": "Qwen/Qwen2-VL-2B-Instruct"}),
                "enhancer_style": ([
                    "Modern (SDXL / SD3 / Flux)",
                    "Classic Tags (SD 1.5 / Anime)",
                    "Instructional (PixArt / Qwen)",
                    "WAN (Word As Neyword)",
                ],),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "max": 1.0, "step": 0.05}),
                "max_new_tokens": ("INT", {"default": 256, "min": 16, "max": 1024, "step": 8}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "enforce_char_limit": ("BOOLEAN", {"default": False}),
                "char_limit": ("INT", {"default": 512, "min": 32, "max": 2000, "step": 8}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "persistent": ("STRING", {"multiline": True, "default": ""}),
                "combine_persistent": ("BOOLEAN", {"default": True}),
                "style": ([
                    "none", "photorealistic", "cinematic", "anime", "manga", "fantasy",
                    "sci-fi", "comic book art", "cartoon", "watercolor painting",
                    "oil painting", "noir", "black & white", "isometric"
                ],),
                "lighting": ([
                    "none", "soft lighting", "dramatic lighting", "cinematic lighting",
                    "studio lighting", "natural lighting", "golden hour lighting",
                    "neon lighting", "rim lighting", "backlighting"
                ],),
                "camera_lens": ([
                    "none", "35mm", "50mm", "85mm portrait", "fisheye lens", "tilt-shift lens"
                ],),
                "f_stop": ([
                    "none", "f/1.8", "f/2.8", "f/4", "f/8", "f/16"
                ],),
                "bokeh": ("BOOLEAN", {"default": False}),
                "dslr": ("BOOLEAN", {"default": False}),
                "separator": ("STRING", {"default": ", "}),
                "prefix": ("STRING", {"default": ""}),
                "suffix": ("STRING", {"default": ""}),
                "show_section_labels": ("BOOLEAN", {"default": False}),
                "apply_to_negative": ("BOOLEAN", {"default": False}),
                "position": (["append", "prepend"], {"default": "append"}),
                "keep_alive_minutes": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "force_unload": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt", "persistent")
    FUNCTION = "build"
    CATEGORY = "🦑 Kraken / Prompt"
    OUTPUT_NODE = False

    def __init__(self):
        self.pipeline = None
        self._last_model_name = None
        self._has_cuda = torch.cuda.is_available()
        self._last_used_ts = 0.0
        self._keep_alive_minutes = 3

    def _polish_caption(self, caption: str) -> str:
        if not caption:
            return caption
        if not re.search(r'[.!?]$', caption):
            sentences = re.split(r'[.!?]+', caption)
            if len(sentences) > 1:
                caption = '. '.join(sentences[:-1]) + '.'
            if len(caption) > 100 and ',' in caption:
                last_comma = caption.rfind(',')
                if last_comma > 50:
                    caption = caption[:last_comma + 1].rstrip()
        return caption.strip()

    def _label_if(self, label: str, value: str, show: bool) -> str:
        v = (value or "").strip()
        return f"{label}: {v}" if show and v and v != "none" else (v if v and v != "none" else "")

    def _assemble_core(
        self,
        base_text,
        style,
        lighting,
        camera_lens,
        f_stop,
        bokeh,
        dslr,
        separator,
        prefix,
        suffix,
        show_section_labels,
        position,
    ):
        parts = [
            self._label_if("style", style, show_section_labels),
            self._label_if("lighting", lighting, show_section_labels),
            self._label_if("lens", camera_lens, show_section_labels),
            self._label_if("aperture", f_stop, show_section_labels),
            "bokeh" if bokeh else "",
            "DSLR" if dslr else "",
        ]
        parts = [p for p in parts if p]
        if position == "prepend":
            combined = _join(parts + [base_text], separator)
        else:
            combined = _join([base_text] + parts, separator)
        if prefix:
            combined = prefix + combined
        if suffix:
            combined = combined + suffix
        return combined

    def _unload_model(self):
        try:
            pipe = self.pipeline
            if pipe is not None:
                try:
                    model = getattr(pipe, 'model', None)
                    tok = getattr(pipe, 'tokenizer', None)
                    del pipe
                    if model is not None:
                        del model
                    if tok is not None:
                        del tok
                except Exception:
                    pass
        finally:
            self.pipeline = None
            self._last_model_name = None
            if self._has_cuda:
                torch.cuda.empty_cache()
            gc.collect()

    def _load_model(self, model_name: str):
        if (self.pipeline is not None) and (self._last_model_name == model_name):
            return self.pipeline
        print(f"[Kraken Unbound Prompt] Loading model: {model_name} ...")
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self._has_cuda else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True,
            )
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=processor.tokenizer,
                processor=processor,
                device_map="auto",
            )
            self._last_model_name = model_name
            print("[Kraken Unbound Prompt] Model loaded successfully.")
        except Exception as e:
            print(f"[Kraken Unbound Prompt] ERROR: Failed to load model: {e}")
            print("[Kraken Tip] Ensure model is downloaded to 'ComfyUI/models/llm/' or available on Hugging Face. Try: pip install --upgrade transformers")
            self.pipeline = e
        return self.pipeline

    def _maybe_expire(self):
        if self.pipeline is None:
            return
        if self._keep_alive_minutes <= 0:
            self._unload_model()
            return
        idle = (time.time() - self._last_used_ts) / 60.0
        if idle >= self._keep_alive_minutes:
            self._unload_model()

    def build(
        self,
        prompt="",
        use_image_as_source=False,
        input_image=None,
        enhance_prompt=False,
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        enhancer_style="Modern (SDXL / SD3 / Flux)",
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=256,
        do_sample=True,
        negative_prompt="",
        persistent="",
        combine_persistent=True,
        style="none",
        lighting="none",
        camera_lens="none",
        f_stop="none",
        bokeh=False,
        dslr=False,
        separator=", ",
        prefix="",
        suffix="",
        show_section_labels=False,
        apply_to_negative=False,
        position="append",
        enforce_char_limit=False,
        char_limit=512,
        keep_alive_minutes=3,
        force_unload=True,
    ):
        try:
            self._keep_alive_minutes = int(max(0, min(10, keep_alive_minutes)))
        except Exception:
            self._keep_alive_minutes = 3
        self._maybe_expire()

        source_content = (prompt or "").strip()
        if use_image_as_source and input_image is not None:
            print(f"[Kraken Unbound Prompt] Vision mode activated: Processing image with shape {getattr(input_image, 'shape', 'N/A')}")
            try:
                pipeline = self._load_model(model_name)
                if isinstance(pipeline, Exception):
                    return (f"VISION ERROR: Model failed to load. See console. | Type: {type(pipeline).__name__}", negative_prompt, persistent)
                processor = pipeline.processor
                if not isinstance(input_image, Image.Image):
                    if isinstance(input_image, torch.Tensor):
                        print(f"[Kraken Unbound Prompt] Converting tensor: original shape {input_image.shape}")
                        if len(input_image.shape) == 4:
                            input_image = input_image[0]
                        if input_image.shape[-1] == 4:
                            input_image = input_image[:, :, :3]
                        if input_image.dtype != torch.uint8:
                            input_image = (input_image * 255).clamp(0, 255).byte()
                        pil_img = Image.fromarray(input_image.numpy())
                        print(f"[Kraken Unbound Prompt] Converted to PIL: {pil_img.size}")
                    else:
                        raise ValueError("Unsupported image type; please provide a torch.Tensor (ComfyUI IMAGE) or PIL.Image.")
                else:
                    pil_img = input_image
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "Describe the image in 1-2 complete sentences, ending with a period. Focus on salient visual details."},
                        ],
                    }
                ]
                text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(text=[text_prompt], images=[pil_img], padding=True, return_tensors="pt")
                inputs = inputs.to(pipeline.device)
                with torch.inference_mode():
                    gen_kwargs = {
                        "max_new_tokens": 128,
                        "eos_token_id": processor.tokenizer.eos_token_id,
                        "pad_token_id": processor.tokenizer.pad_token_id,
                        "do_sample": True,
                        "temperature": 0.3,
                    }
                    output_ids = pipeline.model.generate(**inputs, **gen_kwargs)
                generated_ids = [
                    output_ids[i, inputs.input_ids[i].shape[0]:]
                    for i in range(output_ids.shape[0])
                ]
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                caption = self._polish_caption(caption)
                source_content = caption.strip()
                print(f"[Kraken Unbound Prompt] VLM caption generated: '{source_content}'")
                print(f"[Kraken Unbound Prompt] Original typed prompt ignored: '{prompt}'")
            except Exception as e:
                print(f"[Kraken Unbound Prompt] WARNING: VLM failed ({type(e).__name__}): {e}. Falling back to typed prompt.")
                print("[Kraken Tip] Ensure model is downloaded to 'ComfyUI/models/llm/' or available on Hugging Face. Try: pip install --upgrade transformers")
                source_content = (prompt or "").strip()

        if enhance_prompt and source_content:
            if not source_content:
                return ("ERROR: Please provide a non-empty input text or image for enhancement.", negative_prompt, persistent)
            pipeline = self._load_model(model_name)
            if isinstance(pipeline, Exception):
                return (f"ENHANCER ERROR: Model failed to load. See console. | Type: {type(pipeline).__name__}", negative_prompt, persistent)
            processor = pipeline.processor
            style_instructions = {
                "Modern (SDXL / SD3 / Flux)": (
                    "Rewrite the following concept into a single, highly descriptive and visually rich prompt for a modern text-to-image AI."
                    " Focus on composition, lighting, textures, and camera language. Do not add commentary or disclaimers."
                ),
                "Classic Tags (SD 1.5 / Anime)": (
                    "Convert the following concept into a comma-separated list of high-signal tags in danbooru style."
                    " Include quality tags like 'masterpiece, best quality'. Keep it one line."
                ),
                "Instructional (PixArt / Qwen)": (
                    "Express the following concept as one concise imperative sentence that fully describes the scene."
                    " Avoid extra words, no meta text."
                ),
                "WAN (Word As Neyword)": (
                    "Break the concept into core components separated by '||' for the WAN technique."
                    " Keep components terse and visual."
                ),
            }
            instruction = style_instructions.get(
                enhancer_style, style_instructions["Modern (SDXL / SD3 / Flux)"]
            )
            if enforce_char_limit and char_limit > 0:
                instruction += f" Limit the final output to approximately {int(char_limit)} characters. Keep it concise and end cleanly."
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{instruction}\nConcept: '{source_content}'"}
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=[text_prompt], padding=True, return_tensors="pt")
            inputs = inputs.to(pipeline.device)
            print(
                f"[Kraken Unbound Prompt] Enhancing source with style: {enhancer_style} using {model_name}..."
            )
            with torch.inference_mode():
                gen_kwargs = dict(
                    max_new_tokens=int(max_new_tokens),
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    repetition_penalty=1.12,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
                stopping = None
                if enforce_char_limit and char_limit > 0:
                    prompt_len_tokens = len(processor.tokenizer(text_prompt, return_tensors="pt").input_ids[0])
                    stopping = StoppingCriteriaList([CharLimitCriteria(processor.tokenizer, prompt_len_tokens, int(char_limit))])
                    gen_kwargs["stopping_criteria"] = stopping
                output_ids = pipeline.model.generate(**inputs, **gen_kwargs)
            generated_ids = [
                output_ids[i, inputs.input_ids[i].shape[0]:]
                for i in range(output_ids.shape[0])
            ]
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            source_content = generated_text.strip()
            if enforce_char_limit and char_limit > 0 and len(source_content) > char_limit:
                trimmed = source_content[:int(char_limit)]
                cut = max(trimmed.rfind("."), trimmed.rfind("!"), trimmed.rfind("?"), trimmed.rfind(","))
                if cut > 40:
                    trimmed = trimmed[:cut+1]
                source_content = trimmed.rstrip()
            print(f"[Kraken Unbound Prompt] Enhanced source: '{source_content}'")
            self._last_used_ts = time.time()

        base_prompt = source_content
        if combine_persistent and persistent.strip():
            base_prompt = _join([base_prompt, persistent], separator)
            print(f"[Kraken Unbound Prompt] Combined with persistent: '{persistent}'")

        out_prompt = self._assemble_core(
            base_prompt,
            style,
            lighting,
            camera_lens,
            f_stop,
            bokeh,
            dslr,
            separator,
            prefix,
            suffix,
            show_section_labels,
            position,
        )
        out_negative = (
            self._assemble_core(
                negative_prompt,
                style,
                lighting,
                camera_lens,
                f_stop,
                bokeh,
                dslr,
                separator,
                prefix,
                suffix,
                show_section_labels,
                position,
            )
            if apply_to_negative
            else (negative_prompt or "")
        )
        out_persistent = persistent or ""
        if force_unload or self._keep_alive_minutes == 0:
            self._unload_model()
        return (out_prompt, out_negative, out_persistent)

# Keep your existing registration line exactly as you already have it.
# e.g. _safe_register("kraken_unbound_prompt", "KrakenUnboundPrompt", "🦑 Kraken Unbound Prompt")