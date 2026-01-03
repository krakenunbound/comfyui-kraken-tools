# -*- coding: utf-8 -*-
# kraken_wan_prompt.py
#
# Kraken WAN Prompt Splitter — ALL options via dropdowns/toggles
# - "prompt" stays a STRING (textbox or cable input).
# - "persistent" stays a STRING (so you can type or drive it).
# - EVERYTHING ELSE is a dropdown (or BOOLEAN).
# - Five STRING outputs (Prompt 1..5).

import re

# ------------- Style & pack choices -------------
STYLE_CHOICES = {
    "shot_type": [
        "None","extreme close-up","close-up","medium shot","cowboy shot",
        "wide shot","extreme wide shot","over-the-shoulder","POV","aerial","top-down"
    ],
    "lens": [
        "None","24mm","28mm","35mm","50mm","85mm","105mm macro","135mm","200mm",
        "anamorphic","tilt-shift","fisheye"
    ],
    "aperture": [
        "None","f/1.4 dreamy bokeh","f/2.8 shallow depth","f/5.6 balanced","f/8 sharp","f/16 deep focus"
    ],
    "lighting": [
        "None","natural soft daylight","golden hour rim light","overcast diffused","hard noon sun",
        "tungsten practicals","fluorescent ambience","neon mix light","studio 3-point",
        "backlit silhouette","candlelight","moonlit blue hour","high-key","low-key film noir"
    ],
    "film": [
        "None","photoreal neutral","portra-like","ektar-like","provia-like","velvia-like",
        "cinestill-like","black-and-white","cross-processed","bleach-bypass","vintage chrome","instant film"
    ],
    "grade": [
        "None","warm amber teal","cool cyan steel","muted pastels","earthy natural","high contrast",
        "soft cinematic fade","desaturated documentary","vibrant pop","monochrome sepia","monochrome blue"
    ],
    "composition": [
        "None","rule of thirds","centered symmetry","leading lines","dutch angle","deep perspective",
        "minimal negative space","foreground framing","reflections","atmospheric haze"
    ],
    "environment": [
        "None","clear","overcast","fog","light rain","heavy rain","snowfall","dust storm","haze"
    ],
    "texture": [
        "None","ultra-detailed","fine film grain","clean minimal","matte finish","glossy wet","gritty texture"
    ],
}

WAN_PACKS = [
    "None",
    "Cinematic Natural",
    "Moody Neon",
    "Documentary Daylight",
    "Epic Vista",
    "Noir Classic",
]

SEPARATOR_CHOICES = [
    ", ", " | ", " · ", " — ", " – ", " / ", " ; ", " : "
]

DELIMITER_CHOICES = [
    "---", "###", "***", "~~~", "|||"
]

PERSISTENT_MODE_CHOICES = ["append", "prepend"]

def _clean_parts(parts, strip_whitespace, keep_empty):
    out = []
    for p in parts:
        if strip_whitespace and isinstance(p, str):
            p = p.strip()
        if p or keep_empty:
            out.append(p)
    return out

def _merge_persistent(base, persistent, sep, mode):  # mode: "append" or "prepend"
    base = (base or "").strip()
    persistent = (persistent or "").strip()
    if not base and not persistent:
        return ""
    if not base:
        return persistent
    if not persistent:
        return base
    if mode == "prepend":
        return f"{persistent}{sep}{base}"
    return f"{base}{sep}{persistent}"

def _build_style_tail(sel: dict, extra_choice: str) -> str:
    tokens = []
    for k in ["shot_type","lens","aperture","lighting","film","grade","composition","environment","texture"]:
        v = sel.get(k, "None")
        if v and v != "None":
            tokens.append(v)
    if extra_choice and extra_choice != "None":
        tokens.append(extra_choice)
    return ", ".join(tokens)

# extra style add-on (dropdown, not free text)
STYLE_EXTRA_CHOICES = [
    "None",
    "soft cinematic fade",
    "high contrast punch",
    "desaturated documentary",
    "muted pastels",
    "earthy natural",
    "fine film grain",
    "atmospheric haze",
    "glossy highlights",
    "matte finish",
]

# WAN pack defaults
WAN_PACK_MAP = {
    "Cinematic Natural": {
        "shot_type":"wide shot","lens":"35mm","aperture":"f/2.8 shallow depth",
        "lighting":"golden hour rim light","film":"portra-like","grade":"soft cinematic fade",
        "composition":"leading lines","environment":"haze","texture":"ultra-detailed"
    },
    "Moody Neon": {
        "shot_type":"medium shot","lens":"50mm","aperture":"f/1.4 dreamy bokeh",
        "lighting":"neon mix light","film":"cinestill-like","grade":"high contrast",
        "composition":"centered symmetry","environment":"light rain","texture":"gritty texture"
    },
    "Documentary Daylight": {
        "shot_type":"medium shot","lens":"28mm","aperture":"f/8 sharp",
        "lighting":"natural soft daylight","film":"photoreal neutral","grade":"desaturated documentary",
        "composition":"rule of thirds","environment":"overcast","texture":"fine film grain"
    },
    "Epic Vista": {
        "shot_type":"extreme wide shot","lens":"24mm","aperture":"f/8 sharp",
        "lighting":"golden hour rim light","film":"velvia-like","grade":"vibrant pop",
        "composition":"deep perspective","environment":"fog","texture":"ultra-detailed"
    },
    "Noir Classic": {
        "shot_type":"close-up","lens":"85mm","aperture":"f/2.8 shallow depth",
        "lighting":"low-key film noir","film":"black-and-white","grade":"high contrast",
        "composition":"dutch angle","environment":"haze","texture":"matte finish"
    },
}

class KrakenWanPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        placeholder_text = (
            'Place a prompt here. Put three dashes "---" on their own line to split into multiple prompts.'
        )
        return {
            "required": {
                # Must remain a textbox so users can paste large blocks or wire from upstream.
                "prompt": ("STRING", { "multiline": True, "default": "", "placeholder": placeholder_text }),
            },
            "optional": {
                # Persistent text remains a STRING (needs to accept arbitrary tags or be wired in).
                "persistent": ("STRING", {"multiline": True, "default": ""}),
                "combine_persistent": ("BOOLEAN", {"default": False}),
                "persistent_mode": (PERSISTENT_MODE_CHOICES,),    # dropdown
                "separator": (SEPARATOR_CHOICES,),                # dropdown

                # Split behavior
                "delimiter": (DELIMITER_CHOICES,),                # dropdown
                "strip_whitespace": ("BOOLEAN", {"default": True}),
                "keep_empty": ("BOOLEAN", {"default": False}),

                # WAN pack + style dropdowns
                "wan_pack": (WAN_PACKS,),                         # dropdown
                "apply_pack": ("BOOLEAN", {"default": True}),

                "shot_type":   (STYLE_CHOICES["shot_type"],),     # dropdown
                "lens":        (STYLE_CHOICES["lens"],),          # dropdown
                "aperture":    (STYLE_CHOICES["aperture"],),      # dropdown
                "lighting":    (STYLE_CHOICES["lighting"],),      # dropdown
                "film":        (STYLE_CHOICES["film"],),          # dropdown
                "grade":       (STYLE_CHOICES["grade"],),         # dropdown
                "composition": (STYLE_CHOICES["composition"],),   # dropdown
                "environment": (STYLE_CHOICES["environment"],),   # dropdown
                "texture":     (STYLE_CHOICES["texture"],),       # dropdown

                # Extra style tag as a dropdown (no free text)
                "style_extra": (STYLE_EXTRA_CHOICES,),            # dropdown
                "combine_style_presets": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4", "Prompt 5")
    FUNCTION = "split"
    CATEGORY = "🦑 Kraken / Prompt"
    OUTPUT_NODE = False

    def _split_text(self, text, delimiter_choice):
        delim = (delimiter_choice or "---").strip()
        if not delim:
            delim = "---"
        pattern = r"(?m)^\s*{}\s*$".format(re.escape(delim))
        if re.search(pattern, text or ""):
            return re.split(pattern, text or "")
        return (text or "").split(delim)

    def split(
        self,
        prompt="",
        persistent="",
        combine_persistent=False,
        persistent_mode="append",
        separator=", ",
        delimiter="---",
        strip_whitespace=True,
        keep_empty=False,
        wan_pack="None",
        apply_pack=True,
        shot_type="wide shot",
        lens="35mm",
        aperture="f/2.8 shallow depth",
        lighting="natural soft daylight",
        film="photoreal neutral",
        grade="soft cinematic fade",
        composition="rule of thirds",
        environment="clear",
        texture="ultra-detailed",
        style_extra="None",
        combine_style_presets=True,
    ):
        # Apply WAN pack if chosen
        if apply_pack and wan_pack in WAN_PACK_MAP:
            pack = WAN_PACK_MAP[wan_pack]
            shot_type   = pack.get("shot_type", shot_type)
            lens        = pack.get("lens", lens)
            aperture    = pack.get("aperture", aperture)
            lighting    = pack.get("lighting", lighting)
            film        = pack.get("film", film)
            grade       = pack.get("grade", grade)
            composition = pack.get("composition", composition)
            environment = pack.get("environment", environment)
            texture     = pack.get("texture", texture)

        # Split & clean
        parts = self._split_text(prompt or "", delimiter_choice=delimiter)
        parts = _clean_parts(parts, strip_whitespace=strip_whitespace, keep_empty=keep_empty)

        # Persistent (prepend/append using dropdown mode and dropdown separator)
        if combine_persistent and (persistent or "").strip():
            parts = [_merge_persistent(p, persistent, separator, persistent_mode) for p in parts]

        # Style tail (always appended using same dropdown separator)
        if combine_style_presets:
            sel = dict(
                shot_type=shot_type, lens=lens, aperture=aperture, lighting=lighting,
                film=film, grade=grade, composition=composition, environment=environment, texture=texture
            )
            tail = _build_style_tail(sel, style_extra)
            if tail:
                parts = [f"{(p or '').strip()}{separator}{tail}" if (p or "").strip() else tail for p in parts]

        # pad / cap to 5
        parts = (parts + ["", "", "", "", ""])[:5]
        return tuple(parts)
