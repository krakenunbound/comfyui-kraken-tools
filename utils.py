
import hashlib
import json
import os

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

CIVITAI_API = "https://civitai.com/api/v1"


def calculate_sha256(path: str) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest().upper()
    except Exception:
        return None


def get_model_version_info(file_hash: str, token: str | None = None) -> dict | None:
    """Look up Civitai model version by file hash; returns JSON or None."""
    if not file_hash or not requests:
        return None
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        url = f"{CIVITAI_API}/model-versions/by-hash/{file_hash}"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def get_metadata(path: str) -> dict | None:
    """Return safetensors metadata dict if available, else None."""
    if not path or not os.path.isfile(path) or not safe_open:
        return None
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.metadata() or {}
    except Exception:
        return None


def sort_tags_by_frequency(meta_tags: dict | None) -> list[str]:
    """
    Parse Kohya-style 'ss_tag_frequency' in metadata and return tags sorted by total count.
    Falls back to common fields if available.
    """
    if not meta_tags:
        return []
    # Kohya format: "ss_tag_frequency" is a JSON string of {dataset: {tag: count, ...}, ...}
    if "ss_tag_frequency" in meta_tags:
        try:
            freq = json.loads(meta_tags["ss_tag_frequency"])
        except Exception:
            freq = {}
        counts = {}
        for _, dataset in (freq or {}).items():
            for tag, count in (dataset or {}).items():
                tag = str(tag).strip()
                counts[tag] = counts.get(tag, 0) + int(count or 0)
        # sort by count desc
        return [t for t, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    # Other common keys
    for k in ["ss_tag_names", "tag_frequency", "tags", "trainedWords"]:
        v = meta_tags.get(k)
        if isinstance(v, str):
            try:
                arr = json.loads(v)
                if isinstance(arr, list):
                    return [str(x).strip() for x in arr if str(x).strip()]
            except Exception:
                pass
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    return []


def append_lora_name_if_empty(tags: list[str], lora_name: str) -> list[str]:
    base = os.path.basename(lora_name).rsplit(".", 1)[0]
    return tags if tags else [base]
