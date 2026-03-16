# autobench/common.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json, os, yaml
from pathlib import Path

DATA_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parents[1]
RAW_DATA_DIR = f"{DATA_DIR}/data/raw_data"
INPUT_DATA_DIR = f"{DATA_DIR}/data/input_data"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

@dataclass
class ContextPack:
    text: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)
    style: Dict[str, Any] = field(default_factory=dict)   # language/format/modality
    tags: List[str] = field(default_factory=list)          # domain, concept, body_system, ability
    seed_meta: Dict[str, Any] = field(default_factory=dict)  # id, source, etc.

def normalize_language(lang: str) -> str:
    lang = (lang or "").strip().lower()
    if lang in {"en", "eng", "english"}:
        return "English"
    if lang in {"zh", "ch", "chi", "chinese", "cn"}:
        return "Chinese"
    if lang in {"fr", "fre", "french"}:
        return "French"
    if lang in {"de", "ger", "german"}:
        return "German"
    return lang.capitalize() if lang else "Unknown"


def normalize_style(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "language": normalize_language(item.get("language", "en")),
        "format": item.get("format", "Multiple-choice"),
        "modality": item.get("modality", "Text")
    }

# ---- schema loading ----
_CONFIG_CACHE: Optional[Dict[str, Any]] = None

def load_schema(config_path: str = "schema.yaml") -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Schema config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = yaml.safe_load(f) or {}
    return _CONFIG_CACHE

# ---- image import driven by schema ----
def _to_abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(DATA_DIR, path))

def _dedupe_keep_existing(paths: List[str], strict_exists_check: bool = False) -> List[str]:
    seen, out = set(), []
    for p in paths:
        if not p: continue
        if strict_exists_check and not os.path.exists(p):
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def resolve_images(item: Dict[str, Any], source_key: str, config: Dict[str, Any],
                   strict_exists_check: bool = False) -> List[str]:
    key = _canonical_ds_key(source_key)
    ds_cfg = (config.get("datasets") or {}).get(key, {})
    img_cfg = ds_cfg.get("images") or {}
    if not img_cfg.get("enabled", False):
        return []

    mode = img_cfg.get("mode")

    # Case A: list of dicts with a known key
    if mode == "list_of_dicts":
        base = _to_abs_path(img_cfg.get("base", ""))
        key  = img_cfg.get("dict_key", "image_path")
        results: List[str] = []
        for obj in item.get("images") or []:
            if isinstance(obj, dict) and obj.get(key):
                results.append(os.path.normpath(os.path.join(base, obj[key])))
        return _dedupe_keep_existing(results, strict_exists_check)

    # Case B: prefix-based rewrite rules for paths
    if mode == "prefix_rules":
        rules = img_cfg.get("rules") or []
        raw: List[str] = []
        # collect from top-level images (list[str]) and background.image_path
        imgs = item.get("images")
        if isinstance(imgs, list):
            for p in imgs:
                if isinstance(p, str):
                    raw.append(p)
        bg = item.get("background") or {}
        if isinstance(bg, dict) and isinstance(bg.get("image_path"), str):
            raw.append(bg["image_path"])

        resolved: List[str] = []
        for p in raw:
            pp = p.replace("\\", "/")
            matched = False
            for rule in rules:
                prefix = rule.get("prefix", "")
                base   = _to_abs_path(rule.get("base", ""))
                strip  = bool(rule.get("strip_prefix", True))
                if pp.startswith(prefix):
                    rel = pp[len(prefix):] if strip else pp
                    resolved.append(os.path.normpath(os.path.join(base, rel)))
                    matched = True
                    break
            if not matched:
                # no rule matched → keep as-is (or join to first rule’s base if you prefer)
                resolved.append(pp)
        return _dedupe_keep_existing(resolved, strict_exists_check)

    # Default: return any string paths present
    paths: List[str] = []
    imgs = item.get("images")
    if isinstance(imgs, list):
        for p in imgs:
            if isinstance(p, str):
                paths.append(p)
    bg = item.get("background") or {}
    if isinstance(bg, dict) and isinstance(bg.get("image_path"), str):
        paths.append(bg["image_path"])
    return _dedupe_keep_existing(paths, strict_exists_check)

# ---- generic context builder using schema ----
def _canonical_ds_key(source_key: str) -> str:
    sk = (source_key or "").lower()
    if sk.startswith("csbench"): return "csbench"
    if sk.startswith("wemath"): return "wemath"
    if sk.startswith("medxpertqa"): return "medxpertqa"
    if sk.startswith("tomb"): return "tombench"
    return sk



def build_context_from_schema(item: Dict[str, Any], source_key: str, config: Dict[str, Any]) -> ContextPack:
    key = _canonical_ds_key(source_key)
    ds_cfg = (config.get("datasets") or {}).get(key, {})
    bg_fields = ds_cfg.get("background_fields") or []
    context_fields = ds_cfg.get("context_fields") or []
    tag_fields = ds_cfg.get("tag_fields") or []

    bg = item.get("background") or {}
    text_bits: List[str] = []
    for k in bg_fields:
        v = bg.get(k)
        if v is None: 
            continue
        # abilities may be a list of dicts; make it readable
        if k == "abilities" and isinstance(v, list):
            pretty = []
            for a in v:
                if isinstance(a, dict):
                    dim = a.get("dimension", "")
                    ab  = a.get("ability", "")
                    pretty.append(f"{dim}: {ab}".strip(": "))
                else:
                    pretty.append(str(a))
            text_bits.append(" | ".join([p for p in pretty if p]))
        else:
            text_bits.append(str(v))

    ct = item.get("context") or {} # add context to be readable for compling context
    text_bits: List[str] = []
    for k in context_fields:
        v = ct.get(k)
        if v:
            if isinstance(v, list):
                text_bits.extend([str(x) for x in v if x])
            else:
                text_bits.append(str(v))

    tags: List[str] = []
    for k in tag_fields:
        v = bg.get(k)
        if v:
            # flatten list-ish values
            if isinstance(v, list):
                tags.extend([str(x) for x in v if x])
            else:
                tags.append(str(v))

    images = resolve_images(item, source_key, config)

    return ContextPack(
        text=[t for t in text_bits if str(t).strip()],
        images=images,
        style=normalize_style(item),
        tags=tags,
        seed_meta={"id": item.get("id"), "source": source_key}
    )