import re, os, json, unicodedata
from langcodes import Language  
from langdetect import detect, LangDetectException

def strip_question_only(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFKC", s)
    # s = re.sub(r'^\s*\(?\d+\)?[\.、]?\s*', '', s)
    s = re.sub(r'^\s*\(?\d+\)?[\.、]\s*', '', s)
    return s.strip()

def parse_options_generic(obj):
    """
    reformatting the letters：
      - {"A":"...", "B":"..."} / [{"letter":"A","content":"..."}] / ["文本A","文本B"] / 顶层 A/B/C/D
    return：list[{"letter","content"}]（按 A,B,C… 排序）
    """
    pairs = []
    if isinstance(obj, dict):
        for L, txt in obj.items():
            if txt: pairs.append((str(L).upper(), str(txt).strip()))
    elif isinstance(obj, list):
        for i, it in enumerate(obj):
            if isinstance(it, dict) and it.get("content"):
                L = (it.get("letter") or chr(ord('A')+i)).upper()
                pairs.append((L, it["content"].strip()))
            else:
                pairs.append((chr(ord('A')+i), str(it).strip()))
    # reorder and reallocate letters if needed
    pairs = [(L, t) for L, t in pairs if t]
    pairs.sort(key=lambda p: "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(p[0]) if p[0] else 99)
    return [{"letter": L, "content": t} for L, t in pairs]

def map_answer_letters(ans, options):
    """
    ans: "A" | ["A","C"] | text
    options: [{"letter","content"}]
    return：retain letters or texts
    """
    if ans is None:
        return []
    if isinstance(ans, str):
        return [ans]
    if isinstance(ans, list):
        return ans
    return [str(ans)]

# def detect_language(sample, fallback="en"):
#     # check language field
#     return sample.get("Language") or sample.get("language") or fallback


def detect_language(sample, fallback="English"):
    # Try to use explicit language field first
    lang = sample.get("Language") or sample.get("language")

    text = sample.get("question") or sample.get("Question") or sample.get("内容") or ""
    
    try:
        detected = detect(text)
        if lang:
            detected = lang
        # Optionally, map detected code to your normalized language
        if detected.startswith("zh") or detected.startswith("ch"):
            return "Chinese"
        if detected.startswith("en"):
            return "English"
        if detected.startswith("fr"):
            return "French"
        if detected.startswith("de") or detected.startswith("ge"):
            return "German"
        return detected
    
    except LangDetectException:
        return fallback

def collect_images(sample):
    imgs = []
    for k in ("images","image_paths"):
        if isinstance(sample.get(k), list):
            imgs += [str(p) for p in sample[k] if p]
    bg = sample.get("background") or {}
    if isinstance(bg, dict):
        if isinstance(bg.get("images"), list): imgs += [str(p) for p in bg["images"] if p]
        if isinstance(bg.get("image_paths"), list): imgs += [str(p) for p in bg["image_paths"] if p]
        if bg.get("image_path"): imgs.append(str(bg["image_path"]))
    
    seen, out = set(), []
    for p in imgs:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out


def iter_json_any(path):
    """Yield rows from .json(list) or .jsonl (NDJSON)."""
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL parse error at line {lineno}: {e}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for i, row in enumerate(data, 1):
                    yield row
            else:
                raise ValueError("Expected a JSON array for .json file.")

def write_jsonl(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def finalize_row(base):
    """check and fill defaults"""
    base["id"] = str(base["id"])
    base.setdefault("options", [])
    base.setdefault("answer", [])
    base.setdefault("language", "en")
    base.setdefault("format", "Multiple-choice")
    base.setdefault("modality", "Text")
    base.setdefault("background", {})
    base.setdefault("context", {})
    return base
