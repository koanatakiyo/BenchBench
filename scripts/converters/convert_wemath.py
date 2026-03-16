# scripts/converters/convert_wemath.py
import json, re
from base_normalizer import (
    strip_question_only, parse_options_generic,
    map_answer_letters, detect_language,
    write_jsonl, finalize_row
)

def parse_lettered_option_string(s: str):
    # Matches: "A. text", "B: text", "C、text" separated by ';' or end of string
    pairs = []
    for m in re.finditer(r'\s*([A-Ea-e])\s*[\.、:：]\s*(.*?)\s*(?=;|$)', s or "", flags=re.S):
        L = m.group(1).upper()
        txt = m.group(2).strip()
        if txt:
            pairs.append({"letter": L, "content": txt})
    return pairs

def map_wemath_record(r):
    raw_opts = r.get("option") or r.get("options") or []
    if isinstance(raw_opts, str):
        options = parse_lettered_option_string(raw_opts)
    else:
        options = parse_options_generic(raw_opts)

    bg = {
        "knowledge concept": r.get("knowledge concept"),
        "knowledge concept description": r.get("knowledge concept description"),
        "key": r.get("key"),
        "question number": r.get("question number"),
        "image_path": r.get("image_path")
    }
    return finalize_row({
        "id": r.get("ID") or r.get("id"),
        "question": strip_question_only(r.get("question","")),
        "options": options,
        "answer": map_answer_letters(r.get("answer") or r.get("Answer"), options),
        "language": detect_language(r),
        "format": "Multiple-choice",
        "modality": "Multimodal" if bg.get("image_path") else "Text",
        "context": [],
        "background": bg,
        "images": [bg["image_path"]] if bg.get("image_path") else []
    })

def convert(in_path, out_path):
    rows = json.load(open(in_path, "r", encoding="utf-8"))
    out = [map_wemath_record(r) for r in rows]
    write_jsonl(out, out_path)