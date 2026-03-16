# scripts/converters/convert_medxpertqa.py
import json, os

import re

def clean_question(q: str) -> str:
    if not q:
        return q
    # Remove everything from the first "\nAnswer Choices:" onward (case-insensitive, tolerant to spaces)
    return re.split(r'\n\s*answer\s*choices\s*:?', q, flags=re.IGNORECASE)[0].strip()

from base_normalizer import (
    strip_question_only, parse_options_generic,
    map_answer_letters, detect_language,
    write_jsonl, finalize_row, iter_json_any
)


def map_medxpert_record(r):
    # r["options"] or dict{"A": "..."} to list
    options = parse_options_generic(r.get("options"))
    return finalize_row({
        "id": r.get("id"),
        "question": strip_question_only(clean_question(r.get("question",""))),        "options": options,
        "answer": map_answer_letters(r.get("label") or r.get("answer"), options),
        "language": detect_language(r),
        "format": "Multiple-choice",
        "modality": "Multimodal" if r.get("images") or (r.get("background") or {}).get("image_path") else "Text",
        "context": [],
        "background": {
            "question_type": r.get("question_type") or (r.get("background") or {}).get("question_type"),
            "medical_task":  r.get("medical_task")  or (r.get("background") or {}).get("medical_task"),
            "body_system":   r.get("body_system")   or (r.get("background") or {}).get("body_system"),
        },
        "images": r.get("images") or (r.get("background") or {}).get("images") or []
    })

def convert(in_path, out_path):
    rows = (map_medxpert_record(r) for r in iter_json_any(in_path))  # read JSONL
    # out = [map_medxpert_record(r) for r in rows]
    write_jsonl(rows, out_path)
