# scripts/converters/convert_tombench.py
import json, os
from base_normalizer import (
    strip_question_only, parse_options_generic,
    map_answer_letters, detect_language,
    write_jsonl, finalize_row, iter_json_any
)

def map_tomb_record(r):
    options = parse_options_generic(r.get("options"))
    return finalize_row({
        "id": r.get("id"),
        "question": (r.get("question") or "").strip(),
        "options": options,
        "answer": map_answer_letters(r.get("answer"), options),
        "language": r.get("language") or detect_language(r),
        "format": "Multiple-choice",
        "modality": "Text",
        "context": {
            "story": r.get("story",""),
        },
        "background": {
            "abilities": r.get("abilities", [])
        }
    })

def convert(in_path, out_path):
    rows = [map_tomb_record(r) for r in iter_json_any(in_path)]
    write_jsonl(rows, out_path)