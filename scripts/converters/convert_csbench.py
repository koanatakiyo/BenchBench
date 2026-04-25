# scripts/converters/convert_csbench.py
import json
from base_normalizer import (
    strip_question_only, parse_options_generic,
    map_answer_letters, detect_language,
    write_jsonl, finalize_row
)

def map_csbench_record(row):
    options = []
    for L in "ABCD":
        if row.get(L):
            options.append({"letter": L, "content": str(row[L]).strip()})
    options = options or parse_options_generic(row.get("options"))

    return finalize_row({
        "id": row.get("ID") or row.get("id"),
        "question": strip_question_only(row.get("Question") or row.get("question") or ""),
        "options": options,
        "answer": map_answer_letters(row.get("Answer") or row.get("answer"), options),
        "language": detect_language(row),
        "format": row.get("Format") or "Multiple-choice",
        "modality": "Text",
        "context": [],
        "background": {
            "Domain": row.get("Domain"), "SubDomain": row.get("SubDomain"), "Tag": row.get("Tag")
        }
    })

def convert(in_path, out_path):
    rows = json.load(open(in_path, "r", encoding="utf-8"))
    out = [map_csbench_record(r) for r in rows]
    write_jsonl(out, out_path)
