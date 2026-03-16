#!/usr/bin/env python3
"""
Parse Anthropic message batch results (JSONL) into "questions" JSONL.

Input format (one JSON per line), e.g.:
  {"custom_id": "...", "result": {"type": "succeeded", "message": {"content": [{"type":"text","text":"```json\\n[ ... ]\\n```"}]}}}

Output format:
  one question dict per line (JSONL), matching the format used in
  `questions_anthropic_anthropic_*.jsonl`.

Designed to be robust to truncated model outputs (e.g. stop_reason=max_tokens):
we parse as many complete JSON objects as possible from the JSON array.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _extract_text_from_result_obj(obj: Dict[str, Any]) -> str:
    """
    Extract the assistant text from an Anthropic batch result item.
    Prefers result.message.content[].text but falls back to best-effort.
    """
    result = obj.get("result") or {}
    if not isinstance(result, dict):
        return ""
    message = result.get("message") or {}
    if isinstance(message, dict):
        content = message.get("content") or []
        if isinstance(content, list):
            parts: List[str] = []
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    parts.append(str(blk.get("text") or ""))
            return "\n".join(parts)
    # Fallback (older shapes)
    content = result.get("content")
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "text":
                parts.append(str(blk.get("text") or ""))
        return "\n".join(parts)
    if isinstance(content, str):
        return content
    return ""


def _extract_json_payload(text: str) -> str:
    """
    Extract the JSON payload from a markdown code block (```json ... ```).
    If no fenced block is found, return the original text stripped.
    """
    if not text:
        return ""
    s = text.strip()

    # Prefer ```json fenced blocks
    tag = "```json"
    start = s.find(tag)
    if start != -1:
        start = start + len(tag)
        # Skip optional newline(s)
        while start < len(s) and s[start] in "\r\n":
            start += 1
        end = s.find("```", start)
        if end != -1:
            return s[start:end].strip()
        # Truncated: return from start to end
        return s[start:].strip()

    # Generic fenced block
    start = s.find("```")
    if start != -1:
        start = start + 3
        # strip optional language id until newline
        nl = s.find("\n", start)
        if nl != -1:
            start = nl + 1
        end = s.find("```", start)
        if end != -1:
            return s[start:end].strip()
        return s[start:].strip()

    return s


def _parse_json_array_best_effort(payload: str) -> List[Dict[str, Any]]:
    """
    Parse a JSON array payload into a list of dicts.
    Robust to truncation: extracts as many complete objects as possible.
    """
    if not payload:
        return []

    # Find the array start
    i = payload.find("[")
    if i == -1:
        return []
    s = payload[i:]

    decoder = json.JSONDecoder()
    pos = 0

    # Skip whitespace
    while pos < len(s) and s[pos].isspace():
        pos += 1
    if pos >= len(s) or s[pos] != "[":
        return []
    pos += 1

    items: List[Dict[str, Any]] = []
    while True:
        # Skip whitespace and commas
        while pos < len(s) and (s[pos].isspace() or s[pos] == ","):
            pos += 1
        if pos >= len(s):
            break
        if s[pos] == "]":
            break

        try:
            obj, next_pos = decoder.raw_decode(s, pos)
        except json.JSONDecodeError:
            # Truncation or garbage: stop parsing this payload.
            break
        pos = next_pos
        if isinstance(obj, dict):
            items.append(obj)
        # If it isn't a dict, ignore (unexpected)
    return items


def iter_batch_lines(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            yield lineno, json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_jsonl", type=Path, help="Anthropic batch results JSONL")
    ap.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Output questions JSONL (default: <input>_questions.jsonl)",
    )
    ap.add_argument(
        "--emit-source",
        action="store_true",
        help="Add helper fields: _batch_custom_id and _batch_line_no",
    )
    args = ap.parse_args()

    in_path: Path = args.input_jsonl
    out_path: Path = args.output_jsonl or in_path.with_name(in_path.stem + "_questions.jsonl")

    total_questions = 0
    per_custom_id: Dict[str, int] = {}
    skipped = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for lineno, obj in iter_batch_lines(in_path):
            custom_id = str(obj.get("custom_id") or "")

            result = obj.get("result") or {}
            rtype = result.get("type") if isinstance(result, dict) else None
            if rtype not in {None, "succeeded"}:
                skipped += 1
                continue

            text = _extract_text_from_result_obj(obj)
            payload = _extract_json_payload(text)
            questions = _parse_json_array_best_effort(payload)
            if not questions:
                skipped += 1
                continue

            for q in questions:
                if args.emit_source:
                    q["_batch_custom_id"] = custom_id
                    q["_batch_line_no"] = lineno
                out.write(json.dumps(q, ensure_ascii=False))
                out.write("\n")
                total_questions += 1
                per_custom_id[custom_id] = per_custom_id.get(custom_id, 0) + 1

    # Print a tiny summary for CLI usage
    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "total_questions": total_questions,
        "unique_custom_ids": len([k for k, v in per_custom_id.items() if v > 0]),
        "per_custom_id": {k: per_custom_id[k] for k in sorted(per_custom_id)},
        "skipped_items": skipped,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


