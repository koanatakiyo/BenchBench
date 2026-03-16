#!/usr/bin/env python3
"""
Assign ToM super-taxonomy tags + scenario labels to parsed Stage-1 outputs.

Usage:
    python tag_tom_super_axes.py \
        --pair tombench_en=/path/to/parsed/tombench_en_results.jsonl \
        --pair tombench_cn=/path/to/parsed/tombench_cn_results.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from tom_taxonomy import (
    TOM_SUPER_TAGS,
    KEYWORD_MAP,
    SCENARIO_ALIASES
)

DATASET_LANGUAGE = {
    'tombench_en': 'en',
    'tombench_cn': 'zh'
}


def normalize(text: str) -> str:
    return text.strip().lower()


def collect_candidates(extraction: Dict[str, any]) -> List[str]:
    values: List[str] = []
    for field in ("subdomains_canonical", "subdomains_native", "subdomains", "terms"):
        seq = extraction.get(field) or []
        for item in seq:
            if isinstance(item, str):
                values.append(item.strip())
    raw = extraction.get("raw_response")
    if isinstance(raw, str):
        values.append(raw)
    return values


def match_super_tags(values: List[str], language: str) -> Set[str]:
    tags: Set[str] = set()
    lowered = [normalize(v) for v in values if isinstance(v, str)]
    for tag, meta in TOM_SUPER_TAGS.items():
        aliases = meta["aliases"].get(language, [])
        for alias in aliases:
            alias_norm = normalize(alias)
            if any(alias_norm in v for v in lowered):
                tags.add(tag)
                break
    return tags


def heuristic_tags(values: List[str], language: str) -> Set[str]:
    tags: Set[str] = set()
    aggregate = " ".join(values).lower()
    for tag, lang_map in KEYWORD_MAP.items():
        keywords = lang_map.get(language, [])
        if any(keyword in aggregate for keyword in keywords):
            tags.add(tag)
    return tags


def scenario_tags(values: List[str]) -> List[str]:
    tags: List[str] = []
    combined = " ".join(values).lower()
    for label, aliases in SCENARIO_ALIASES.items():
        if any(alias.lower() in combined for alias in aliases):
            tags.append(label)
    seen: Set[str] = set()
    deduped = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped[:3]


def process_file(dataset: str, path: Path, backup: bool = True) -> None:
    language = DATASET_LANGUAGE.get(dataset)
    if not language:
        raise ValueError(f"Dataset {dataset} missing language mapping.")

    lines_out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            extraction = record.get("extraction") or {}

            if "subdomains_native" not in extraction:
                native = extraction.get("subdomains")
                if native is not None:
                    extraction["subdomains_native"] = native

            values = collect_candidates(extraction)
            tags = match_super_tags(values, language)
            if not tags:
                tags = heuristic_tags(values, language)
            if not tags:
                # fallback to generic tag based on language
                tags = {"ToM.BeliefReasoning"}

            scenarios = scenario_tags(values)

            canonical_tags = sorted(tags)

            extraction["subdomains_canonical"] = canonical_tags
            extraction["subdomains"] = canonical_tags
            if scenarios:
                extraction["scenario_tags"] = scenarios

            lines_out.append(json.dumps(record, ensure_ascii=False))

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"[{dataset}] Tagged ToM super axes in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Assign ToM super-tags and scenario metadata."
    )
    parser.add_argument(
        "--pair",
        required=True,
        action="append",
        help="dataset_name=/path/to/results.jsonl (repeatable)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write .bak backups."
    )
    args = parser.parse_args()

    for spec in args.pair:
        if "=" not in spec:
            raise ValueError(f"Invalid --pair format: {spec}")
        dataset, path_str = spec.split("=", 1)
        dataset = dataset.strip()
        path = Path(path_str.strip())
        if not path.exists():
            raise FileNotFoundError(path)
        process_file(dataset, path, backup=not args.no_backup)


if __name__ == "__main__":
    main()

