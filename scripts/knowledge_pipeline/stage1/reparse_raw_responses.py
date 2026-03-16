#!/usr/bin/env python3
"""
Re-parse raw LLM responses inside Stage-1 extraction outputs to recover
full subdomain/term lists.

Usage:
    python reparse_raw_responses.py \
        --pair csbench_en_test=/path/to/csbench_en_results.jsonl \
        --pair tombench_cn=/path/to/tombench_cn_results.jsonl

The script will:
    • Load the corresponding dataset items (for anchoring metadata).
    • For each extraction record, try to json.loads the raw_response,
      stripping ``` fences if needed.
    • If parsing succeeds and schema matches, overwrite the subdomains,
      terms, and confidence fields with the parsed values.
    • Recompute anchored_count / anchored_fraction using lexical matches.
    • Recompute tom_categories using LeanDomainExtractor heuristics.
    • Write the updated JSONL back to disk (with a .bak backup by default).
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set

from lean_domain_extraction import LeanDomainExtractor

EN_CANONICAL_DATASETS = {
    'csbench_en_test',
    'medxpertqa_text',
    'medxpertqa_mm',
    'wemath'
}

ZH_CS_DATASETS = {'csbench_cn_test'}

TOMBENCH_CN_DATASETS = {'tombench_cn'}

EN_GENERIC_TERMS = {
    'the circle',
    'the volume',
    'the base',
    'the following',
    'the emergency'
}

EN_MORPHOLOGY_MAP = {
    'radiu': 'radius',
    'triangle propertie': 'triangle properties',
    'mensuration': 'solid geometry',
    'solid geometry': 'solid geometry'
}

ZH_BLACKLIST = {'机系统', '背景', '内容'}

TOMBENCH_CN_PARTICLES = {'的', '了', '吗', '呢', '吧'}
TOMBENCH_CN_LOCATION_FRAGMENTS = ('哪里', '在哪', '在何处')


def strip_code_fences(raw: str) -> str:
    """Remove ```json ... ``` fences if present."""
    text = raw.strip()
    if text.startswith("```"):
        # Drop leading ``` or ```json
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline+1:]
        text = text.rstrip("`").rstrip()
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def try_parse_raw_response(raw: str) -> Tuple[bool, Dict[str, Any]]:
    """Attempt to parse a raw_response JSON and validate schema."""
    if not raw:
        return False, {}
    candidate = strip_code_fences(raw)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return False, {}

    if not isinstance(data, dict):
        return False, {}

    subdomains = data.get("subdomains")
    terms = data.get("terms")
    confidence = data.get("confidence")

    if (
        not isinstance(subdomains, list)
        or not isinstance(terms, list)
        or not isinstance(confidence, (int, float))
    ):
        return False, {}

    if not all(isinstance(s, str) for s in subdomains):
        return False, {}
    if not all(isinstance(t, str) for t in terms):
        return False, {}

    return True, {
        "subdomains": subdomains,
        "terms": terms,
        "confidence": float(confidence),
        "visual_facts": data.get("visual_facts"),
        "native_glosses": data.get("native_glosses"),
    }


def recompute_metadata(
    extractor: LeanDomainExtractor,
    item: Dict[str, Any],
    extraction: Dict[str, Any]
) -> None:
    """Recompute anchored metadata & ToM tags without mutating term list."""
    terms = extraction.get("terms") or []
    if not isinstance(terms, list):
        return

    item_text = extractor._item_to_text_whitelisted(item).lower()

    lexical_terms = [
        term for term in terms
        if isinstance(term, str) and term.strip()
    ]

    anchored_count = 0
    for term in lexical_terms:
        if term.lower() in item_text:
            anchored_count += 1

    extraction["anchored_count"] = anchored_count
    extraction["anchored_fraction"] = (
        anchored_count / len(lexical_terms) if lexical_terms else 0.0
    )
    extraction["tom_categories"] = [
        extractor._classify_tom_term(term) if isinstance(term, str) else "other"
        for term in terms
    ]


def process_file(
    dataset_name: str,
    results_path: Path,
    backup: bool = True
) -> None:
    extractor = LeanDomainExtractor(
        oracle_name="gpt-5-mini",
        visual_oracle_name=None,
        embedding_model_name="intfloat/multilingual-e5-small"
    )

    extractor.dataset_name = dataset_name
    extractor.dataset_language = extractor._get_dataset_language(dataset_name)

    items = extractor._load_dataset(dataset_name, num_items=None)
    item_lookup = {}
    for idx, item in enumerate(items):
        item_id = str(item.get("id", idx))
        item_lookup[item_id] = item

    updated_lines = []
    modified = 0
    total = 0

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            record = json.loads(line)
            extraction = record.get("extraction") or {}
            raw = extraction.get("raw_response", "")
            parsed_ok, parsed = try_parse_raw_response(raw or "")
            if parsed_ok:
                native_subdomains = [
                    s for s in parsed["subdomains"] if isinstance(s, str)
                ]
                extraction["subdomains_native"] = native_subdomains
                extraction["subdomains_canonical"] = list(native_subdomains)
                extraction["subdomains"] = list(native_subdomains)
                extraction["terms"] = clean_terms(
                    dataset_name,
                    parsed["terms"],
                    extractor
                )
                extraction["confidence"] = parsed["confidence"]
                if parsed.get("visual_facts") is not None:
                    extraction["visual_facts"] = parsed["visual_facts"]
                if parsed.get("native_glosses") is not None:
                    extraction["native_glosses"] = parsed["native_glosses"]

                item_id = str(record.get("item_id", ""))
                item = item_lookup.get(item_id, {})
                recompute_metadata(extractor, item, extraction)
                modified += 1

            updated_lines.append(json.dumps(record, ensure_ascii=False))

    parsed_output_root = extractor.project_root / "outputs" / "parsed_stage1_output"
    parsed_output_root.mkdir(parents=True, exist_ok=True)
    dest_dir = parsed_output_root / dataset_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / results_path.name

    if backup and dest_path.exists():
        backup_path = dest_path.with_suffix(dest_path.suffix + ".bak")
        shutil.copyfile(dest_path, backup_path)

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")

    print(
        f"[{dataset_name}] Updated {modified}/{total} records "
        f"→ {dest_path}"
    )

def clean_terms(
    dataset_name: str,
    terms: List[str],
    extractor: LeanDomainExtractor
) -> List[str]:
    if not terms:
        return []

    if dataset_name in EN_CANONICAL_DATASETS:
        return _clean_terms_en(terms)
    if dataset_name in ZH_CS_DATASETS:
        return _clean_terms_zh_cs(terms)
    if dataset_name in TOMBENCH_CN_DATASETS:
        return _clean_terms_tombench_cn(terms, extractor)

    return _dedup_preserve_order(terms)


def _clean_terms_en(terms: List[str]) -> List[str]:
    normalized: List[str] = []
    seen: Set[str] = set()

    for term in terms:
        if not isinstance(term, str):
            continue
        raw = term.strip()
        if not raw:
            continue
        key = raw.lower()
        mapped = EN_MORPHOLOGY_MAP.get(key, raw)
        lower = mapped.lower()

        if lower in EN_GENERIC_TERMS:
            continue
        if lower in seen:
            continue

        normalized.append(mapped)
        seen.add(lower)

    normalized = _prune_nested_phrases(normalized)
    return normalized


def _prune_nested_phrases(terms: List[str]) -> List[str]:
    result: List[str] = []
    lowered = [t.lower() for t in terms]

    for idx, term in enumerate(terms):
        lower = lowered[idx]
        drop = False
        if lower.endswith(' of') or lower.endswith(' of the'):
            for j, other in enumerate(terms):
                if j == idx:
                    continue
                other_lower = lowered[j]
                if other_lower.startswith(lower) and len(other_lower) > len(lower):
                    remainder = other_lower[len(lower):].strip()
                    if remainder:
                        drop = True
                        break
        if not drop:
            result.append(term)
    return result


def _clean_terms_zh_cs(terms: List[str]) -> List[str]:
    filtered: List[str] = []
    for term in terms:
        if not isinstance(term, str):
            continue
        raw = term.strip()
        if not raw:
            continue
        if raw in ZH_BLACKLIST:
            continue
        if len(raw) <= 3 and raw.endswith('的'):
            continue
        filtered.append(raw)

    result: List[str] = []
    for term in filtered:
        if any(term != other and term in other for other in filtered if len(other) > len(term)):
            continue
        if term not in result:
            result.append(term)
    return result


def _clean_terms_tombench_cn(
    terms: List[str],
    extractor: LeanDomainExtractor
) -> List[str]:
    cleaned: List[str] = []
    protected_terms: Set[str] = set()

    for term in terms:
        if not isinstance(term, str):
            continue
        raw = term.strip()
        if not raw:
            continue

        if raw[-1] in TOMBENCH_CN_PARTICLES and len(raw) > 1:
            stem = raw[:-1]
            if stem and any(stem == t or stem in t for t in terms):
                raw = stem

        if _is_protected_tom_term(raw, extractor):
            protected_terms.add(raw)

        if any(fragment in raw for fragment in TOMBENCH_CN_LOCATION_FRAGMENTS) and not _is_protected_tom_term(raw, extractor):
            continue

        if raw not in cleaned:
            cleaned.append(raw)

    result: List[str] = []
    for term in cleaned:
        if term in protected_terms:
            result.append(term)
            continue
        drop = False
        if term not in protected_terms:
            for other in cleaned:
                if term == other:
                    continue
                if len(other) > len(term) and term in other:
                    drop = True
                    break
        if drop:
            continue
        if term not in result:
            result.append(term)
    return result


def _is_protected_tom_term(term: str, extractor: LeanDomainExtractor) -> bool:
    if not term:
        return False
    category = extractor._classify_tom_term(term)
    return category in {'mental_state', 'emotion'}


def _dedup_preserve_order(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for value in values:
        key = value if not isinstance(value, str) else value.strip()
        if not key:
            continue
        lowered = key.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        output.append(key)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Re-parse raw_response fields to restore terms/subdomains."
    )
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="dataset_name=/path/to/results.jsonl (can be supplied multiple times)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backup files."
    )

    args = parser.parse_args()

    for pair in args.pair:
        if "=" not in pair:
            raise ValueError(f"Invalid --pair format: {pair}")
        dataset_name, path_str = pair.split("=", 1)
        process_file(dataset_name.strip(), Path(path_str.strip()), backup=not args.no_backup)


if __name__ == "__main__":
    main()
 


