#!/usr/bin/env python3
"""
Global subdomain canonicalization across domain families (CS, Med, ToM, Math).

Usage:
    python canonicalize_subdomains.py \
        --pair csbench_en_test=/path/to/csbench_en_test_results.jsonl \
        --pair tombench_cn=/path/to/tombench_cn_results.jsonl \
        --no-backup

The script:
    1. Collects all subdomain labels per family.
    2. Embeds them via SentenceTransformer.
    3. Greedily merges near-duplicates using cosine threshold per family.
    4. Prints the top canonical parents for inspection.
    5. Rewrites each JSONL with canonicalized subdomains (backups by default).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

DATASET_META = {
    'csbench_en_test': ('cs', 'en'),
    'csbench_cn_test': ('cs', 'zh'),
    'csbench_fr_test': ('cs', 'fr'),
    'csbench_de_test': ('cs', 'de'),
    'medxpertqa_text': ('med', 'en'),
    'medxpertqa_mm': ('med', 'en'),
    'tombench_en': ('tom', 'en'),
    'tombench_cn': ('tom', 'zh'),
    'wemath': ('math', 'en'),
}

FAMILY_THRESHOLDS = {
    'cs': 0.91,
    'med': 0.90,
    'tom': 0.88,
    'math': 0.92
}


def normalize_subdomain(text: str) -> str:
    return text.strip()


def build_family_clusters(
    freq: Counter,
    embeddings: Dict[str, np.ndarray],
    threshold: float
) -> Tuple[Dict[str, str], List[Dict[str, any]]]:
    canonical_map: Dict[str, str] = {}
    clusters: List[Dict[str, any]] = []

    ordered = sorted(
        freq.keys(),
        key=lambda x: (-freq[x], len(x), x.lower())
    )

    for label in ordered:
        vec = embeddings[label]
        assigned = False
        for cluster in clusters:
            canon_vec = cluster['vector']
            if float(np.dot(vec, canon_vec)) >= threshold:
                cluster['members'].append(label)
                cluster['total_freq'] += freq[label]
                if freq[label] > cluster['best_freq']:
                    cluster['canonical'] = label
                    cluster['best_freq'] = freq[label]
                    cluster['vector'] = vec
                canonical_map[label] = cluster['canonical']
                assigned = True
                break
        if not assigned:
            clusters.append({
                'canonical': label,
                'members': [label],
                'vector': vec,
                'best_freq': freq[label],
                'total_freq': freq[label]
            })
            canonical_map[label] = label

    for cluster in clusters:
        for member in cluster['members']:
            canonical_map[member] = cluster['canonical']

    return canonical_map, clusters


def rewrite_file(path: Path, mapping: Dict[str, str], backup: bool) -> None:
    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')

    updated_lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            extraction = record.get("extraction") or {}
            subdomains = extraction.get("subdomains_native") or extraction.get("subdomains") or []
            canonical = []
            seen = set()
            for sub in subdomains:
                if not isinstance(sub, str):
                    continue
                label = mapping.get(sub.strip(), sub.strip())
                if label not in seen:
                    canonical.append(label)
                    seen.add(label)
            if "subdomains_native" not in extraction:
                extraction["subdomains_native"] = subdomains
            extraction["subdomains_canonical"] = canonical
            extraction["subdomains"] = canonical
            updated_lines.append(json.dumps(record, ensure_ascii=False))

    with open(path, "w", encoding='utf-8') as f:
        f.write("\n".join(updated_lines) + "\n")


def print_summary(
    family: str,
    language: str,
    clusters: List[Dict[str, any]],
    top_k: int = 20
) -> None:
    print(f"\n=== {family.upper()} / {language} canonical parents (top {top_k}) ===")
    ranked = sorted(clusters, key=lambda c: c['total_freq'], reverse=True)
    for idx, cluster in enumerate(ranked[:top_k], start=1):
        canonical = cluster['canonical']
        count = cluster['total_freq']
        size = len(cluster['members'])
        sample_members = cluster['members'][:5]
        print(f"{idx:02d}. {canonical}  (freq={count}, variants={size}) -> {sample_members}")


def main():
    parser = argparse.ArgumentParser(
        description="Globally canonicalize subdomains across domain families."
    )
    parser.add_argument(
        "--pair",
        required=True,
        action="append",
        help="dataset_name=/path/to/results.jsonl (may be repeated)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create .bak backups before rewriting."
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top parents to print per family (default 20)."
    )
    args = parser.parse_args()

    family_strings: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    dataset_files: Dict[str, Path] = {}

    for spec in args.pair:
        if "=" not in spec:
            raise ValueError(f"Invalid --pair format: {spec}")
        dataset, path_str = spec.split("=", 1)
        dataset = dataset.strip()
        path = Path(path_str.strip())
        if dataset not in DATASET_META:
            raise ValueError(f"Unknown dataset {dataset}. Update DATASET_FAMILY_MAP.")
        if not path.exists():
            raise FileNotFoundError(path)
        dataset_files[dataset] = path

        family, language = DATASET_META[dataset]
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                extraction = record.get("extraction") or {}
                subdomains = extraction.get("subdomains") or []
                for sub in subdomains:
                    if not isinstance(sub, str):
                        continue
                    norm = normalize_subdomain(sub)
                    if not norm:
                        continue
                    family_strings[(family, language)][norm] += 1

    model = SentenceTransformer("intfloat/multilingual-e5-small")
    family_mappings: Dict[Tuple[str, str], Dict[str, str]] = {}

    for (family, language), freq in family_strings.items():
        if not freq:
            continue
        unique_strings = list(freq.keys())
        embeddings = model.encode(unique_strings, normalize_embeddings=True)
        emb_map = {label: embeddings[idx] for idx, label in enumerate(unique_strings)}
        threshold = FAMILY_THRESHOLDS.get(family, 0.90)
        mapping, clusters = build_family_clusters(freq, emb_map, threshold)
        family_mappings[(family, language)] = mapping
        print_summary(family, language, clusters, top_k=args.top)

    for dataset, path in dataset_files.items():
        family, language = DATASET_META[dataset]
        mapping = family_mappings.get((family, language))
        if not mapping:
            continue
        rewrite_file(path, mapping, backup=not args.no_backup)
        print(f"[{dataset}] canonicalized subdomains in {path}")


if __name__ == "__main__":
    main()

