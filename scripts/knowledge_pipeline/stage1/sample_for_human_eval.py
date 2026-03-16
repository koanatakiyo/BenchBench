#!/usr/bin/env python3
"""
Sample Stage-1 items for quick human spot-checking.

For each dataset, we sample N items and emit a CSV with the fields a human
grader needs to mark:
- subdomain correct? (Y/N)
- ≥50% terms useful? (Y/N)
- any hallucinated/unanchored terms? (Y/N)

Inputs:
- Phase-0 dataset items: outputs/phase0_datasets/{dataset}/dataset.jsonl
- Stage-1 results: prefers parsed_stage1_output/{dataset}/*.jsonl, falls back
  to lean_domain_extraction/*{dataset}_results*.jsonl

Outputs:
- outputs/human_spotcheck/{dataset}_sample_for_human_eval.csv
"""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE0_DIR = PROJECT_ROOT / "outputs" / "phase0_datasets"
PARSED_DIR = PROJECT_ROOT / "outputs" / "parsed_stage1_output"
RAW_STAGE1_DIR = PROJECT_ROOT / "outputs" / "lean_domain_extraction"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "human_spotcheck"


def load_items(dataset: str) -> List[Dict[str, Any]]:
    path = PHASE0_DIR / dataset / "dataset.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset items: {path}")
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def find_results_file(dataset: str) -> Path:
    parsed_dir = PARSED_DIR / dataset
    if parsed_dir.exists():
        files = sorted(parsed_dir.glob("*results*.jsonl"))
        if files:
            return files[-1]
    # fallback: raw stage1 outputs
    candidates = sorted(RAW_STAGE1_DIR.glob(f"*{dataset}*results*.jsonl"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(
        f"No Stage-1 results found for {dataset} in {parsed_dir} or {RAW_STAGE1_DIR}"
    )


def load_extractions(results_path: Path) -> Dict[str, Dict[str, Any]]:
    extractions: Dict[str, Dict[str, Any]] = {}
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            item_id = str(record.get("item_id", ""))
            extraction = record.get("extraction") or {}
            extractions[item_id] = extraction
    return extractions


def flatten_options(options: Any) -> str:
    if options is None:
        return ""
    if isinstance(options, dict):
        return " | ".join(f"{k}) {v}" for k, v in options.items())
    if isinstance(options, list):
        return " | ".join(str(o) for o in options)
    return str(options)


def first_n_lines(text: str, max_lines: int = 6, max_chars: int = 800) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    trimmed = "\n".join(lines[:max_lines])
    if len(trimmed) > max_chars:
        trimmed = trimmed[: max_chars - 3] + "..."
    return trimmed


def sample_items(
    dataset: str,
    sample_size: int,
    seed: int
) -> Path:
    items = load_items(dataset)
    results_path = find_results_file(dataset)
    extractions = load_extractions(results_path)

    if not items:
        raise ValueError(f"No items loaded for {dataset}")

    random.seed(seed)
    sample = random.sample(items, min(sample_size, len(items)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{dataset}_sample_for_human_eval.csv"

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "item_id",
                "question",
                "options",
                "context",
                "subdomains",
                "subdomains_native",
                "terms",
                "anchored_fraction",
                "anchored_count",
                "confidence",
                "human_subdomain_correct_YN",
                "human_terms_useful_ge50_YN",
                "human_hallucinated_terms_YN",
                "notes",
            ],
        )
        writer.writeheader()

        for item in sample:
            item_id = str(item.get("id", ""))
            extraction = extractions.get(item_id, {})
            writer.writerow({
                "dataset": dataset,
                "item_id": item_id,
                "question": first_n_lines(str(item.get("question", ""))),
                "options": first_n_lines(flatten_options(item.get("options"))),
                "context": first_n_lines(str(item.get("context", ""))),
                "subdomains": "; ".join(extraction.get("subdomains", [])),
                "subdomains_native": "; ".join(
                    extraction.get("subdomains_native", [])
                ),
                "terms": "; ".join(extraction.get("terms", [])),
                "anchored_fraction": extraction.get("anchored_fraction", ""),
                "anchored_count": extraction.get("anchored_count", ""),
                "confidence": extraction.get("confidence", ""),
                "human_subdomain_correct_YN": "",
                "human_terms_useful_ge50_YN": "",
                "human_hallucinated_terms_YN": "",
                "notes": "",
            })

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Sample Stage-1 items for human QC")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Datasets to sample, e.g. csbench_en_test tombench_en",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Items per dataset (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    paths: List[Path] = []
    for dataset in args.datasets:
        path = sample_items(dataset, args.sample_size, args.seed)
        paths.append(path)
        print(f"[{dataset}] wrote {path}")

    print("\nDone. Open the CSVs above and mark Y/N for the three human fields.")


if __name__ == "__main__":
    main()

