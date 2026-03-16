#!/usr/bin/env python3
"""
Annotate MedXpertQA outputs with specialty vs task/modality axes.
"""

import argparse
import json
from pathlib import Path
from typing import List, Set

from med_specialty_taxonomy import lookup_category, SPECIALTY_MAP, TASK_AXIS_MAP

MED_DATASETS = {'medxpertqa_text', 'medxpertqa_mm'}


def assign_tags(subdomains: List[str]) -> (List[str], List[str]):
    specialties: Set[str] = set()
    axes: Set[str] = set()
    for label in subdomains:
        if not isinstance(label, str):
            continue
        specialty = lookup_category(label, SPECIALTY_MAP)
        if specialty:
            specialties.add(specialty)
        axis = lookup_category(label, TASK_AXIS_MAP)
        if axis:
            axes.add(axis)
    return sorted(specialties), sorted(axes)


def process_file(dataset: str, path: Path, backup: bool = True) -> None:
    if dataset not in MED_DATASETS:
        raise ValueError(f"{dataset} is not a MedXpertQA dataset.")

    lines_out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            extraction = record.get("extraction") or {}

            native = extraction.get("subdomains_native")
            if native is None:
                native = extraction.get("subdomains")
                if native is not None:
                    extraction["subdomains_native"] = native

            specialties, axes = assign_tags(native or [])
            if specialties:
                extraction["medical_specialties"] = specialties
            if axes:
                extraction["medical_task_axes"] = axes

            lines_out.append(json.dumps(record, ensure_ascii=False))

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"[{dataset}] annotated specialty + task axes in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Tag MedXpertQA outputs with specialty vs task axes."
    )
    parser.add_argument(
        "--pair",
        required=True,
        action="append",
        help="dataset_name=/path/to/parsed_med_results.jsonl (repeatable)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip .bak backups."
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


