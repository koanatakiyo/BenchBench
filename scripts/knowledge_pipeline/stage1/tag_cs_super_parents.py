#!/usr/bin/env python3
"""
Normalize CSBench subdomains into CS.Core.* super parents.
"""

import argparse
import json
from pathlib import Path
from typing import List

from cs_super_parents import map_cs_parents

CS_DATASETS = {
    'csbench_en_test',
    'csbench_cn_test',
    'csbench_fr_test',
    'csbench_de_test'
}

TARGET_DATASETS = set(CS_DATASETS)


def process_file(dataset: str, path: Path, backup: bool = True) -> None:
    if dataset not in TARGET_DATASETS:
        raise ValueError(f"{dataset} not supported for this script.")

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

            if dataset in CS_DATASETS:
                tags = map_cs_parents(native or [])
            else:
                tags = map_wemath_parents(native or [])
            if tags:
                extraction["subdomains_canonical"] = tags
                extraction["subdomains"] = tags

            lines_out.append(json.dumps(record, ensure_ascii=False))

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        backup_path.write_text(path.read_text(encoding='utf-8'), encoding='utf-8')

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"[{dataset}] assigned CS.Core super parents in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Assign CS.Core.* super parents to CSBench outputs."
    )
    parser.add_argument(
        "--pair",
        required=True,
        action="append",
        help="dataset_name=/path/to/parsed_cs_results.jsonl (repeatable)"
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

