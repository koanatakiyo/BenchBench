#!/usr/bin/env python3
"""
Normalize WeMath subdomains into Math.Geometry.* super parents.
"""

import argparse
import json
from pathlib import Path
from typing import List

from wemath_super_parents import map_wemath_parents

TARGET_DATASET = 'wemath'


def process_file(path: Path, backup: bool = True) -> None:
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

    print(f"[wemath] assigned Math.Geometry super parents in {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Assign Math.Geometry.* super parents to WeMath outputs."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to parsed WeMath results JSONL"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip .bak backups."
    )
    args = parser.parse_args()

    path = Path(args.path.strip())
    if not path.exists():
        raise FileNotFoundError(path)
    process_file(path, backup=not args.no_backup)


if __name__ == "__main__":
    main()

