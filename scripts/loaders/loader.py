# autobench/step1_loader.py
from typing import Dict, Any, Iterable, List
from common import iter_jsonl, load_schema, build_context_from_schema, ContextPack
import os, json, argparse
from pathlib import Path

# register your four inputs (adjust paths if needed)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_BASE = f"{Path(CURRENT_DIR).parents[1]}/data/input_data"
DATA_BASE = f"{Path(CURRENT_DIR).parents[1]}/data"


DATASETS = {
    "csbench_cn_valid":  f"{INPUT_BASE}/csbench/csbench_CN_valid_input.jsonl",
    "csbench_cn_test":   f"{INPUT_BASE}/csbench/csbench_CN_test_input.jsonl",
    "wemath":            f"{INPUT_BASE}/wemath/wemath_input.jsonl",
    "medxpertqa_mm":       f"{INPUT_BASE}/medxpertqa/medxpertqa_mm_input.jsonl",
    "medxpertqa_text":     f"{INPUT_BASE}/medxpertqa/medxpertqa_text_input.jsonl",
    "tomb_cn":           f"{INPUT_BASE}/tombench/tombench_CN_input.jsonl",
    "tomb_en":           f"{INPUT_BASE}/tombench/tombench_EN_input.jsonl",
}

def run(inputs: Dict[str, str], dataset: str, out_preview: str, max_per_ds: int = 200):
    cfg = load_schema("schema.yaml")
    os.makedirs(os.path.dirname(f"out_preview/{dataset}/contexts_preview.jsonl"), exist_ok=True)

    n = 0
    with open(out_preview, "w", encoding="utf-8") as f:
        for source_key, path in inputs.items():
            if not os.path.exists(path):
                print(f"[skip] {source_key}: {path} not found")
                continue
            count = 0
            for item in iter_jsonl(path):
                cp: ContextPack = build_context_from_schema(item, source_key, cfg)
                f.write(json.dumps({
                    "seed_meta": cp.seed_meta,
                    "style": cp.style,
                    "tags": cp.tags,
                    "text": cp.text[:3],
                    "images": cp.images[:2],
                }, ensure_ascii=False) + "\n")
                n += 1
                count += 1
                if count >= max_per_ds:
                    break
    print(f"[ok] wrote {n} context previews -> {out_preview}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=f"{DATA_BASE}")
    ap.add_argument("--max_per_ds", type=int, default=200)
    args = ap.parse_args()
    run(DATASETS, args.dataset, args.out, args.max_per_ds)
