"""
Rerun answerer calls for records with empty responses.

Usage:
  python rerun_empty_answers.py --dataset medxpertqa_mm_stage2_textonly \
      [--config benchbench/config/stage3.yaml] \
      [--answerer-filter gpt-5-mini] \
      [--output-dir benchbench/outputs/stage3_answers/medxpertqa_mm_stage2_textonly]

Only records whose `model_output_raw`/`model_raw_response` are blank or whose
`parsed_answer` is empty are re-answered. Updated records are written back into
the same JSONL file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MAX_RETRY_TOKENS = 8192
MAX_RETRY_ATTEMPTS = 3
DEFAULT_TIMEOUT_SECONDS = 180

# Repo root (benchb/) used for resolving config paths and imports
# __file__ = benchbench/scripts/knowledge_pipeline/stage3/rerun_empty_answers.py
# parents: [0]=stage3, [1]=knowledge_pipeline, [2]=scripts, [3]=benchbench, [4]=benchb
REPO_ROOT = Path(__file__).resolve().parents[4]
# Ensure repo root is on sys.path for absolute imports
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    # When invoked as a module (python -m ...), relative imports work.
    from .run_stage3 import (
        AnswerModelCaller,
        AnswerParser,
        AnswerScorer,
        PromptBuilder,
        Stage3Config,
        slugify,
        LLMManager,
        Stage3DatasetConfig,
    )
except ImportError:
    from benchbench.scripts.knowledge_pipeline.stage3.run_stage3 import (
        AnswerModelCaller,
        AnswerParser,
        AnswerScorer,
        PromptBuilder,
        Stage3Config,
        slugify,
        LLMManager,
        Stage3DatasetConfig,
    )


def _is_empty(raw: Any, parsed: Any) -> bool:
    raw_empty = raw is None or (isinstance(raw, str) and not raw.strip())
    parsed_empty = False
    if parsed is None:
        parsed_empty = True
    elif isinstance(parsed, str):
        parsed_empty = not parsed.strip()
    elif isinstance(parsed, (list, tuple, set)):
        parsed_empty = len(parsed) == 0
    return raw_empty or parsed_empty


def _needs_rerun(rec: Dict[str, Any]) -> bool:
    # Rerun any record that previously errored, OR whose output/parsed answer is empty.
    if rec.get("call_error"):
        return True
    raw = rec.get("model_output_raw")
    if raw is None:
        raw = rec.get("model_raw_response")
    parsed = rec.get("parsed_answer")
    return _is_empty(raw, parsed)


def _cleanup_missing_flags(rec: Dict[str, Any]) -> None:
    flags = rec.get("item_quality_flags") or []
    rec["item_quality_flags"] = [f for f in flags if f != "missing_answer"]
    sources = rec.get("item_quality_source") or []
    rec["item_quality_source"] = [s for s in sources if s != "guardrail_missing_answer"]
    if rec.get("item_status") == "broken_missing_answer":
        rec["item_status"] = "candidate"


def _set_soft_score(row: Dict[str, Any]) -> None:
    """Match Stage3AnswererPanel._set_soft_score logic for analysis compatibility."""
    scoring_method = row.get("scoring_method")
    score_details = row.get("score_details") or {}
    if scoring_method == "llm_judge":
        row["soft_score"] = score_details.get("judge_score")
    elif scoring_method in {"mcq_exact", "numeric_exact", "symbolic_exact", "code_exec"}:
        row["soft_score"] = 1.0 if row.get("is_correct") else 0.0
    elif scoring_method == "skip_core":
        row["soft_score"] = None
    else:
        row["soft_score"] = 1.0 if row.get("is_correct") else 0.0


async def _reanswer_record(
    rec: Dict[str, Any],
    answerer_cfg,
    caller: AnswerModelCaller,
    prompt_builder: PromptBuilder,
    parser: AnswerParser,
    *,
    timeout_seconds: int,
) -> Dict[str, Any]:
    # Build prompt from stored question fields
    system_prompt, user_prompt, letters = prompt_builder.build(rec)
    question_type = rec.get("question_type", "open_ended")
    max_tokens = answerer_cfg.tokens_for(question_type)

    start = time.perf_counter()
    attempts = 0
    raw_response: str = ""
    parsed_answer: Any = None
    current_max_tokens = max_tokens

    while attempts < MAX_RETRY_ATTEMPTS:
        attempts += 1
        try:
            raw_response = await asyncio.wait_for(
                caller.call(
                    answerer_cfg,
                    system_prompt,
                    user_prompt,
                    current_max_tokens,
                ),
                timeout=timeout_seconds,
            )
        except Exception as exc:
            msg = str(exc)
            if ("max_tokens" in msg) or ("output limit" in msg):
                bumped = min(current_max_tokens * 2, MAX_RETRY_TOKENS)
                logger.warning(
                    "Max token limit hit for %s; retrying with max_tokens=%s (was %s)",
                    rec.get("question_id"),
                    bumped,
                    current_max_tokens,
                )
                current_max_tokens = bumped
                continue
            raise

        parsed_answer = parser.parse(rec, raw_response, letters)
        if not _is_empty(raw_response, parsed_answer):
            break

        # Empty response from provider; retry once with higher budget.
        bumped = min(current_max_tokens * 2, MAX_RETRY_TOKENS)
        if bumped <= current_max_tokens:
            logger.error(
                "Empty response for %s; at token cap (%s). Not retrying higher.",
                rec.get("question_id"),
                current_max_tokens,
            )
            break
        logger.warning(
            "Empty response for %s; retrying with max_tokens=%s (was %s)",
            rec.get("question_id"),
            bumped,
            current_max_tokens,
        )
        current_max_tokens = bumped
    else:
        if _is_empty(raw_response, parsed_answer):
            logger.error(
                "Giving up after %s attempts for %s; still empty response.",
                MAX_RETRY_ATTEMPTS,
                rec.get("question_id"),
            )

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Score
    scoring_question = dict(rec)
    scoring_question.pop("scoring_method", None)
    scoring_profile = AnswerScorer.select_scoring_profile(scoring_question)
    is_correct, score_meta = AnswerScorer.score(scoring_question, parsed_answer, scoring_profile=scoring_profile)

    # Update record fields
    rec["model_output_raw"] = raw_response
    rec["model_raw_response"] = raw_response
    rec["parsed_answer"] = parsed_answer
    rec["scoring_method"] = score_meta.get("method")
    rec["score_details"] = score_meta
    rec["is_correct"] = is_correct
    rec["correctness_source"] = "heuristic"
    rec["correctness_reason"] = score_meta.get("reason")
    rec["answer_time_ms"] = elapsed_ms
    _set_soft_score(rec)

    # Clear previous error marker once we have a non-empty answer.
    if not _is_empty(raw_response, parsed_answer):
        rec.pop("call_error", None)

    return rec


def _iter_answer_files(output_dir: Path, dataset: str, answerer_filter: Sequence[str]) -> List[Path]:
    pattern = f"{dataset}_stage3_answers_*.jsonl"
    files = sorted(output_dir.glob(pattern))
    if not answerer_filter:
        return files
    wanted = {slugify(x) for x in answerer_filter}
    keep: List[Path] = []
    for path in files:
        # filename: {dataset}_stage3_answers_{designer}_vs_{answerer}.jsonl
        stem = path.stem
        if "_vs_" not in stem:
            continue
        answerer = stem.split("_vs_", 1)[1]
        if slugify(answerer) in wanted:
            keep.append(path)
    return keep


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun empty answers for Stage3 outputs.")
    parser.add_argument(
        "--config",
        default="benchbench/config/stage3.yaml",
        help="Stage3 YAML config. If relative, we try: CWD, then repo root, then repo_root/benchbench.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (matches stage3 config)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir containing stage3_answers_*.jsonl (default from config)",
    )
    parser.add_argument(
        "--answerer-filter",
        nargs="*",
        default=[],
        help="Restrict to answerer display_name/key (slugified match).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout (default: {DEFAULT_TIMEOUT_SECONDS}s).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report counts; do not rewrite files.")
    args = parser.parse_args()

    # Resolve config path relative to repo root if not absolute
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        # Prefer CWD (what users expect when they run from benchbench/),
        # but keep backward compatibility with benchb/ relative paths.
        candidate_paths = [
            (Path.cwd() / config_path).resolve(),
            (REPO_ROOT / config_path).resolve(),
            (REPO_ROOT / "benchbench" / config_path).resolve(),
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                config_path = candidate
                break
        else:
            # Fall back to the historical repo-root resolution (will error later).
            config_path = candidate_paths[1]

    cfg = Stage3Config.from_yaml(config_path)
    # select dataset config
    ds_cfg: Optional[Stage3DatasetConfig] = None
    for entry in cfg.dataset_configs:
        if entry.dataset == args.dataset:
            ds_cfg = entry
            break
    if ds_cfg is None:
        raise ValueError(f"Dataset {args.dataset} not found in config.")

    answerers = {slugify(a.display_name): a for a in cfg.answer_models if a.enabled}
    answerers.update({slugify(a.key): a for a in cfg.answer_models if a.enabled})

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = ds_cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def _collect_providers(answ: List[Any]) -> List[str]:
        provider_map = {
            "openai": "openai",
            "gemini": "gemini",
            "anthropic": "anthropic",
            "deepseek": "deepseek",
            "grok": "grok",
            "qwen": "qwen",
            "doubao": "doubao",
            "llama": "llama",
        }
        providers = {provider_map[a.provider] for a in answ if a.provider in provider_map}
        return sorted(providers)

    llm = LLMManager(models_to_init=_collect_providers(cfg.answer_models))
    caller = AnswerModelCaller(llm)
    prompt_builder = PromptBuilder()
    parser_obj = AnswerParser()

    files = _iter_answer_files(output_dir, args.dataset, args.answerer_filter)
    if not files:
        print("No answer files found matching filters.")
        return

    async def _process() -> None:
        nonlocal files, answerers, prompt_builder, parser_obj, caller
        total_fixed = 0
        total_empty = 0

        for path in files:
            records: List[Dict[str, Any]] = []
            empties: List[int] = []
            with open(path, encoding="utf-8") as f:
                for _line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if _needs_rerun(rec):
                        empties.append(len(records))
                    records.append(rec)

            if not empties:
                continue

            stem = path.stem
            if "_vs_" not in stem:
                print(f"Skip malformed filename: {path.name}")
                continue
            answerer_name = stem.split("_vs_", 1)[1]
            answerer_key = slugify(answerer_name)
            answerer_cfg = answerers.get(answerer_key)
            if not answerer_cfg:
                print(f"Answerer config not found for {answerer_name}; skipping {path.name}")
                continue

            print(f"{path.name}: {len(empties)} empty responses; re-answering...")
            if args.dry_run:
                total_empty += len(empties)
                continue

            fixed = 0
            still_empty = 0
            for idx in empties:
                rec = records[idx]
                try:
                    updated = await _reanswer_record(
                        rec,
                        answerer_cfg,
                        caller,
                        prompt_builder,
                        parser_obj,
                        timeout_seconds=int(args.timeout_seconds),
                    )
                    raw_new = updated.get("model_output_raw") or updated.get("model_raw_response")
                    parsed_new = updated.get("parsed_answer")
                    if _is_empty(raw_new, parsed_new):
                        # Still empty; keep the missing flags to signal unresolved issues.
                        flags = updated.get("item_quality_flags") or []
                        if "missing_answer" not in flags:
                            flags.append("missing_answer")
                        updated["item_quality_flags"] = flags
                        sources = updated.get("item_quality_source") or []
                        if "guardrail_missing_answer" not in sources:
                            sources.append("guardrail_missing_answer")
                        updated["item_quality_source"] = sources
                        if updated.get("item_status") not in {"broken_missing_answer", "broken"}:
                            updated["item_status"] = "broken_missing_answer"
                        records[idx] = updated
                        still_empty += 1
                    else:
                        _cleanup_missing_flags(updated)
                        records[idx] = updated
                        fixed += 1
                except Exception as exc:
                    print(f"Failed to re-answer {rec.get('question_id')}: {exc}")

            with open(path, "w", encoding="utf-8") as sink:
                for rec in records:
                    sink.write(json.dumps(rec, ensure_ascii=False))
                    sink.write("\n")

            total_fixed += fixed
            total_empty += len(empties)
            unresolved = len(empties) - fixed
            print(f"{path.name}: fixed {fixed}/{len(empties)}, still empty: {still_empty}.")

        print(f"Done. Empty encountered: {total_empty}, fixed: {total_fixed}.")

    asyncio.run(_process())


if __name__ == "__main__":
    main()
