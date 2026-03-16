#!/usr/bin/env python3
"""
Scan Stage3 answer JSONL files for rows missing judge outputs and fill them.

Criteria for (re)judging a row:
  - score_details.reason == "judge_missing", OR
  - score_details.judge_score is null/absent, OR
  - record-level judge_score is null/absent, OR
  - model_output_raw/model_raw_response is null/empty (reported only; cannot judge without an answer).

For rows with empty model output, we log and skip judging.
For eligible rows, we call the configured judge model from stage3.yaml and update the record in place.

Usage:
  python fill_missing_judge.py --dataset csbench_cn \
      --config benchbench/config/stage3.yaml \
      --answers-dir benchbench/outputs/stage3_answers/csbench_cn

Optional filters:
  --answerer-filter gpt-5-mini gemini_2_5_flash

This script mirrors the judge prompt from run_stage3._call_llm_answer_judge.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sys

# Repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from benchbench.scripts.knowledge_pipeline.stage3.run_stage3 import (
    AnswerModelCaller,
    AnswerModelConfig,
    AnswerParser,
    Stage3Config,
    JUDGE_CORRECT_THRESHOLD,
    slugify,
    LLMManager,
    DEFAULT_MAX_NEW_TOKENS,
)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("fill_missing_judge")
DEFAULT_TIMEOUT_SECONDS = 180


def _is_empty_text(val: Any) -> bool:
    return val is None or (isinstance(val, str) and not val.strip())


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


def _needs_judge(rec: Dict[str, Any]) -> Tuple[bool, bool]:
    """Return (needs_judge, has_empty_answer) only for scoring_method == llm_judge."""
    sd = rec.get("score_details") or {}
    reason = sd.get("reason") or rec.get("reason")
    judge_score = sd.get("judge_score", rec.get("judge_score"))
    scoring_method = (sd.get("method") or rec.get("scoring_method") or "").lower()

    need = scoring_method == "llm_judge" and (reason == "judge_missing" or judge_score is None)

    raw = rec.get("model_output_raw")
    if raw is None:
        raw = rec.get("model_raw_response")
    empty_answer = _is_empty_text(raw)
    return need, empty_answer


def _iter_answer_files(answers_dir: Path, dataset: str, answerer_filter: Sequence[str]) -> List[Path]:
    pattern = f"{dataset}_stage3_answers_*.jsonl"
    files = sorted(answers_dir.glob(pattern))
    if not answerer_filter:
        return files
    wanted = {slugify(x) for x in answerer_filter}
    keep: List[Path] = []
    for path in files:
        stem = path.stem
        if "_vs_" not in stem:
            continue
        answerer = stem.split("_vs_", 1)[1]
        if slugify(answerer) in wanted:
            keep.append(path)
    return keep


def _build_judge_config(cfg: Dict[str, Any]) -> Optional[AnswerModelConfig]:
    if not cfg:
        return None
    tokens = dict(DEFAULT_MAX_NEW_TOKENS)
    tokens.update(cfg.get("max_new_tokens") or {})
    return AnswerModelConfig(
        key=str(cfg.get("display_name") or cfg.get("model") or "judge"),
        display_name=str(cfg.get("display_name") or cfg.get("model") or "judge"),
        provider=str(cfg.get("provider") or "").lower(),
        model=str(cfg.get("model")),
        batch_size=int(cfg.get("batch_size", 4)) if cfg.get("batch_size") is not None else 4,
        max_concurrent_requests=int(cfg.get("max_concurrent_requests", 4)) if cfg.get("max_concurrent_requests") is not None else 4,
        temperature=float(cfg.get("temperature", 0.0)),
        top_p=float(cfg.get("top_p", 1.0)),
        max_new_tokens=tokens,
        extra_body=cfg.get("extra_body") or {},
        enabled=True,
    )


async def _call_judge(
    caller: AnswerModelCaller,
    judge_cfg: AnswerModelConfig,
    question: Dict[str, Any],
    model_answer: str,
    *,
    timeout_seconds: int,
) -> Tuple[Optional[bool], Dict[str, Any]]:
    stem = question.get("question_stem") or ""
    gold = question.get("gold_answer") or question.get("answer") or ""
    lang = (question.get("language") or "").lower()
    is_zh = "zh" in lang or "chinese" in lang

    # English judge prompt
    prompt_en = (
        "You are an expert grader evaluating a student's answer.\n\n"
        "Question:\n"
        f"{stem}\n\n"
        "Gold reference answer:\n"
        f"{gold}\n\n"
        "Student answer:\n"
        f"{model_answer}\n\n"
        "Instructions:\n"
        "- Core ideas = essential facts, constraints, or conditions without which the answer is technically inaccurate (e.g., for associative property: must specify it holds only for addition/multiplication, not all operations).\n"
        "- Be strict on technical accuracy, but tolerant of rephrasing, examples, or extra correct detail.\n"
        '- If uncertain, output "broken_item". Do NOT guess.\n\n'
        '- If you are running out of tokens or near the limit, respond with ONLY this tiny JSON (no markdown): {"label": "correct" | "partially_correct" | "incorrect" | "broken_item", "score": <float>}.\n\n'
        "Grading rubric:\n"
        "correct (0.9-1.0): All core ideas present; minor phrasing differences OK.\n"
        "partially_correct (0.4-0.8): Main idea present, but key details missing, vague, or slightly misstated. Higher score = only minor omissions.\n"
        "incorrect (0.0-0.3): Core concept missing or fundamentally wrong.\n"
        "broken_item: Question/gold answer is flawed, ambiguous, or factually incorrect.\n\n"
        "Output requirements:\n"
        "- JSON only. No markdown, no extra text.\n"
        '- "missing" and "errors" must be lists (empty if none).\n'
        "- Keep items in \"missing\"/\"errors\" concise (<=12 words).\n"
        "- KEEP THE MISSING AND ERRORS FIELDS SHORT AND SIMPLE. LESS THAN 30 WORDS.\n"
        "- Do NOT include reasoning outside JSON.\n\n"
        "{\n"
        '  "label": "correct" | "partially_correct" | "incorrect" | "broken_item",\n'
        '  "score": <float>,\n'
        '  "missing": ["..."],\n'
        '  "errors": ["..."]\n'
        "}"
    )

    # Chinese judge prompt for Chinese-language datasets
    prompt_zh = (
        "你是一名严谨的算法题评分专家，请严格按以下规则评分：\n\n"
        "题目：\n"
        f"{stem}\n\n"
        "标准答案：\n"
        f"{gold}\n\n"
        "学生答案：\n"
        f"{model_answer}\n\n"
        "评分规则：\n"
        "- 若标准答案本身错误/不合法，标为 \"broken_item\"；\n"
        "- 若学生答案正确（即使格式略有不同），标为 \"correct\"；\n"
        "- 仅当学生答案遗漏关键步骤或逻辑错误，才标 \"partially_correct\" / \"incorrect\"；\n"
        "- 一旦接近 token 上限，立即输出极简 JSON，不加任何说明。\n\n"
        "请输出纯 JSON，无 markdown：\n"
        "{\n"
        '  \"label\": \"correct\" | \"partially_correct\" | \"incorrect\" | \"broken_item\",\n'
        '  \"score\": <float>,\n'
        '  \"missing\": [],\n'
        '  \"errors\": []\n'
        "}"
    )

    prompt = prompt_zh if is_zh else prompt_en

    judge_sem = asyncio.Semaphore(4)
    max_tokens = judge_cfg.tokens_for("open_ended")

    # First attempt: full prompt
    async with judge_sem:
        response_text = await asyncio.wait_for(
            caller.call(
                judge_cfg,
                "You are an expert grader. Output JSON only.",
                prompt,
                max_tokens,
            ),
            timeout=timeout_seconds,
        )

    stripped_first = (response_text or "").strip()

    # If the model effectively returned nothing (often due to max-token issues),
    # fall back to a tiny prompt that only asks for the minimal JSON.
    if not stripped_first:
        LOGGER.warning(
            "fill_missing_judge: first judge call empty, retrying with tiny JSON-only prompt"
        )
        if is_zh:
            tiny_prompt = (
                "你是一名严谨的算法题评分专家。\n\n"
                "题目：\n"
                f"{stem}\n\n"
                "标准答案：\n"
                f"{gold}\n\n"
                "学生答案：\n"
                f"{model_answer}\n\n"
                "现在请你立刻输出极简 JSON（不加任何说明、不加 markdown）：\n"
                '{ "label": "correct" | "partially_correct" | "incorrect" | "broken_item",'
                ' "score": <float between 0.0 and 1.0> }\n'
            )
        else:
            tiny_prompt = (
                "You are an expert grader evaluating a student's answer.\n\n"
                "Question:\n"
                f"{stem}\n\n"
                "Gold reference answer:\n"
                f"{gold}\n\n"
                "Student answer:\n"
                f"{model_answer}\n\n"
                "Return ONLY this tiny JSON (no markdown, no explanations, no reasoning, no extra text): \n"
                '{ "label": "correct" | "partially_correct" | "incorrect" | "broken_item",'
                ' "score": <float between 0.0 and 1.0> }\n'
            )

        async with judge_sem:
            response_text = await asyncio.wait_for(
                caller.call(
                    judge_cfg,
                    "You are an expert grader. Output JSON only.",
                    tiny_prompt,
                    max_tokens,
                ),
                timeout=timeout_seconds,
            )

    def _parse_judge_response(response: str) -> Dict[str, Any]:
        """Parse JSON from judge response, handling markdown code blocks."""
        if not response:
            return {"raw_response": response}
        stripped = response.strip()
        # Try direct parse
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data | {"raw_response": response}
        except Exception:
            pass
        # Try fenced code block ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S | re.I)
        if fence_match:
            try:
                data = json.loads(fence_match.group(1))
                if isinstance(data, dict):
                    return data | {"raw_response": response}
            except Exception:
                pass
        # Try first JSON object anywhere
        brace_match = re.search(r"\{.*\}", stripped, re.S)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
                if isinstance(data, dict):
                    return data | {"raw_response": response}
            except Exception:
                pass
        return {"raw_response": response}

    parsed = _parse_judge_response(response_text)
    label = str(parsed.get("label") or "").strip().lower()
    score = parsed.get("score")

    # Fallbacks so we do not keep re-judging forever
    fallback_meta: Dict[str, Any] = {
        "judge_raw_response": response_text,
        "judge_label": label or "broken_item",
        "judge_errors": [],
    }

    if label not in {"correct", "partially_correct", "incorrect", "broken_item"}:
        fallback_meta["judge_errors"] = ["unparsable_judge_response"]
        fallback_meta["judge_score"] = 0.0
        return False, fallback_meta

    if score is None:
        # Treat missing score as 0.0 so it no longer appears as judge_missing
        score = 0.0
        fallback_meta["judge_errors"] = ["missing_score"]
        fallback_meta["judge_score"] = score
        return False, fallback_meta

    meta = {
        "judge_label": label,
        "judge_score": score,
        "judge_missing": parsed.get("missing") if isinstance(parsed.get("missing"), list) else [],
        "judge_errors": parsed.get("errors") if isinstance(parsed.get("errors"), list) else [],
        "judge_raw_response": response_text,
    }

    if label == "broken_item":
        return None, meta

    try:
        numeric_score = float(score)
    except Exception:
        numeric_score = None
    if numeric_score is not None:
        return bool(numeric_score >= JUDGE_CORRECT_THRESHOLD), meta
    if label == "correct":
        return True, meta
    if label in {"partially_correct", "incorrect"}:
        return False, meta
    return None, meta


async def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing judge outputs in Stage3 answers.")
    parser.add_argument("--config", default="benchbench/config/stage3.yaml", help="Stage3 YAML config.")
    parser.add_argument("--dataset", required=True, help="Dataset name (matches stage3 config).")
    parser.add_argument(
        "--answers-dir",
        default=None,
        help="Dir containing stage3 answer JSONL files (defaults to dataset output_dir from config).",
    )
    parser.add_argument("--answerer-filter", nargs="*", default=[], help="Filter by answerer key/display (slugified).")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-request timeout (default: {DEFAULT_TIMEOUT_SECONDS}s).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only; do not rewrite files.")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        candidate_paths = [
            (Path.cwd() / cfg_path).resolve(),
            (REPO_ROOT / cfg_path).resolve(),
            (REPO_ROOT / "benchbench" / cfg_path).resolve(),
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                cfg_path = candidate
                break
        else:
            cfg_path = candidate_paths[1]

    cfg = Stage3Config.from_yaml(cfg_path)
    judge_cfg = _build_judge_config(cfg.judge_model or {})
    if not judge_cfg:
        raise SystemExit("No judge_model configured in stage3.yaml.")

    # Locate dataset entry
    ds_cfg = None
    for ds in cfg.dataset_configs:
        if ds.dataset == args.dataset:
            ds_cfg = ds
            break
    if not ds_cfg:
        raise SystemExit(f"Dataset {args.dataset} not found in config.")

    answers_dir = Path(args.answers_dir).expanduser() if args.answers_dir else ds_cfg.output_dir
    if not answers_dir.is_absolute():
        candidate_paths = [
            (Path.cwd() / answers_dir).resolve(),
            (REPO_ROOT / answers_dir).resolve(),
            (REPO_ROOT / "benchbench" / answers_dir).resolve(),
        ]
        for candidate in candidate_paths:
            if candidate.exists():
                answers_dir = candidate
                break
        else:
            # Default to CWD-relative (what users expect when running from benchbench/).
            answers_dir = candidate_paths[0]
    files = _iter_answer_files(answers_dir, args.dataset, args.answerer_filter)
    if not files:
        print("No answer files found.")
        return

    llm = LLMManager(models_to_init=[judge_cfg.provider])
    caller = AnswerModelCaller(llm)

    total_need = 0
    total_fixed = 0
    total_empty_ans = 0

    for path in files:
        rows: List[Dict[str, Any]] = []
        need_indices: List[int] = []
        empty_only: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for _line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec_idx = len(rows)
                need, empty_ans = _needs_judge(rec)
                if empty_ans:
                    empty_only.append(rec_idx)
                if need:
                    need_indices.append(rec_idx)
                rows.append(rec)

        if not need_indices:
            continue

        total_need += len(need_indices)
        total_empty_ans += len(empty_only)

        if args.dry_run:
            # In dry-run mode, report which records would be (re)judged so users can inspect them.
            print(f"{path.name}: needs_judge={len(need_indices)}, empty_answer={len(empty_only)}")
            for idx in need_indices:
                rec = rows[idx]
                qid = rec.get("question_id")
                reason = (rec.get("score_details") or {}).get("reason") or rec.get("reason")
                print(f"  - line={idx}, question_id={qid}, reason={reason}")
            continue

        fixed = 0
        for idx in need_indices:
            rec = rows[idx]
            raw = rec.get("model_output_raw") or rec.get("model_raw_response")
            if _is_empty_text(raw):
                LOGGER.warning("Skip judge (empty answer) question_id=%s", rec.get("question_id"))
                continue

            judge_ok, meta = await _call_judge(
                caller,
                judge_cfg,
                rec,
                raw,
                timeout_seconds=int(args.timeout_seconds),
            )
            sd = rec.get("score_details") or {}
            rec["score_details"] = sd
            sd.update(meta or {})

            sd["method"] = sd.get("method") or "llm_judge"
            sd["reason"] = "judge_label" if judge_ok is not None else "judge_missing"
            rec["correctness_source"] = "judge_llm"
            rec["correctness_reason"] = sd["reason"]
            rec["is_correct"] = judge_ok
            if sd.get("judge_label") == "broken_item" and rec.get("item_status") not in {"broken_static", "skip_core"}:
                rec["item_status"] = "broken_judge"
            _set_soft_score(rec)
            fixed += 1

        with open(path, "w", encoding="utf-8") as out:
            for rec in rows:
                out.write(json.dumps(rec, ensure_ascii=False))
                out.write("\n")

        total_fixed += fixed
        print(f"{path.name}: judged {fixed}/{len(need_indices)} (empty answers skipped: {len(empty_only)}).")

    print(f"Done. Needs judge: {total_need}, fixed: {total_fixed}, empty answers: {total_empty_ans}.")


if __name__ == "__main__":
    asyncio.run(main())
