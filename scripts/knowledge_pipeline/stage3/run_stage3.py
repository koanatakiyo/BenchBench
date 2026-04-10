#!/usr/bin/env python3
"""
Stage 3 Answerer Panel runner for csbench_en.

This script loads Stage-2 question drops (after the Stage-3 post-hoc cleanup),
builds rigid answer prompts per question format, calls the configured answerer
models, parses their responses, scores them locally (exact/numeric), and stores
an answer matrix per (designer_model, answer_model) pair.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import math
import string
import sys
import time
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

# --------------------------------------------------------------------------- #
# Local imports (ensure scripts/ is on sys.path)
# --------------------------------------------------------------------------- #
SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from llm_manager import LLMManager  # noqa: E402


LOGGER = logging.getLogger("stage3.answerer_panel")
JUDGE_CORRECT_THRESHOLD = 0.9  # threshold to convert judge soft score to binary correctness

DEFAULT_MAX_NEW_TOKENS = {
    "default": 64,
    "mcq_single": 32,
    "mcq_multi": 48,
    "open_ended": 64,
    "structured": 64,
}

NUMERIC_PATTERN = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


# --------------------------------------------------------------------------- #
# Dataclasses for config
# --------------------------------------------------------------------------- #
@dataclass
class DesignerRunConfig:
    key: str
    designer_model: str
    provider_dir: Path
    question_file: Optional[Path] = None


@dataclass
class AnswerModelConfig:
    key: str
    display_name: str
    provider: str
    model: str
    batch_size: int = 32
    max_concurrent_requests: Optional[int] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: Dict[str, int] = field(default_factory=dict)
    extra_body: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def tokens_for(self, question_type: str) -> int:
        if question_type in self.max_new_tokens:
            return int(self.max_new_tokens[question_type])
        if "default" in self.max_new_tokens:
            return int(self.max_new_tokens["default"])
        return int(DEFAULT_MAX_NEW_TOKENS.get(question_type, DEFAULT_MAX_NEW_TOKENS["default"]))


# --------------------------------------------------------------------------- #
# Normalization helpers for open-ended scoring
# --------------------------------------------------------------------------- #
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[.,;:!?]+$", "", s)
    s = re.sub(r'^[\"\'“”‘’]+', "", s)
    s = re.sub(r'[\"\'“”‘’]+$', "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_math_expr(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(" ", "")
    s = s.replace("·", "*")
    s = s.replace("×", "*")
    return s


def score_open_ended(question: Dict[str, Any], parsed_answer: str) -> Tuple[bool, str]:
    gold = question.get("gold_answer", question.get("answer"))
    answer_type = (question.get("answer_type") or "term").lower()
    if answer_type in {"open_ended", "structured", "unknown"}:
        answer_type = "term"

    if parsed_answer is None or parsed_answer.strip() == "":
        return False, "empty"

    if answer_type == "numeric":
        try:
            gold_val = float(str(gold))
            nums = NUMERIC_PATTERN.findall(parsed_answer)
            pred_val = float(nums[0])
            ok = math.isclose(gold_val, pred_val, rel_tol=1e-6, abs_tol=1e-6)
            return ok, "numeric_exact" if ok else "numeric_mismatch"
        except Exception:
            return False, "numeric_parse_fail"

    if answer_type == "expression":
        g = normalize_math_expr(str(gold))
        p = normalize_math_expr(parsed_answer)
        if g == p:
            return True, "expr_exact"
        if g.startswith("o(") and p.startswith("o(") and g == p:
            return True, "expr_big_o_norm"
        return False, "expr_mismatch"

    if answer_type == "term":
        g = normalize_text(str(gold))
        p = normalize_text(parsed_answer)
        if p == g:
            return True, "term_exact"
        alias_map = {
            "fifo queue": ["first in first out queue", "fifo"],
            "lru": ["least recently used"],
        }
        if g in alias_map and p in alias_map[g]:
            return True, "term_alias"
        return False, "term_mismatch"

    if answer_type == "definition":
        g = normalize_text(str(gold))
        p = normalize_text(parsed_answer)
        key_phrases = [w for w in g.split() if len(w) > 3][:3]
        if all(k in p for k in key_phrases):
            return True, "def_heuristic"
        return False, "def_mismatch"

    return False, "unknown_type"


@dataclass
class Stage3DatasetConfig:
    dataset: str
    output_dir: Path
    designers: List[DesignerRunConfig]


@dataclass
class Stage3Config:
    dataset_configs: List[Stage3DatasetConfig]
    answer_models: List[AnswerModelConfig]
    judge_model: Optional[Dict[str, Any]] = None
    strong_models: List[str] = field(default_factory=list)
    dataset: Optional[str] = None  # legacy convenience
    output_dir: Optional[Path] = None  # legacy convenience

    @classmethod
    def from_yaml(cls, path: Path) -> "Stage3Config":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        stage3_section = payload.get("stage3") or {}
        base_dir_raw = stage3_section.get("current_dir")
        base_dir = Path(base_dir_raw or path.parent).expanduser()
        if not base_dir.is_absolute():
            base_dir = (path.parent / base_dir).resolve()
        else:
            base_dir = base_dir.resolve()

        def _resolve_path(value: Union[str, Path]) -> Path:
            p = Path(os.path.expandvars(str(value))).expanduser()
            if p.is_absolute():
                return p.resolve()
            return (base_dir / p).resolve()

        def _build_designers(designers_raw: Dict[str, Any]) -> List[DesignerRunConfig]:
            designers: List[DesignerRunConfig] = []
            for key, cfg in designers_raw.items():
                provider_dir = _resolve_path(cfg["provider_dir"])
                question_file = cfg.get("question_file")
                designers.append(
                    DesignerRunConfig(
                        key=str(key),
                        designer_model=cfg.get("designer_model", key),
                        provider_dir=provider_dir,
                        question_file=_resolve_path(question_file) if question_file else None,
                    )
                )
            return designers

        dataset_entries = stage3_section.get("datasets")
        dataset_configs: List[Stage3DatasetConfig] = []

        if dataset_entries:
            if isinstance(dataset_entries, dict):
                dataset_entries = [
                    {"dataset": name, **cfg} for name, cfg in dataset_entries.items()
                ]
            if not isinstance(dataset_entries, list):
                raise ValueError("stage3.datasets must be a list or mapping.")

            for entry in dataset_entries:
                dataset_name = entry.get("dataset") or entry.get("name")
                if not dataset_name:
                    raise ValueError("Each entry in stage3.datasets needs a dataset/name.")
                designers_raw = entry.get("designers") or {}
                if not designers_raw:
                    raise ValueError(f"No designers listed for dataset {dataset_name}.")
                designers = _build_designers(designers_raw)
                output_dir = Path(
                    entry.get(
                        "output_dir",
                        Path("outputs") / "stage3_answers" / dataset_name,
                    )
                )
                output_dir = _resolve_path(output_dir)
                dataset_configs.append(
                    Stage3DatasetConfig(
                        dataset=dataset_name,
                        output_dir=output_dir,
                        designers=designers,
                    )
                )
        else:
            dataset = payload.get("dataset")
            if not dataset:
                raise ValueError("Config missing top-level 'dataset'.")
            output_dir = _resolve_path(
                stage3_section.get(
                    "output_dir",
                    Path("outputs") / "stage3_answers" / dataset,
                )
            )

            designers_raw = stage3_section.get("designers") or {}
            if not designers_raw:
                raise ValueError("No designers listed under stage3.designers.")

            designers = _build_designers(designers_raw)
            dataset_configs.append(
                Stage3DatasetConfig(
                    dataset=dataset,
                    output_dir=output_dir,
                    designers=designers,
                )
            )

        answerers_raw = stage3_section.get("answer_models") or {}
        if not answerers_raw:
            raise ValueError("No answer_models listed under stage3.answer_models.")

        judge_model = stage3_section.get("judge_model") or {}
        strong_models_raw = stage3_section.get("strong_models") or []

        answerers: List[AnswerModelConfig] = []
        for key, cfg in answerers_raw.items():
            max_tokens = dict(DEFAULT_MAX_NEW_TOKENS)
            max_tokens.update(cfg.get("max_new_tokens") or {})
            answerers.append(
                AnswerModelConfig(
                    key=str(key),
                    display_name=cfg.get("display_name", key),
                    provider=cfg.get("provider", "").lower(),
                    model=cfg.get("model"),
                    batch_size=int(cfg.get("batch_size", 32)),
                    max_concurrent_requests=(
                        int(cfg["max_concurrent_requests"])
                        if cfg.get("max_concurrent_requests") is not None
                        else None
                    ),
                    temperature=float(cfg.get("temperature", 0.0)),
                    top_p=float(cfg.get("top_p", 1.0)),
                    max_new_tokens=max_tokens,
                    extra_body=cfg.get("extra_body") or {},
                    enabled=bool(cfg.get("enabled", True)),
                )
            )

        return cls(
            dataset_configs=dataset_configs,
            answer_models=answerers,
            judge_model=judge_model or None,
            strong_models=[str(item) for item in strong_models_raw],
            dataset=dataset_configs[0].dataset if dataset_configs else None,
            output_dir=dataset_configs[0].output_dir if dataset_configs else None,
        )


# --------------------------------------------------------------------------- #
# Stage-2 question normalization (Step-0 prep)
# --------------------------------------------------------------------------- #
REQUIRED_STAGE2_FIELDS = [
    "question_id",
    "super_parent",
    "subdomain",
    "design_type",
    "question_type",
    "declared_difficulty",
    "answer_type",
    "language",
    "visual_condition",
    "generation_index",
    "gold_answer",
]


def _ensure_quality_fields(record: Dict[str, Any]) -> None:
    record.setdefault("item_status", "candidate")
    flags = record.get("item_quality_flags")
    if flags is None:
        flags = []
    record["item_quality_flags"] = list(flags)
    sources = record.get("item_quality_source")
    if sources is None:
        sources = []
    if isinstance(sources, str):
        sources = [sources]
    if "static_rules" not in sources:
        sources.append("static_rules")
    record["item_quality_source"] = sources


def _infer_answer_type(question: Dict[str, Any]) -> str:
    if question.get("answer_type"):
        return str(question["answer_type"])
    qtype = (question.get("question_type") or "").lower()
    answer = question.get("answer")
    if isinstance(answer, (int, float)):
        return "numeric"
    if isinstance(answer, list) or qtype == "mcq_multi":
        return "multi_choice"
    if qtype == "mcq_single":
        return "single_choice"
    if qtype == "structured":
        return "structured"
    return "open_ended"


def _infer_visual_condition(question: Dict[str, Any]) -> str:
    modality = (question.get("modality") or "").lower()
    if "existing_image" in modality:
        return "text+existing_image"
    if "imagined_image" in modality:
        return "text+imagined_image"
    if question.get("uses_visual") is True:
        return "text+visual"
    return "text_only"


def normalize_question_record(question: Dict[str, Any], generation_index: int) -> Dict[str, Any]:
    record = dict(question)
    question_id = record.get("question_id") or record.get("id")
    if not question_id:
        question_id = f"q_{generation_index:04d}"
    record["question_id"] = question_id
    record.setdefault("id", question_id)

    gold_answer = record.get("gold_answer", record.get("answer"))
    record["gold_answer"] = gold_answer
    record["answer"] = gold_answer

    record["answer_type"] = _infer_answer_type(record)
    # Normalize answer_type for open-ended/structured to avoid "unknown_type" during scoring
    if record["answer_type"] in {None, "", "open_ended", "structured", "unknown"}:
        record["answer_type"] = "term"
    record["visual_condition"] = record.get("visual_condition") or _infer_visual_condition(record)
    record["generation_index"] = record.get("generation_index", generation_index)
    record["language"] = record.get("language") or "en"
    record["declared_difficulty"] = record.get("declared_difficulty") or record.get("difficulty") or ""

    missing = [field for field in REQUIRED_STAGE2_FIELDS if field not in record or record.get(field) in (None, "")]
    if missing:
        LOGGER.warning("Question %s is missing required fields: %s", question_id, ", ".join(sorted(missing)))
    return record


def _normalize_stem(stem: Optional[str]) -> str:
    if stem is None:
        return ""
    return normalize_text(stem)


def apply_static_quality_rules(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Pass 1 static quality flagging per designer drop."""
    from collections import Counter

    stem_counts = Counter(_normalize_stem(q.get("question_stem")) for q in questions if q.get("question_stem"))
    flagged: List[Dict[str, Any]] = []

    for q in questions:
        _ensure_quality_fields(q)
        flags: List[str] = q["item_quality_flags"]
        status = q.get("item_status", "candidate")
        fatal = status == "broken_static"

        # Structural validity
        missing = [f for f in REQUIRED_STAGE2_FIELDS if q.get(f) in (None, "")]
        if missing:
            flags.append("missing_fields")
            fatal = True
        if not q.get("question_stem"):
            flags.append("empty_stem")
            fatal = True
        if q.get("gold_answer", q.get("answer")) in (None, ""):
            flags.append("missing_answer")
            fatal = True

        # Option sanity
        qtype = (q.get("question_type") or "").lower()
        options = q.get("options", [])
        if qtype in {"mcq_single", "mcq_multi"}:
            if not isinstance(options, list) or len(options) < 2:
                flags.append("option_sanity_failed")
                fatal = True
            ans = q.get("gold_answer", q.get("answer"))
            if qtype == "mcq_single" and not isinstance(ans, str):
                flags.append("answer_type_mismatch")
                fatal = True
            if qtype == "mcq_multi" and not isinstance(ans, (list, tuple)):
                flags.append("answer_type_mismatch")
                fatal = True
        else:
            # Non-MCQ should not carry option lists
            if isinstance(options, list) and len(options) > 0:
                flags.append("options_present_non_mcq")

        # Explanation length
        expl = q.get("answer_explanation") or ""
        if len(expl.strip()) < 20:
            flags.append("short_explanation")

        q["item_status"] = "broken_static" if fatal else "candidate"
        q["item_quality_flags"] = sorted(set(flags))

        flagged.append(q)

    # Duplicate stems
    for q in flagged:
        stem_norm = _normalize_stem(q.get("question_stem"))
        if stem_norm and stem_counts.get(stem_norm, 0) > 1:
            flags = q.get("item_quality_flags") or []
            flags.append("duplicate_stem")
            q["item_quality_flags"] = sorted(set(flags))

    return flagged


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def slugify(value: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in value)
    return "_".join(filter(None, cleaned.split("_")))


def chunked(seq: Sequence[Any], size: int) -> Iterable[List[Any]]:
    size = max(1, size)
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


def normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    text = value.strip().lower()
    text = text.replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    if text.endswith((".", "!", "?")):
        text = text[:-1]
    return text.strip()


def extract_numeric(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    match = NUMERIC_PATTERN.search(value)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


# --------------------------------------------------------------------------- #
# Answer model caller
# --------------------------------------------------------------------------- #
class AnswerModelCaller:
    def __init__(self, llm: LLMManager):
        self.llm = llm
        # Rate-limit Gemini to avoid RPM exhaustion (target ~60 RPM => ~1.1s between calls)
        self._gemini_last_ts: float = 0.0
        self._gemini_semaphore = asyncio.Semaphore(1)
        # Separate semaphore for judge use (also single-flight to respect RPM)
        self._gemini_judge_last_ts: float = 0.0
        self._gemini_judge_semaphore = asyncio.Semaphore(1)

    async def call(
        self,
        cfg: AnswerModelConfig,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        provider = cfg.provider
        if provider == "openai":
            return await self._call_openai(cfg, system_prompt, user_prompt, max_new_tokens)
        if provider == "gemini":
            return await self._call_gemini(cfg, system_prompt, user_prompt, max_new_tokens)
        if provider == "anthropic":
            return await self._call_anthropic(cfg, system_prompt, user_prompt, max_new_tokens)
        if provider == "grok":
            return await self._call_grok(cfg, system_prompt, user_prompt)
        if provider == "deepseek":
            return await self._call_openai_compatible(
                client=getattr(self.llm, "deepseek_client", None),
                model_name=cfg.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                extra_body=cfg.extra_body or getattr(self.llm, "deepseek_extra_body", None),
            )
        if provider == "qwen":
            return await self._call_openai_compatible(
                client=getattr(self.llm, "qwen_client", None),
                model_name=cfg.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
        if provider == "doubao":
            return await self._call_openai_compatible(
                client=getattr(self.llm, "doubao_client", None),
                model_name=cfg.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
        if provider == "llama":
            return await self._call_openai_compatible(
                client=getattr(self.llm, "llama_client", None),
                model_name=cfg.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_new_tokens=max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )
        if provider == "zai":
            return await self._call_zai(cfg, system_prompt, user_prompt, max_new_tokens)
        raise ValueError(f"Unsupported provider '{provider}' for answer model '{cfg.display_name}'.")

    async def _call_openai(
        self,
        cfg: AnswerModelConfig,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        if not getattr(self.llm, "openai_available", False):
            raise RuntimeError("OpenAI client is not initialized.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        client = self.llm.openai_client
        model_name = cfg.model

        def _collect_output_text_blocks(obj: Any, *, _limit: int = 20) -> List[str]:
            """
            Conservative extractor: only collect text from 'output_text' blocks/parts.
            Avoids accidentally collecting config metadata (e.g., text: disabled).
            """
            out: List[str] = []

            def _walk(node: Any) -> None:
                if len(out) >= _limit:
                    return
                if node is None:
                    return
                if isinstance(node, dict):
                    node_type = node.get("type")
                    if node_type == "output_text":
                        tv = node.get("text")
                        if isinstance(tv, str) and tv.strip():
                            out.append(tv.strip())
                        elif isinstance(tv, dict):
                            v = tv.get("value") or tv.get("text")
                            if isinstance(v, str) and v.strip():
                                out.append(v.strip())
                    for v in node.values():
                        _walk(v)
                    return
                if isinstance(node, (list, tuple)):
                    for item in node:
                        _walk(item)
                    return
                if hasattr(node, "model_dump"):
                    try:
                        _walk(node.model_dump())
                        return
                    except Exception:
                        return

            _walk(obj)
            # de-dup while preserving order
            seen = set()
            deduped: List[str] = []
            for s in out:
                if s in seen:
                    continue
                seen.add(s)
                deduped.append(s)
            return deduped

        if "gpt-5" in (model_name or "").lower():
            system_block = (
                [{"type": "input_text", "text": system_prompt}] if system_prompt else []
            )
            input_payload = []
            if system_block:
                input_payload.append({"role": "system", "content": system_block})
            input_payload.append(
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]}
            )
            response = await client.responses.create(
                model=model_name,
                input=input_payload,
                max_output_tokens=max_new_tokens,
                **(cfg.extra_body or {}),
            )
            # Extract text robustly from Responses API
            # 1) Preferred: output_text
            if getattr(response, "output_text", None):
                return response.output_text
            texts: List[str] = []
            # 2) Gather from output blocks
            output_blocks = getattr(response, "output", None)
            if isinstance(output_blocks, list):
                for block in output_blocks:
                    content = None
                    if isinstance(block, dict):
                        content = block.get("content")
                    elif hasattr(block, "content"):
                        content = getattr(block, "content")
                    if isinstance(content, list):
                        for item in content:
                            text_val = None
                            if isinstance(item, dict):
                                text_val = item.get("text")
                                if not text_val and item.get("type") in {"output_text", "input_text"}:
                                    text_val = item.get("text")
                            elif hasattr(item, "text"):
                                text_val = getattr(item, "text")
                            if text_val:
                                texts.append(text_val)
                    elif isinstance(content, str) and content:
                        texts.append(content)
                    elif hasattr(block, "text"):
                        text_val = getattr(block, "text")
                        if text_val:
                            texts.append(text_val)
            elif isinstance(output_blocks, str) and output_blocks:
                texts.append(output_blocks)

            # 3) Fallback to choices if present
            if not texts and getattr(response, "choices", None):
                try:
                    texts.append(response.choices[0].message.content)
                except Exception:
                    pass

            # 4) Last-chance: scan only output_text blocks in the SDK object
            if not texts:
                try:
                    texts = _collect_output_text_blocks(response)
                except Exception:
                    texts = []

            if texts:
                return "\n".join(texts)
            # Helpful debug breadcrumb (avoid dumping the whole response object)
            try:
                resp_id = getattr(response, "id", None)
                resp_status = getattr(response, "status", None)
                LOGGER.warning(
                    "OpenAI Responses API returned no extractable text (model=%s, id=%s, status=%s). Falling back to chat.completions.",
                    model_name,
                    resp_id,
                    resp_status,
                )
            except Exception:
                LOGGER.warning(
                    "OpenAI Responses API returned no extractable text (model=%s). Falling back to chat.completions.",
                    model_name,
                )
            # Fallback: try chat.completions if Responses returned no text
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            chat_kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": messages,
            }
            # GPT-5 via chat.completions requires max_completion_tokens (OpenAI returns 400 otherwise)
            chat_kwargs["max_completion_tokens"] = max_new_tokens
            if cfg.extra_body:
                chat_kwargs.update(cfg.extra_body)
            chat_resp = await client.chat.completions.create(**chat_kwargs)
            fallback_text = chat_resp.choices[0].message.content
            if fallback_text and fallback_text.strip():
                return fallback_text
            # As a last resort, return empty string to avoid exception
            return fallback_text or ""

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            **(cfg.extra_body or {}),
        )
        return response.choices[0].message.content

    async def _call_gemini(
        self,
        cfg: AnswerModelConfig,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        *,
        _retried: bool = False,
    ) -> str:
        if not getattr(self.llm, "gemini_available", False):
            raise RuntimeError("Gemini client is not initialized.")

        async with self._gemini_semaphore:
            now = time.perf_counter()
            gap = now - self._gemini_last_ts
            if gap < 1.1:
                await asyncio.sleep(1.1 - gap)
            self._gemini_last_ts = time.perf_counter()

        combined_prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}" if system_prompt else user_prompt

        # Attempt to include safety settings if available to reduce silent blocks
        safety_settings = None
        if hasattr(self.llm, "_build_gemini_safety_settings"):
            try:
                safety_settings = self.llm._build_gemini_safety_settings()
            except Exception:
                safety_settings = None

        config_kwargs = {
            "max_output_tokens": max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
        }
        if safety_settings:
            config_kwargs["safety_settings"] = safety_settings

        config = self.llm.genai_types.GenerateContentConfig(**config_kwargs)
        response = await asyncio.to_thread(
            self.llm.gemini_client.models.generate_content,
            model=cfg.model or getattr(self.llm, "gemini_model_name", None),
            contents=combined_prompt,
            config=config,
        )

        primary_text = getattr(response, "text", None)
        if primary_text:
            return primary_text

        candidates = getattr(response, "candidates", None) or []
        truncated = False
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                part = content.parts[0]
                text = getattr(part, "text", None)
                if text:
                    return text
            finish_reason = getattr(candidate, "finish_reason", None)
            if str(getattr(finish_reason, "name", finish_reason)).upper() == "MAX_TOKENS":
                truncated = True

        # ADD DEBUGGING HERE
        debug_bits = []
        
        # Check prompt feedback (why prompt was blocked)
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason:
                debug_bits.append(f"prompt_block={getattr(block_reason, 'name', block_reason)}")
            safety_ratings = getattr(prompt_feedback, "safety_ratings", None)
            if safety_ratings:
                ratings = ", ".join(
                    f"{getattr(r.category, 'name', r.category)}:{getattr(r.threshold, 'name', r.threshold)}"
                    for r in safety_ratings
                )
                debug_bits.append(f"prompt_safety=[{ratings}]")
        
        # Check candidate finish reasons
        for idx, cand in enumerate(candidates):
            finish_reason = getattr(cand, "finish_reason", None)
            if finish_reason:
                debug_bits.append(f"cand{idx}_finish={getattr(finish_reason, 'name', finish_reason)}")
            
            # Check if candidate has function calls instead of text
            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                parts = getattr(cand.content, "parts", None)
                if parts is not None:
                    for part_idx, part in enumerate(parts):
                        if hasattr(part, "function_call"):
                            debug_bits.append(f"cand{idx}_part{part_idx}_has_function_call")
            
            safety_ratings = getattr(cand, "safety_ratings", None)
            if safety_ratings:
                ratings = ", ".join(
                    f"{getattr(r.category, 'name', r.category)}:{getattr(r.probability, 'name', r.probability)}"
                    for r in safety_ratings
                )
                debug_bits.append(f"cand{idx}_safety=[{ratings}]")
        
        debug_msg = " | ".join(debug_bits) if debug_bits else "no response metadata"
        logging.warning(
            f"Gemini response did not include text content; returning empty string. "
            f"Debug info: {debug_msg}"
        )
        if truncated and not _retried:
            # Retry once with a larger budget (cap raised to 2048 to reduce repeated retries)
            new_tokens = min(int(max_new_tokens * 2), 2048)
            logging.info(
                "Retrying Gemini call with higher max_new_tokens (%d -> %d) due to truncation.",
                max_new_tokens,
                new_tokens,
            )
            return await self._call_gemini(cfg, system_prompt, user_prompt, new_tokens, _retried=True)
        # Graceful fallback: return empty string instead of raising, so the call
        # is logged as an incorrect answer rather than stopping the run.
        # logging.warning("Gemini response did not include text content; returning empty string.")
        return ""

    async def _call_anthropic(
        self,
        cfg: AnswerModelConfig,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        if not getattr(self.llm, "anthropic_available", False):
            raise RuntimeError("Anthropic client is not initialized.")

        message = {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}],
        }
        return await self.llm.call_anthropic_messages(
            messages=[message],
            system_prompt=system_prompt or None,
            max_tokens=max_new_tokens,
            temperature=cfg.temperature,
            model_name=cfg.model or getattr(self.llm, "anthropic_model_name", None),
        )

    async def _call_grok(
        self,
        cfg: AnswerModelConfig,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        if not getattr(self.llm, "grok_available", False):
            raise RuntimeError("Grok client is not initialized.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        payload = {
            "model": cfg.model or getattr(self.llm, "grok_model_name", None),
            "messages": messages,
            "stream": False,
            "temperature": cfg.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.llm.grok_api_key}",
        }
        timeout = getattr(self.llm, "config", {}).get("grok", {}).get("timeout", 120.0)
        response = await self.llm.grok_client.post(
            f"{self.llm.grok_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"]
        raise RuntimeError(f"Grok response missing choices: {data}")

    async def _call_openai_compatible(
        self,
        client: Any,
        *,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        if client is None:
            raise RuntimeError("Requested provider client is not initialized.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = await client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    async def _call_zai(
        self,
        cfg: "AnswerModelConfig",
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        """Call ZhipuAI (GLM) via the ``zai`` SDK.

        The ZhipuAiClient exposes a synchronous ``chat.completions.create``
        that mirrors the OpenAI interface, so we wrap it in ``asyncio.to_thread``
        to keep the event loop responsive.
        """
        client = getattr(self.llm, "zai_client", None)
        if client is None:
            raise RuntimeError("ZhipuAI (zai) client is not initialized.")

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        def _sync_call():
            response = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=cfg.temperature,
                thinking={"type": "disabled"},
            )
            return response.choices[0].message.content

        return await asyncio.to_thread(_sync_call)


# --------------------------------------------------------------------------- #
# Prompt builder and scoring logic
# --------------------------------------------------------------------------- #
class PromptBuilder:
    LETTERS = list(string.ascii_uppercase)

    @staticmethod
    def build(question: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        qtype = question.get("question_type", "open_ended")
        stem = question.get("question_stem", "").strip()
        options = [opt for opt in question.get("options") or [] if isinstance(opt, str) and opt.strip()]
        letters = PromptBuilder.LETTERS[: len(options)]

        if qtype == "mcq_single":
            system = "You are taking a exam. Choose exactly one correct option."
            options_block = "\n".join(
                f"{letter}. {text}"
                for letter, text in zip(letters, options)
            )
            allowed = ", ".join(letters) if letters else "A, B, C, D"
            user = (
                f"Question:\n{stem}\n\nOptions:\n{options_block}\n\n"
                f"Please answer with only one capital letter ({allowed}) and nothing else."
            )
            return system, user, letters

        if qtype == "mcq_multi":
            system = "You are taking a exam. Some questions may have multiple correct options."
            options_block = "\n".join(
                f"{letter}. {text}"
                for letter, text in zip(letters, options)
            )
            user = (
                f"Question:\n{stem}\n\nOptions:\n{options_block}\n\n"
                'Return the correct options as a JSON array of capital letters.\n'
                'Examples: ["A"], ["A","C"], ["B","D","E"].\n'
                "Answer with only the JSON array and nothing else."
            )
            return system, user, letters

        if qtype == "structured":
            system = (
                "You are taking a structured computer science exam. "
                "Follow the requested output format exactly."
            )
            user = (
                f"Question:\n{stem}\n\n"
                "Produce your answer in the requested structure only "
                "(e.g., a numbered list, table schema, or JSON) with no additional commentary."
            )
            return system, user, []

        # open_ended and fallback
        system = "You are taking a short-answer exam. Give concise, direct answers."
        user = (
            f"Question:\n{stem}\n\n"
            "Answer in one sentence or a short expression. Do not show your reasoning, just the final answer."
        )
        return system, user, []


class AnswerParser:
    @staticmethod
    def parse(question: Dict[str, Any], raw: Optional[str], letters: List[str]) -> Any:
        if raw is None:
            return None
        text = raw.strip()
        qtype = question.get("question_type", "open_ended")
        if qtype == "mcq_single":
            return AnswerParser._parse_mcq_single(text, letters)
        if qtype == "mcq_multi":
            return AnswerParser._parse_mcq_multi(text, letters)
        return text

    @staticmethod
    def _parse_mcq_single(text: str, letters: List[str]) -> Optional[str]:
        valid = set(letters) if letters else set(PromptBuilder.LETTERS[:5])
        for char in text.upper():
            if char in valid:
                return char
        return None

    @staticmethod
    def _parse_mcq_multi(text: str, letters: List[str]) -> Optional[List[str]]:
        valid = set(letters) if letters else set(PromptBuilder.LETTERS[:5])
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                cleaned = []
                for item in parsed:
                    if isinstance(item, str):
                        letter = item.strip().upper()
                        if letter in valid and letter not in cleaned:
                            cleaned.append(letter)
                return sorted(cleaned)
        except json.JSONDecodeError:
            pass
        fallback = []
        for char in text.upper():
            if char in valid and char not in fallback:
                fallback.append(char)
        return sorted(fallback) if fallback else None


class AnswerScorer:
    @staticmethod
    def _count_tokens(text: str) -> int:
        return len(re.findall(r"\w+", text))

    @staticmethod
    def _count_sentences(text: str) -> int:
        return len([s for s in re.split(r"[.!?]+", text) if s.strip()])

    @staticmethod
    def select_scoring_profile(question: Dict[str, Any]) -> str:
        qtype = (question.get("question_type") or "").lower()
        gold = str(question.get("gold_answer", question.get("answer")) or "").strip()

        if qtype in {"mcq_single", "mcq_multi"}:
            return "mcq_exact"

        if qtype in {"open_ended", "structured", "cloze", "judgment"}:
            if re.fullmatch(r"[\s\d\.,eE+\-*/^()]+", gold):
                return "numeric_exact"
            tokens = AnswerScorer._count_tokens(gold)
            if tokens <= 3:
                return "symbolic_exact"
            # if any(tok in gold for tok in ["{", "}", ";", "def ", "class ", "function", "return ", "console.log", "import "]):
            #     return "code_exec"
            if tokens > 80 or AnswerScorer._count_sentences(gold) > 3:
                return "skip_core"
            return "llm_judge"

        return "mcq_exact"
    @staticmethod
    def _normalize_choice(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip().upper()
        return text or None

    @classmethod
    def _normalize_choice_list(cls, value: Any) -> List[str]:
        if isinstance(value, (list, tuple, set)):
            letters = [cls._normalize_choice(item) for item in value]
        else:
            letters = [cls._normalize_choice(value)]
        return sorted({letter for letter in letters if letter})

    @staticmethod
    def _is_empty_prediction(parsed_answer: Any) -> bool:
        if parsed_answer is None:
            return True
        if isinstance(parsed_answer, str):
            return parsed_answer.strip() == ""
        if isinstance(parsed_answer, (list, tuple, set)):
            return len(parsed_answer) == 0
        return False

    @staticmethod
    def score(
        question: Dict[str, Any],
        parsed_answer: Any,
        *,
        scoring_profile: Optional[str] = None,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        qtype = (question.get("question_type") or "open_ended").lower()
        gold = question.get("gold_answer", question.get("answer"))

        profile = scoring_profile or AnswerScorer.select_scoring_profile(question)
        details: Dict[str, Any] = {"method": profile}

        if gold is None:
            details["reason"] = "missing_gold"
            return False, details

        # MCQ_single: exact letter match
        if qtype == "mcq_single" or profile == "mcq_exact":
            details["method"] = "mcq_single_exact"
            pred = AnswerScorer._normalize_choice(parsed_answer)
            g = AnswerScorer._normalize_choice(gold)
            details["pred_normalized"] = pred
            details["gold_normalized"] = g
            if pred is None:
                details["reason"] = "empty_pred"
                return False, details
            if g is None:
                details["reason"] = "missing_gold"
                return False, details
            ok = pred == g
            details["reason"] = "mcq_exact_match" if ok else "mcq_mismatch"
            return ok, details

        # MCQ_multi: set match
        if qtype == "mcq_multi" or profile == "mcq_exact":
            details["method"] = "mcq_multi_set"
            pred_list = AnswerScorer._normalize_choice_list(parsed_answer)
            if not pred_list:
                details["reason"] = "empty_pred"
                return False, details

            gold_value: Any = gold
            if isinstance(gold_value, str):
                # tolerate gold stored as JSON string
                try:
                    gold_value = json.loads(gold_value)
                except Exception:
                    gold_value = gold_value
            gold_list = AnswerScorer._normalize_choice_list(gold_value)
            details["pred_normalized"] = pred_list
            details["gold_normalized"] = gold_list
            if not gold_list:
                details["reason"] = "missing_gold"
                return False, details
            ok = set(pred_list) == set(gold_list)
            details["reason"] = "mcq_multi_set_match" if ok else "mcq_multi_set_mismatch"
            return ok, details

        if AnswerScorer._is_empty_prediction(parsed_answer):
            details["reason"] = "empty_pred"
            return False, details

        if profile == "numeric_exact":
            details["method"] = "numeric_exact"
            pred_value = extract_numeric(str(parsed_answer))
            gold_value = extract_numeric(str(gold))
            if pred_value is None or gold_value is None:
                details["reason"] = "numeric_parse_fail"
                return False, details
            tolerance = max(1e-6, abs(gold_value) * 1e-4)
            ok = abs(pred_value - gold_value) <= tolerance
            details["tolerance"] = tolerance
            details["reason"] = "numeric_exact" if ok else "numeric_mismatch"
            return ok, details

        if profile == "symbolic_exact":
            details["method"] = "symbolic_exact"
            p = normalize_text(str(parsed_answer))
            g = normalize_text(str(gold))
            ok = p == g and g != ""
            details["pred_normalized"] = p
            details["gold_normalized"] = g
            details["reason"] = "symbolic_exact_match" if ok else "symbolic_exact_mismatch"
            return ok, details

        if profile == "code_exec":
            details["method"] = "code_exec"
            p = normalize_text(str(parsed_answer))
            g = normalize_text(str(gold))
            ok = p == g and g != ""
            details["pred_normalized"] = p
            details["gold_normalized"] = g
            details["reason"] = "code_text_match" if ok else "code_text_mismatch"
            return ok, details

        if profile == "skip_core":
            details["method"] = "skip_core"
            details["reason"] = "skip_core"
            return None, details

        if profile == "llm_judge":
            details["reason"] = "llm_judge_pending"
            return None, details

        # Fallback: heuristic open-ended term/definition
        details["method"] = "open_ended_heuristic"
        ok, reason = score_open_ended(question, str(parsed_answer))
        details["reason"] = reason
        return bool(ok), details


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class Stage3AnswererPanel:
    def __init__(
        self,
        config: Stage3Config,
        *,
        dataset_override: Optional[str] = None,
        designer_filter: Optional[List[str]] = None,
        answerer_filter: Optional[List[str]] = None,
        limit_per_designer: Optional[int] = None,
        dry_run: bool = False,
        skip_existing: bool = False,
        skip_dynamic: bool = False,
    ):
        self.config = config
        dataset_filter_value = dataset_override or config.dataset
        self.limit_per_designer = limit_per_designer
        self.dry_run = dry_run
        self.skip_existing = skip_existing
        self.skip_dynamic = skip_dynamic
        self.judge_model = config.judge_model
        self.judge_config = self._build_judge_config(config.judge_model) if config.judge_model else None
        self.strong_models = config.strong_models

        def _variants(s: str) -> set[str]:
            sl = slugify(s)
            return {
                s,
                s.lower(),
                sl,
                sl.replace("_", ""),
                s.replace("-", "").replace("_", "").lower(),
            }

        designer_filter_set = {slugify(item) for item in designer_filter or []}
        answerer_filter_set = {slugify(item) for item in answerer_filter or []}
        answerer_filter_raw = {str(item).strip().lower() for item in answerer_filter or []}
        self.designers_filter = designer_filter_set
        self.answerers_filter = answerer_filter_set
        self.designer_slug_whitelist: set[str] = set()
        self.answerer_slug_whitelist: set[str] = set()

        dataset_filter_set = {slugify(dataset_filter_value)} if dataset_filter_value else set()
        self.dataset_configs = [
            ds
            for ds in config.dataset_configs
            if not dataset_filter_set or slugify(ds.dataset) in dataset_filter_set
        ]

        self.answerers = [
            a
            for a in config.answer_models
            if a.enabled
            and (
                not answerer_filter_set
                or _variants(a.key) & answerer_filter_raw
                or _variants(a.display_name or "") & answerer_filter_raw
                or slugify(a.key) in answerer_filter_set
                or slugify(a.display_name) in answerer_filter_set
            )
        ]

        # Build slug whitelists to support matching filenames in dynamic pass
        self.designer_slug_whitelist = {
            slugify(d.key) for ds in self.dataset_configs for d in ds.designers
        } | {
            slugify(d.designer_model) for ds in self.dataset_configs for d in ds.designers
        }
        self.answerer_slug_whitelist = {
            slugify(a.key) for a in config.answer_models
        } | {slugify(a.display_name) for a in config.answer_models}

        if not self.answerers:
            available = [a.key for a in config.answer_models if a.enabled]
            raise ValueError(
                f"No answerer models selected after filtering. "
                f"Available keys: {available}. "
                f"Provided filter: {list(answerer_filter or [])}"
            )
        if not self.dataset_configs:
            raise ValueError("No datasets selected after filtering.")

        providers_needed = self._collect_providers(self.answerers, self.judge_config)
        self.llm = LLMManager(models_to_init=providers_needed)
        self.caller = AnswerModelCaller(self.llm)
        self.prompt_builder = PromptBuilder()
        self.parser = AnswerParser()

        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        for answerer in self.answerers:
            limit = answerer.max_concurrent_requests or answerer.batch_size
            self._semaphores[answerer.key] = asyncio.Semaphore(max(1, limit))

    @staticmethod
    def _normalize_language(lang: Optional[str]) -> Optional[str]:
        if not lang:
            return lang
        norm_map = {
            "english": "en",
            "en": "en",
            "zh": "zh",
            "chinese": "zh",
            "cn": "zh",
            "fr": "fr",
            "french": "fr",
            "es": "es",
            "spanish": "es",
            "de": "de",
            "german": "de",
            "ja": "ja",
            "japanese": "ja",
            "ko": "ko",
            "korean": "ko",
        }
        key = str(lang).strip().lower()
        return norm_map.get(key, key)

    @staticmethod
    def _set_soft_score(row: Dict[str, Any]) -> Dict[str, Any]:
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
        return row

    @staticmethod
    def _collect_providers(answerers: List[AnswerModelConfig], judge_cfg: Optional[AnswerModelConfig]) -> List[str]:
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
        providers = {provider_map[a.provider] for a in answerers if a.provider in provider_map}
        if judge_cfg and judge_cfg.provider in provider_map:
            providers.add(provider_map[judge_cfg.provider])
        return sorted(providers)

    @staticmethod
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

    async def run(self) -> None:
        if self.judge_model:
            LOGGER.info("Judge model configured for rare checks: %s", self.judge_model.get("model") or self.judge_model)
        if self.strong_models:
            LOGGER.info("Strong model set for quality checks: %s", ", ".join(self.strong_models))

        for dataset_cfg in self.dataset_configs:
            designers = [
                d
                for d in dataset_cfg.designers
                if not self.designers_filter or slugify(d.key) in self.designers_filter or slugify(d.designer_model) in self.designers_filter
            ] if hasattr(self, "designers_filter") else dataset_cfg.designers

            if not designers:
                LOGGER.warning("No designers selected for dataset %s after filtering.", dataset_cfg.dataset)
                continue

            total_pairs = len(designers) * len(self.answerers)
            LOGGER.info(
                "Starting Stage-3 answering for %s: %d designers x %d answerers = %d pairs.",
                dataset_cfg.dataset,
                len(designers),
                len(self.answerers),
                total_pairs,
            )

            output_dir = dataset_cfg.output_dir
            output_dir.mkdir(parents=True, exist_ok=True)

            for designer in designers:
                questions = self._load_questions(designer)
                if self.limit_per_designer:
                    questions = questions[: self.limit_per_designer]
                if not questions:
                    LOGGER.warning("Designer %s produced no questions for %s; skipping.", designer.designer_model, dataset_cfg.dataset)
                    continue

                for answerer in self.answerers:
                    await self._run_pair(dataset_cfg.dataset, output_dir, designer, answerer, questions)

            # Dynamic quality pass across all answerers for this dataset
            if not self.dry_run and not self.skip_dynamic:
                await self._run_dynamic_quality(dataset_cfg)

    def _load_questions(self, designer: DesignerRunConfig) -> List[Dict[str, Any]]:
        path = designer.question_file or self._infer_question_file(designer.provider_dir)
        LOGGER.info("Loading questions for %s from %s", designer.designer_model, path)
        questions: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    normalized = normalize_question_record(raw, len(questions))
                    questions.append(normalized)
                except json.JSONDecodeError as err:
                    LOGGER.error("JSON decode error in %s: %s", path, err)
        questions = apply_static_quality_rules(questions)
        broken = sum(1 for q in questions if q.get("item_status") == "broken_static")
        usable = [q for q in questions if q.get("item_status") != "broken_static"]
        if broken:
            LOGGER.warning(
                "Static quality pass flagged %d broken questions for %s; %d remain for answering.",
                broken,
                designer.designer_model,
                len(usable),
            )
        else:
            LOGGER.info("Static quality pass clean for %s (%d questions).", designer.designer_model, len(usable))
        return usable

    @staticmethod
    def _infer_question_file(provider_dir: Path) -> Path:
        preferred = sorted(provider_dir.glob("*_stage3_clean.jsonl"))
        if preferred:
            return preferred[0].resolve()
        legacy = sorted(provider_dir.glob("questions_*.jsonl"))
        if legacy:
            return legacy[0].resolve()
        raise FileNotFoundError(f"No question file detected under {provider_dir}")

    async def _run_pair(
        self,
        dataset_name: str,
        output_dir: Path,
        designer: DesignerRunConfig,
        answerer: AnswerModelConfig,
        questions: List[Dict[str, Any]],
    ) -> None:
        designer_slug = slugify(designer.designer_model)
        answerer_slug = slugify(answerer.display_name)
        filename = f"{dataset_name}_stage3_answers_{designer_slug}_vs_{answerer_slug}.jsonl"
        output_path = output_dir / filename

        if self.skip_existing and output_path.exists():
            LOGGER.info("Skipping %s vs %s (output exists).", designer.designer_model, answerer.display_name)
            return

        LOGGER.info(
            "Answering %d questions: dataset=%s, designer=%s, answerer=%s -> %s",
            len(questions),
            dataset_name,
            designer.designer_model,
            answerer.display_name,
            output_path,
        )

        if self.dry_run:
            LOGGER.info("Dry-run enabled; skipping API calls for %s vs %s.", designer.designer_model, answerer.display_name)
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        batch_size = max(1, answerer.batch_size)

        with open(output_path, "w", encoding="utf-8") as sink:
            for batch_idx, batch in enumerate(chunked(questions, batch_size)):
                call_batch_id = f"{answerer_slug}_vs_{designer_slug}_b{batch_idx:04d}"
                records = await self._answer_batch(
                    batch,
                    dataset_name,
                    output_dir,
                    designer,
                    answerer,
                    call_batch_id,
                    batch_idx,
                )
                for record in records:
                    sink.write(json.dumps(record, ensure_ascii=False))
                    sink.write("\n")

    async def _answer_batch(
        self,
        batch: List[Dict[str, Any]],
        dataset_name: str,
        output_dir: Path,
        designer: DesignerRunConfig,
        answerer: AnswerModelConfig,
        call_batch_id: str,
        batch_idx: int,
    ) -> List[Dict[str, Any]]:
        semaphore = self._semaphores[answerer.key]
        tasks = []
        for call_index, question in enumerate(batch):
            tasks.append(
                asyncio.create_task(
                    self._answer_single(
                        semaphore,
                        dataset_name,
                        output_dir,
                        question,
                        designer,
                        answerer,
                        call_batch_id,
                        batch_idx,
                        call_index,
                        len(batch),
                    )
                )
            )
        results = await asyncio.gather(*tasks)
        return results

    async def _answer_single(
        self,
        semaphore: asyncio.Semaphore,
        dataset_name: str,
        output_dir: Path,
        question: Dict[str, Any],
        designer: DesignerRunConfig,
        answerer: AnswerModelConfig,
        call_batch_id: str,
        batch_idx: int,
        call_index: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        system_prompt, user_prompt, letters = self.prompt_builder.build(question)
        max_tokens = answerer.tokens_for(question.get("question_type", "open_ended"))

        start = time.perf_counter()
        raw_response: Optional[str] = None
        error_message: Optional[str] = None

        async with semaphore:
            try:
                raw_response = await self.caller.call(
                    answerer,
                    system_prompt,
                    user_prompt,
                    max_tokens,
                )
            except Exception as err:
                error_message = f"{type(err).__name__}: {err}"
                LOGGER.error(
                    "Error answering question %s with %s: %s",
                    question.get("id"),
                    answerer.display_name,
                    error_message,
                )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        parsed_answer = self.parser.parse(question, raw_response, letters)
        scoring_question = dict(question)
        scoring_question.pop("scoring_method", None)
        scoring_profile = AnswerScorer.select_scoring_profile(scoring_question)
        is_correct, score_meta = AnswerScorer.score(scoring_question, parsed_answer, scoring_profile=scoring_profile)
        correctness_reason = score_meta.get("reason")
        correctness_source = "heuristic"

        judge_label: Optional[str] = None
        # For llm_judge profile, invoke judge LLM once
        if scoring_profile == "llm_judge" and is_correct is None:
            judge_ok, judge_meta = await self._call_llm_answer_judge(
                scoring_question,
                raw_response or "",
                self.judge_config,
            )
            is_correct = judge_ok
            score_meta["method"] = "llm_judge"
            score_meta["reason"] = "judge_label" if judge_ok is not None else "judge_missing"
            if judge_meta:
                judge_label = judge_meta.get("label")
                score_meta["judge_label"] = judge_meta.get("label")
                score_meta["judge_score"] = judge_meta.get("score")
                score_meta["judge_missing"] = judge_meta.get("missing")
                score_meta["judge_errors"] = judge_meta.get("errors")
                score_meta["judge_raw_response"] = judge_meta.get("raw_response")
            correctness_source = "judge_llm"
            correctness_reason = score_meta.get("reason")

        # Skip-core items: mark status so they are excluded from core matrices
        if scoring_profile == "skip_core":
            question_status = question.get("item_status") or "skip_core"
        else:
            question_status = question.get("item_status")
        if judge_label == "broken_item":
            question_status = "broken_judge"

        record: Dict[str, Any] = {
            "dataset": question.get("dataset") or question.get("source_dataset") or dataset_name,
            "designer_model": question.get("designer_model") or designer.designer_model,
            "designer_id": designer.key,
            "answer_model": answerer.display_name,
            "answer_model_id": answerer.key,
            "question_id": question.get("question_id") or question.get("id"),
            "super_parent": question.get("super_parent"),
            "subdomain": question.get("subdomain"),
            "question_type": question.get("question_type"),
            "design_type": question.get("design_type"),
            "declared_difficulty": question.get("declared_difficulty"),
            "answer_type": question.get("answer_type"),
            "language": question.get("language"),
            "visual_condition": question.get("visual_condition"),
            "generation_index": question.get("generation_index"),
            "modality": question.get("modality"),
            "question_stem": question.get("question_stem"),
            "options": question.get("options"),
            "gold_answer": question.get("gold_answer", question.get("answer")),
            "item_status": question_status,
            "item_quality_flags": question.get("item_quality_flags"),
            "item_quality_source": question.get("item_quality_source"),
            "model_output_raw": raw_response,
            "model_raw_response": raw_response,
            "parsed_answer": parsed_answer,
            "scoring_method": score_meta.get("method"),
            "score_details": score_meta,
            "is_correct": is_correct,
            "correctness_source": correctness_source,
            "correctness_reason": correctness_reason,
            "answer_time_ms": elapsed_ms,
            "call_batch_id": call_batch_id,
            "call_batch_index": batch_idx,
            "call_batch_size": batch_size,
            "call_index_in_batch": call_index,
            "output_dir": str(output_dir),
            "model_parameters": {
                "temperature": answerer.temperature,
                "top_p": answerer.top_p,
                "max_new_tokens": max_tokens,
            },
        }
        if error_message:
            record["call_error"] = error_message
            if is_correct is None:
                record["is_correct"] = False
        record = self._set_soft_score(record)
        return record

    # ----------------------------------------------------------------------- #
    # Dynamic quality (Pass-2) + judge calls
    # ----------------------------------------------------------------------- #
    def _filename_models(self, path: Path) -> Tuple[Optional[str], Optional[str]]:
        # Expected: {dataset}_stage3_answers_{designer}_vs_{answer}.jsonl
        parts = path.stem.split("_stage3_answers_")
        if len(parts) != 2:
            return None, None
        tail = parts[1]
        if "_vs_" not in tail:
            return None, None
        designer, answer = tail.split("_vs_", 1)
        return designer, answer

    @staticmethod
    def _safe_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _merge_list(base: Sequence[Any], extra: Sequence[Any]) -> List[Any]:
        merged = list(base) if base is not None else []
        for item in extra:
            if item not in merged:
                merged.append(item)
        return merged

    @staticmethod
    def _parse_json_blob(payload: str) -> Optional[Dict[str, Any]]:
        if not payload:
            return None
        stripped = payload.strip()
        # Try direct parse
        try:
            return json.loads(stripped)
        except Exception:
            pass
        # Try fenced code block ```json ... ```
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.S | re.I)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except Exception:
                pass
        # Try first JSON object anywhere
        brace_match = re.search(r"\{.*\}", stripped, re.S)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except Exception:
                return None
        return None

    async def _call_judge(
        self,
        semaphore: asyncio.Semaphore,
        judge_cfg: AnswerModelConfig,
        question_records: List[Dict[str, Any]],
        overall_rate: float,
        strong_rate: Optional[float],
        *,
        _retried: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not judge_cfg:
            return None
        question = question_records[0]
        stem = question.get("question_stem") or ""
        options = question.get("options") or []
        option_lines = "\n".join(f"- {opt}" for opt in options) if options else "None"

        samples = []
        for rec in question_records[:5]:
            samples.append(
                f"- {rec.get('answer_model_id') or rec.get('answer_model')}: "
                f"parsed={rec.get('parsed_answer')} | correct={rec.get('is_correct')}"
            )
        sample_block = "\n".join(samples) if samples else "None"

        stats_lines = [
            f"overall_correct_rate: {overall_rate:.3f}",
            f"strong_correct_rate: {strong_rate:.3f}" if strong_rate is not None else "strong_correct_rate: n/a",
        ]

        system_prompt = (
            "You are an expert dataset quality judge. "
            "Decide if the question is well-posed and whether the provided gold answer is correct."
        )
        user_prompt = (
            "Question quality check\n"
            f"Question type: {question.get('question_type')}\n"
            f"Stem: {stem}\n"
            f"Options:\n{option_lines}\n"
            f"Gold answer: {question.get('gold_answer')}\n\n"
            "Observed answer stats:\n"
            f"- {stats_lines[0]}\n"
            f"- {stats_lines[1]}\n\n"
            "Sample model answers:\n"
            f"{sample_block}\n\n"
            "Return a JSON object with keys:\n"
            '  "decision": one of ["clean", "not_well_posed", "gold_incorrect", "ambiguous"]\n'
            # '  "notes": brief rationale (one sentence)\n'
            "Output JSON only, in a single line (minified)."
        )

        max_tokens = judge_cfg.tokens_for(question.get("question_type", "open_ended"))
        # Ensure judge has enough budget to return JSON
        if max_tokens < 1024:
            max_tokens = 1024

        async with semaphore:
            try:
                response = await self.caller.call(
                    judge_cfg,
                    system_prompt,
                    user_prompt,
                    max_tokens,
                )
            except Exception as err:
                LOGGER.error("Judge call failed for %s: %s", question.get("question_id"), err)
                return None

        # Retry once with higher token budget if we got an empty response
        if (not response or not str(response).strip()) and not _retried:
            bumped = min(max(int(max_tokens * 2), max_tokens + 256), 2048)
            LOGGER.warning(
                "Judge empty response for %s; retrying with higher max_tokens %d -> %d",
                question.get("question_id"),
                max_tokens,
                bumped,
            )
            return await self._call_judge(
                semaphore,
                judge_cfg,
                question_records,
                overall_rate,
                strong_rate,
                _retried=True,
            )

        parsed = self._parse_json_blob(response or "")
        if parsed is None and not _retried:
            bumped = min(int(max_tokens * 2), 1536)
            LOGGER.warning(
                "Judge response unparsable for %s; retrying with higher max_tokens %d -> %d",
                question.get("question_id"),
                max_tokens,
                bumped,
            )
            return await self._call_judge(
                semaphore,
                judge_cfg,
                question_records,
                overall_rate,
                strong_rate,
                _retried=True,
            )
        if parsed is None:
            snippet = (response or "").strip()
            if not snippet:
                snippet = "<empty response>"
            if len(snippet) > 400:
                snippet = snippet[:400] + "...<truncated>"
            LOGGER.warning("Judge response unparsable for %s: %s", question.get("question_id"), snippet)
            return {
                "decision": "ambiguous",
                "notes": "judge_empty_or_unparsable_response",
                "raw_response": response,
            }
        decision = str(parsed.get("decision") or "").strip().lower()
        notes = parsed.get("notes")
        parsed["raw_response"] = response
        return {"decision": decision, "notes": notes, "raw_response": response}

    async def _call_llm_answer_judge(
        self,
        question: Dict[str, Any],
        model_answer: str,
        judge_cfg: Optional[AnswerModelConfig],
    ) -> Tuple[Optional[bool], Optional[Dict[str, Any]]]:
        """
        Judge a single answer (mainstream scorer) using the judge LLM.
        Returns (is_correct, judge_meta) where judge_meta carries label/score/raw_response.
        """
        if not judge_cfg:
            return None, None
        stem = question.get("question_stem") or ""
        gold = question.get("gold_answer") or question.get("answer") or ""
        prompt = (
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
            '- If uncertain, output \"broken_item\". Do NOT guess.\n\n'
            "Grading rubric:\n"
            "correct (0.9-1.0): All core ideas present; minor phrasing differences OK.\n"
            "partially_correct (0.4-0.8): Main idea present, but key details missing, vague, or slightly misstated. Higher score = only minor omissions.\n"
            "incorrect (0.0-0.3): Core concept missing or fundamentally wrong.\n"
            "broken_item: Question/gold answer is flawed, ambiguous, or factually incorrect.\n\n"
            "Output requirements:\n"
            "- JSON only. No markdown, no extra text.\n"
            '- \"missing\" and \"errors\" must be lists (empty if none).\n'
            "- Keep items in \"missing\"/\"errors\" concise (<=12 words).\n"
            "- Do NOT include reasoning outside JSON.\n\n"
            "{\n"
            '  \"label\": \"correct\" | \"partially_correct\" | \"incorrect\" | \"broken_item\",\n'
            '  \"score\": <float>,\n'
            '  \"missing\": [\"...\"],\n'
            '  \"errors\": [\"...\"]\n'
            "}"
        )

        judge_sem = asyncio.Semaphore(4)
        async with judge_sem:
            try:
                client = getattr(self.llm, f"{judge_cfg.provider}_client", None)
                model_name = judge_cfg.model
                # Route by provider
                if judge_cfg.provider == "gemini":
                    # Use the Gemini-native caller; it already throttles and uses genai client
                    async with self.caller._gemini_judge_semaphore:
                        now = time.perf_counter()
                        gap = now - self.caller._gemini_judge_last_ts
                        if gap < 1.1:
                            await asyncio.sleep(1.1 - gap)
                        self.caller._gemini_judge_last_ts = time.perf_counter()
                        response = await self.caller._call_gemini(
                            judge_cfg,
                            "You are an expert grader. Output JSON only.",
                            prompt,
                            judge_cfg.tokens_for("open_ended"),
                        )
                else:
                    response = await self.caller._call_openai_compatible(
                        client=client,
                        model_name=model_name,
                        system_prompt="You are an expert grader. Output JSON only.",
                        user_prompt=prompt,
                        max_new_tokens=judge_cfg.tokens_for("open_ended"),
                        temperature=judge_cfg.temperature,
                        top_p=judge_cfg.top_p,
                        extra_body=judge_cfg.extra_body,
                    )
            except Exception as err:
                LOGGER.warning("Judge LLM grading failed: %s", err)
                return None, {"raw_response": None, "error": str(err)}

        parsed = None
        try:
            match = re.search(r"\{.*\}", response, re.S)
            if match:
                parsed = json.loads(match.group())
            else:
                parsed = json.loads(response)
        except Exception:
            parsed = None

        if not parsed:
            return None, {"raw_response": response, "label": None, "score": None}

        label = str(parsed.get("label") or "").lower()
        score = parsed.get("score")
        missing = parsed.get("missing") if isinstance(parsed.get("missing"), list) else []
        errors = parsed.get("errors") if isinstance(parsed.get("errors"), list) else []

        judge_meta = {
            "label": label,
            "score": score,
            "missing": missing,
            "errors": errors,
            "raw_response": response,
        }

        # Map to binary correctness using soft score threshold
        if label == "broken_item":
            return None, judge_meta
        if score is not None:
            return bool(score >= JUDGE_CORRECT_THRESHOLD), judge_meta
        if label == "correct":
            return True, judge_meta
        if label == "partially_correct":
            return False, judge_meta
        if label == "incorrect":
            return False, judge_meta
        return None, judge_meta

    async def _run_dynamic_quality(self, dataset_cfg: Stage3DatasetConfig) -> None:
        output_dir = dataset_cfg.output_dir
        pattern = f"{dataset_cfg.dataset}_stage3_answers_*.jsonl"
        files = sorted(output_dir.glob(pattern))
        if not files:
            LOGGER.info("Dynamic quality pass: no outputs found for %s", dataset_cfg.dataset)
            return
        # NOTE: Dynamic quality is computed per question across ALL available answer models.
        # Do not restrict by CLI --designer/--answerer filters here; those are for Stage-3
        # answering runs. Dynamic quality should always reflect the full panel.

        LOGGER.info("Dynamic quality pass: loading %d files for %s", len(files), dataset_cfg.dataset)

        file_records: Dict[Path, List[Dict[str, Any]]] = {}
        all_records: List[Dict[str, Any]] = []
        for path in files:
            records: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        records.append(rec)
                        all_records.append(rec)
                    except json.JSONDecodeError as err:
                        LOGGER.error("Failed to decode line in %s: %s", path, err)
            file_records[path] = records

        if not all_records:
            LOGGER.info("Dynamic quality pass: no records for %s", dataset_cfg.dataset)
            return

        strong_set = {slugify(s) for s in self.strong_models}
        per_question: Dict[str, Dict[str, Any]] = {}
        for rec in all_records:
            qid = rec.get("question_id")
            if not qid:
                continue
            per_question.setdefault(qid, {"records": []})["records"].append(rec)

        judge_sem = asyncio.Semaphore(4)
        MIN_TOTAL_FOR_FLAGS = 3
        MIN_STRONG_FOR_FLAGS = 2
        judge_count = 0
        judge_flag_breakdown: Dict[str, int] = defaultdict(int)

        for qid, bundle in per_question.items():
            records = bundle.get("records") or []
            if not records:
                continue

            total = len(records)
            correct_total = sum(1 for r in records if r.get("is_correct") is True)
            overall_rate = correct_total / total if total else 0.0

            strong_records = [
                r
                for r in records
                if slugify(r.get("answer_model_id") or r.get("answer_model")) in strong_set
            ]
            strong_total = len(strong_records)
            strong_correct = sum(1 for r in strong_records if r.get("is_correct") is True)
            strong_rate: Optional[float] = (strong_correct / strong_total) if strong_total else None

            dynamic_flags: List[str] = []
            # Apply dynamic rules only when we have enough evidence.
            if total >= MIN_TOTAL_FOR_FLAGS:
                if strong_total >= MIN_STRONG_FOR_FLAGS and strong_rate is not None:
                    if strong_rate == 0.0:
                        dynamic_flags.append("all_strong_fail")
                    elif 0 < strong_rate < 0.4:
                        dynamic_flags.append("strong_disagreement")

                if overall_rate > 0.95:
                    dynamic_flags.append("too_easy")
                elif overall_rate < 0.05:
                    dynamic_flags.append("too_hard")

            judge_result: Optional[Dict[str, Any]] = None
            judge_flags: List[str] = []
            judge_used = False
            # Judge gating logic
            def _should_call_judge() -> bool:
                base_status = records[0].get("item_status") or "candidate"
                if base_status == "broken_static":
                    return False
                if total < 3 or strong_total < 2:
                    return False
                if "too_easy" in dynamic_flags:
                    return False
                suspicious = set(dynamic_flags) & {"all_strong_fail", "strong_disagreement", "too_hard"}
                design_type = (records[0].get("design_type") or "").lower()
                if "too_hard" in suspicious:
                    if design_type == "adversarial":
                        if "all_strong_fail" not in dynamic_flags and "strong_disagreement" not in dynamic_flags:
                            suspicious.discard("too_hard")
                return len(suspicious) > 0

            if self.judge_config and _should_call_judge():
                judge_used = True
                judge_count += 1
                judge_result = await self._call_judge(
                    judge_sem,
                    self.judge_config,
                    records,
                    overall_rate,
                    strong_rate,
                )
                if judge_result:
                    decision = judge_result.get("decision")
                    if decision == "not_well_posed":
                        judge_flags.append("judge_not_well_posed")
                    elif decision == "gold_incorrect":
                        judge_flags.append("judge_gold_incorrect")
                    elif decision == "ambiguous":
                        judge_flags.append("judge_ambiguous")

            base_flags = self._safe_list(records[0].get("item_quality_flags"))
            base_sources = self._safe_list(records[0].get("item_quality_source"))
            combined_flags = sorted(set(self._merge_list(base_flags, dynamic_flags + judge_flags)))

            sources = list(base_sources)
            if dynamic_flags and "dynamic_rules" not in sources:
                sources.append("dynamic_rules")
            if judge_used and "judge_llm" not in sources:
                sources.append("judge_llm")

            base_status = records[0].get("item_status") or "candidate"
            if base_status in {"broken_static", "skip_core"}:
                final_status = base_status
            elif "judge_not_well_posed" in combined_flags or "judge_gold_incorrect" in combined_flags:
                final_status = "broken_judge"
            elif "judge_ambiguous" in combined_flags:
                final_status = "ambiguous"
            else:
                # No judge used or judge cleared; mark clean unless upstream had another status.
                final_status = base_status if base_status != "candidate" else "clean"

            for rec in records:
                rec["total_answers"] = total
                rec["strong_total"] = strong_total
                rec["item_quality_flags_dynamic"] = list(dynamic_flags)
                rec["overall_correct_rate"] = overall_rate
                rec["strong_correct_rate"] = strong_rate
                rec["item_quality_flags"] = combined_flags
                rec["item_quality_source"] = sources
                rec["item_status"] = final_status
                if judge_used:
                    for flag in judge_flags:
                        judge_flag_breakdown[flag] += 1
                    rec["judge_model_id"] = self.judge_config.key
                    rec["judge_decision"] = (judge_result or {}).get("decision")
                    rec["judge_notes"] = (judge_result or {}).get("notes")
                    rec["judge_raw_response"] = (judge_result or {}).get("raw_response")

        # Rewrite files with updated records
        for path, records in file_records.items():
            with open(path, "w", encoding="utf-8") as sink:
                for rec in records:
                    sink.write(json.dumps(rec, ensure_ascii=False))
                    sink.write("\n")

        # Compute accuracy summaries on clean rows
        self._write_accuracy_summary(all_records, dataset_cfg.dataset, output_dir)
        self._write_research_metrics(all_records, dataset_cfg.dataset, output_dir)

        if judge_count:
            LOGGER.info(
                "Judge summary: %d questions sent to judge; flags breakdown: %s",
                judge_count,
                dict(judge_flag_breakdown),
            )
        else:
            LOGGER.info("Judge summary: 0 questions sent to judge.")

        LOGGER.info("Dynamic quality pass finished for %s.", dataset_cfg.dataset)

    # ----------------------------------------------------------------------- #
    # Accuracy summaries
    # ----------------------------------------------------------------------- #
    @staticmethod
    def _aggregate_matrix(records: List[Dict[str, Any]], group_fields: Sequence[str]) -> Dict[Tuple[Any, ...], Dict[str, int]]:
        agg: Dict[Tuple[Any, ...], Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        for rec in records:
            # Mean(is_correct) where is_correct ∈ {True, False}. Skip null/unscored rows.
            is_correct = rec.get("is_correct")
            if is_correct is None:
                continue
            key = tuple(rec.get(field) for field in group_fields)
            agg[key]["total"] += 1
            if is_correct is True:
                agg[key]["correct"] += 1
        return agg

    @staticmethod
    def _matrix_to_nested(agg: Dict[Tuple[Any, ...], Dict[str, int]], group_fields: Sequence[str]) -> Dict[str, Any]:
        nested: Dict[str, Any] = {}
        for key, stats in agg.items():
            acc = stats["correct"] / stats["total"] if stats["total"] else None
            cursor: Dict[str, Any] = nested
            for idx, _ in enumerate(group_fields):
                val = key[idx]
                if idx == len(group_fields) - 1:
                    cursor[val] = acc
                else:
                    cursor = cursor.setdefault(val, {})
        return nested

    @staticmethod
    def _matrix_to_nested_avg(agg: Dict[Tuple[Any, ...], Dict[str, Any]], group_fields: Sequence[str]) -> Dict[str, Any]:
        """
        Similar to _matrix_to_nested but expects each stats dict to carry an average value under
        'avg_soft'. Used for soft-score matrices where we don't have correct/total counts.
        """
        nested: Dict[str, Any] = {}
        for key, stats in agg.items():
            acc = stats.get("avg_soft")
            cursor: Dict[str, Any] = nested
            for idx, _ in enumerate(group_fields):
                val = key[idx]
                if idx == len(group_fields) - 1:
                    cursor[val] = acc
                else:
                    cursor = cursor.setdefault(val, {})
        return nested

    @staticmethod
    def _filter_core(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            r
            for r in records
            if (r.get("item_status") in {"clean", "suspect_static"})
            and r.get("scoring_method") != "skip_core"
        ]

    @staticmethod
    def _count_core_vs_skip(records: List[Dict[str, Any]], key_fn) -> Dict[Any, Dict[str, int]]:
        counts: Dict[Any, Dict[str, int]] = defaultdict(lambda: {"core": 0, "skip_core": 0})
        for rec in records:
            key = key_fn(rec)
            if key is None:
                continue
            is_skip = rec.get("scoring_method") == "skip_core" or rec.get("item_status") == "skip_core"
            if is_skip:
                counts[key]["skip_core"] += 1
            else:
                counts[key]["core"] += 1
        return counts

    @staticmethod
    def _collect_design_difficulty(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: {"skip_core": 0, "skip_core_items": []})
        )
        for rec in records:
            design = (rec.get("design_type") or "unknown").lower()
            difficulty = rec.get("declared_difficulty") or "unknown"
            qid = rec.get("question_id")
            designer = rec.get("designer_model") or rec.get("designer_id") or rec.get("designer")
            answer_model_id = rec.get("answer_model_id") or rec.get("answer_model")
            is_skip = rec.get("scoring_method") == "skip_core" or rec.get("item_status") == "skip_core"
            if is_skip:
                bucket = result[design][difficulty]
                bucket["skip_core"] += 1
                if qid:
                    bucket["skip_core_items"].append(
                        {
                            "question_id": qid,
                            "designer": designer,
                            "answer_model_id": answer_model_id,
                        }
                    )
            # Track totals for visibility
            bucket_total = result[design][difficulty]
            bucket_total["total"] = bucket_total.get("total", 0) + 1
        # Convert defaultdicts to normal dicts for JSON serialization
        return {d: dict(inner) for d, inner in result.items()}

    @staticmethod
    def _compute_hard_soft_metrics(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        metrics: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            model_id = rec.get("answer_model_id") or rec.get("answer_model")
            if not model_id:
                continue
            entry = metrics.setdefault(
                model_id,
                {
                    "hard_correct": 0,
                    "hard_total": 0,
                    "soft_sum": 0.0,
                    "soft_total": 0,
                },
            )
            is_correct = rec.get("is_correct")
            if is_correct is not None:
                entry["hard_total"] += 1
                if is_correct is True:
                    entry["hard_correct"] += 1
            soft_score = rec.get("soft_score")
            if soft_score is not None:
                entry["soft_total"] += 1
                entry["soft_sum"] += float(soft_score)

        for model_id, entry in metrics.items():
            entry["hard_acc"] = (entry["hard_correct"] / entry["hard_total"]) if entry["hard_total"] else None
            entry["avg_soft_score"] = (entry["soft_sum"] / entry["soft_total"]) if entry["soft_total"] else None
        return metrics

    @staticmethod
    def _aggregate_soft_matrix(records: List[Dict[str, Any]], group_fields: Sequence[str]) -> Dict[Tuple[Any, ...], Dict[str, float]]:
        agg: Dict[Tuple[Any, ...], Dict[str, float]] = defaultdict(lambda: {"soft_sum": 0.0, "soft_total": 0})
        for rec in records:
            soft = rec.get("soft_score")
            if soft is None:
                continue
            key = tuple(rec.get(field) for field in group_fields)
            agg[key]["soft_sum"] += float(soft)
            agg[key]["soft_total"] += 1
        return {k: {"avg_soft": (v["soft_sum"] / v["soft_total"]) if v["soft_total"] else None} for k, v in agg.items()}

    def _write_accuracy_summary(
        self,
        records: List[Dict[str, Any]],
        dataset_name: str,
        output_dir: Path,
    ) -> None:
        # Normalize languages for grouping
        normalized_records: List[Dict[str, Any]] = []
        for r in records:
            r_copy = dict(r)
            r_copy["language"] = self._normalize_language(r.get("language"))
            # Standardize model IDs for grouping
            ans_id = r_copy.get("answer_model_id") or slugify(r_copy.get("answer_model") or "")
            des_id = r_copy.get("designer_id") or slugify(r_copy.get("designer_model") or "")
            if ans_id:
                r_copy["answer_model"] = ans_id
                r_copy["answer_model_id"] = ans_id
            if des_id:
                r_copy["designer_model"] = des_id
                r_copy["designer_id"] = des_id
            normalized_records.append(r_copy)

        core_records = self._filter_core(normalized_records)
        if not core_records:
            LOGGER.info("Accuracy summary skipped for %s (no core records).", dataset_name)
            return

        def filt(predicate) -> List[Dict[str, Any]]:
            return [r for r in core_records if predicate(r)]

        summary: Dict[str, Any] = {}

        summary["global_matrix"] = self._matrix_to_nested(
            self._aggregate_matrix(core_records, ("answer_model", "designer_model")),
            ("answer_model", "designer_model"),
        )

        summary["standard_matrix"] = self._matrix_to_nested(
            self._aggregate_matrix(
                filt(lambda r: (r.get("design_type") or "").lower() == "standard"),
                ("answer_model", "designer_model"),
            ),
            ("answer_model", "designer_model"),
        )

        summary["adversarial_matrix"] = self._matrix_to_nested(
            self._aggregate_matrix(
                filt(lambda r: (r.get("design_type") or "").lower() == "adversarial"),
                ("answer_model", "designer_model"),
            ),
            ("answer_model", "designer_model"),
        )

        summary["by_difficulty"] = self._matrix_to_nested(
            self._aggregate_matrix(core_records, ("answer_model", "designer_model", "declared_difficulty")),
            ("answer_model", "designer_model", "declared_difficulty"),
        )

        summary["by_question_type"] = self._matrix_to_nested(
            self._aggregate_matrix(core_records, ("answer_model", "designer_model", "question_type")),
            ("answer_model", "designer_model", "question_type"),
        )

        summary["by_visual_condition"] = self._matrix_to_nested(
            self._aggregate_matrix(core_records, ("answer_model", "designer_model", "visual_condition")),
            ("answer_model", "designer_model", "visual_condition"),
        )

        summary["by_language"] = self._matrix_to_nested(
            self._aggregate_matrix(core_records, ("answer_model", "designer_model", "language")),
            ("answer_model", "designer_model", "language"),
        )

        # Soft-score global matrix (avg_soft per answerer/designer)
        soft_matrix = self._aggregate_soft_matrix(core_records, ("answer_model_id", "designer_id"))
        summary["global_matrix_soft"] = self._matrix_to_nested_avg(
            soft_matrix,
            ("answer_model_id", "designer_id"),
        )

        summary["by_design_and_difficulty"] = self._collect_design_difficulty(normalized_records)

        # Per-model hard/soft accuracy
        model_metrics = self._compute_hard_soft_metrics(core_records)
        summary["per_model"] = {
            mid: {
                "hard_acc": entry.get("hard_acc"),
                "avg_soft_score": entry.get("avg_soft_score"),
                "hard_total": entry.get("hard_total"),
                "soft_total": entry.get("soft_total"),
            }
            for mid, entry in model_metrics.items()
        }

        out_path = output_dir / f"{dataset_name}_stage3_accuracy_summary.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Accuracy summary written to %s", out_path)

    @staticmethod
    def _family_of(model_id: Optional[str]) -> str:
        base = slugify(model_id or "")
        return base.split("_", 1)[0] if base else ""

    @staticmethod
    def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
        n = min(len(xs), len(ys))
        if n < 2:
            return None
        mx = sum(xs[:n]) / n
        my = sum(ys[:n]) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        den_x = sum((xs[i] - mx) ** 2 for i in range(n))
        den_y = sum((ys[i] - my) ** 2 for i in range(n))
        if den_x <= 0 or den_y <= 0:
            return None
        return num / (den_x ** 0.5 * den_y ** 0.5)

    def _write_research_metrics(
        self,
        records: List[Dict[str, Any]],
        dataset_name: str,
        output_dir: Path,
    ) -> None:
        # Research metrics are computed on clean, scored rows only.
        clean_records = [
            r
            for r in records
            if (r.get("item_status") or "") == "clean" and r.get("is_correct") is not None
        ]
        if not clean_records:
            LOGGER.info("Research metrics skipped for %s (no clean records).", dataset_name)
            return

        # Model ability = overall accuracy per answer_model
        model_stats = self._aggregate_matrix(clean_records, ("answer_model",))
        model_ability: Dict[str, float] = {}
        for (model,), stats in model_stats.items():
            model_ability[model] = stats["correct"] / stats["total"] if stats["total"] else 0.0

        # Per-item difficulty and discrimination
        per_item: Dict[str, Dict[str, Any]] = {}
        per_designer_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        per_question_group = defaultdict(list)
        for rec in clean_records:
            qid = rec.get("question_id")
            if qid:
                per_question_group[qid].append(rec)

        for qid, recs in per_question_group.items():
            correctness = []
            abilities = []
            for r in recs:
                model = r.get("answer_model")
                if model is None:
                    continue
                abilities.append(model_ability.get(model, 0.0))
                correctness.append(1.0 if r.get("is_correct") is True else 0.0)
            if not correctness:
                continue
            p_correct = sum(correctness) / len(correctness)
            disc = self._pearson(abilities, correctness)
            base = recs[0]
            item_entry = {
                "question_id": qid,
                "designer_model": base.get("designer_model"),
                "p_correct": p_correct,
                "discrimination": disc,
                "num_models": len(correctness),
            }
            per_item[qid] = item_entry
            if base.get("designer_model"):
                per_designer_items[base["designer_model"]].append(item_entry)

        per_designer_summary: Dict[str, Any] = {}
        for designer, items in per_designer_items.items():
            if not items:
                continue
            per_designer_summary[designer] = {
                "mean_p_correct": sum(i["p_correct"] for i in items) / len(items),
                "mean_discrimination": (
                    sum(i["discrimination"] for i in items if i["discrimination"] is not None) /
                    max(1, sum(1 for i in items if i["discrimination"] is not None))
                ),
                "num_items": len(items),
            }

        # Family bias
        fam_bias_answer_model: Dict[str, Dict[str, Any]] = {}
        for model in model_ability:
            own, other = [], []
            model_fam = self._family_of(model)
            for r in clean_records:
                if r.get("answer_model") != model:
                    continue
                q_fam = self._family_of(r.get("designer_model"))
                target_list = own if q_fam == model_fam else other
                target_list.append(1.0 if r.get("is_correct") is True else 0.0)
            def _acc(vals: List[float]) -> Optional[float]:
                return sum(vals) / len(vals) if vals else None
            fam_bias_answer_model[model] = {
                "family": model_fam,
                "acc_on_own_family": _acc(own),
                "acc_on_other_family": _acc(other),
                "n_own": len(own),
                "n_other": len(other),
            }

        fam_matrix_agg = self._aggregate_matrix(
            clean_records,
            ("answer_model", "designer_model"),
        )
        family_matrix: Dict[str, Dict[str, Optional[float]]] = {}
        for (answer, designer), stats in fam_matrix_agg.items():
            ans_fam = self._family_of(answer)
            des_fam = self._family_of(designer)
            family_matrix.setdefault(ans_fam, {})
            acc = stats["correct"] / stats["total"] if stats["total"] else None
            family_matrix[ans_fam][des_fam] = acc

        # Visual behaviour
        visual_keywords = re.compile(r"\b(image|diagram|figure|picture|graph|chart|photo|draw|visual)\b", re.I)
        visual_records = []
        for r in clean_records:
            stem = r.get("question_stem") or ""
            visual_mention = bool(visual_keywords.search(stem))
            visual_records.append(
                {
                    "answer_model": r.get("answer_model"),
                    "designer_model": r.get("designer_model"),
                    "visual_condition": r.get("visual_condition"),
                    "visual_mention": visual_mention,
                    "is_correct": r.get("is_correct"),
                }
            )

        def _acc_over(items: List[Dict[str, Any]]) -> Optional[float]:
            if not items:
                return None
            return sum(1 for i in items if i.get("is_correct") is True) / len(items)

        visual_summary = {}
        for key in ["visual_mention=True", "visual_mention=False"]:
            want = key.endswith("True")
            subset = [i for i in visual_records if i["visual_mention"] is want]
            visual_summary[key] = _acc_over(subset)
        visual_by_condition = {}
        conds = {i["visual_condition"] for i in visual_records}
        for cond in conds:
            subset = [i for i in visual_records if i["visual_condition"] == cond]
            visual_by_condition[cond] = _acc_over(subset)

        # Visual priming interaction: (visual_condition × visual_mention)
        visual_by_condition_and_mention: Dict[str, Dict[str, Optional[float]]] = {}
        for cond in conds:
            visual_by_condition_and_mention.setdefault(cond, {})
            for mention in (True, False):
                subset = [
                    i
                    for i in visual_records
                    if i["visual_condition"] == cond and i["visual_mention"] is mention
                ]
                visual_by_condition_and_mention[cond][str(mention).lower()] = _acc_over(subset)

        # Per-designer: how often the stem mentions visuals + accuracy split
        visual_mention_rate_per_designer: Dict[str, Any] = {}
        for designer in {i["designer_model"] for i in visual_records if i.get("designer_model")}:
            subset = [i for i in visual_records if i.get("designer_model") == designer]
            if not subset:
                continue
            mention_rate = sum(1 for i in subset if i["visual_mention"] is True) / len(subset)
            visual_mention_rate_per_designer[designer] = {
                "visual_mention_rate": mention_rate,
                "acc_visual_mention": _acc_over([i for i in subset if i["visual_mention"] is True]),
                "acc_no_visual_mention": _acc_over([i for i in subset if i["visual_mention"] is False]),
            }

        # Degradation / repetition
        blocks: Dict[str, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        for r in clean_records:
            designer = r.get("designer_model") or ""
            gi = r.get("generation_index") or 0
            block = (int(gi) // 50) * 50 + 1
            blocks[designer][block].append(1 if r.get("is_correct") else 0)

        block_summary: Dict[str, Dict[int, float]] = {}
        for designer, block_map in blocks.items():
            block_summary[designer] = {
                block: (sum(vals) / len(vals) if vals else None)
                for block, vals in sorted(block_map.items())
            }

        repetition_clusters: Dict[str, List[Dict[str, Any]]] = {}
        # Build per-question view to avoid counting multiple answerers repeatedly
        designer_questions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        seen_qids: set[str] = set()
        for r in clean_records:
            qid = r.get("question_id")
            designer = r.get("designer_model")
            if not qid or not designer or qid in seen_qids:
                continue
            seen_qids.add(qid)
            designer_questions[designer].append(
                {"question_id": qid, "stem": normalize_text(r.get("question_stem") or "")}
            )

        def _token_set(s: str) -> set[str]:
            return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t}

        def _jaccard(a: set[str], b: set[str]) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        for designer, qs in designer_questions.items():
            if len(qs) < 2:
                continue
            # Exact duplicates (normalized)
            stems: Dict[str, List[str]] = defaultdict(list)
            for q in qs:
                stems[q["stem"]].append(q["question_id"])
            clusters: List[Dict[str, Any]] = []
            for stem, qids in stems.items():
                if len(qids) > 1 and stem:
                    clusters.append({"type": "exact", "stem": stem, "question_ids": qids, "count": len(qids)})

            # Near-duplicate clusters via token Jaccard similarity
            tokens = [(q["question_id"], _token_set(q["stem"])) for q in qs if q["stem"]]
            used: set[str] = set()
            for i in range(len(tokens)):
                qid_i, tok_i = tokens[i]
                if qid_i in used:
                    continue
                group = [qid_i]
                used.add(qid_i)
                for j in range(i + 1, len(tokens)):
                    qid_j, tok_j = tokens[j]
                    if qid_j in used:
                        continue
                    if _jaccard(tok_i, tok_j) >= 0.9:
                        group.append(qid_j)
                        used.add(qid_j)
                if len(group) > 1:
                    clusters.append({"type": "near", "threshold": 0.9, "question_ids": group, "count": len(group)})

            if clusters:
                repetition_clusters[designer] = clusters

        metrics = {
            "per_item": per_item,
            "per_designer": per_designer_summary,
            "family_bias_per_model": fam_bias_answer_model,
            "family_matrix": family_matrix,
            "visual_summary": visual_summary,
            "visual_by_condition": visual_by_condition,
            "visual_by_condition_and_mention": visual_by_condition_and_mention,
            "visual_mention_rate_per_designer": visual_mention_rate_per_designer,
            "generation_block_accuracy": block_summary,
            "repetition_clusters": repetition_clusters,
        }

        out_path = output_dir / f"{dataset_name}_stage3_research_metrics.json"
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        LOGGER.info("Research metrics written to %s", out_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 answerer panel runner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "config" / "stage3.yaml",
        help="Path to stage3 YAML config.",
    )
    parser.add_argument("--dataset", help="Override dataset name for outputs.")
    parser.add_argument(
        "--designer",
        action="append",
        help="Limit to specific designer key(s) or model names (repeatable).",
    )
    parser.add_argument(
        "--answerer",
        action="append",
        help="Limit to specific answerer key(s) or display names (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Max questions per designer (useful for smoke tests).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load configs and prompts but skip actual API calls.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip pairs whose output files already exist.",
    )
    parser.add_argument(
        "--skip-dynamic",
        action="store_true",
        help="Skip dynamic quality pass (answer-only).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = Stage3Config.from_yaml(args.config.resolve())
    runner = Stage3AnswererPanel(
        config,
        dataset_override=args.dataset,
        designer_filter=args.designer,
        answerer_filter=args.answerer,
        limit_per_designer=args.limit,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
        skip_dynamic=args.skip_dynamic,
    )
    await runner.run()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
