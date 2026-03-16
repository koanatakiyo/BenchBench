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
    answer_type = question.get("answer_type", "term")

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
class Stage3Config:
    dataset: str
    output_dir: Path
    designers: List[DesignerRunConfig]
    answer_models: List[AnswerModelConfig]
    judge_model: Optional[Dict[str, Any]] = None
    strong_models: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "Stage3Config":
        with open(path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        stage3_section = payload.get("stage3") or {}
        dataset = payload.get("dataset")
        if not dataset:
            raise ValueError("Config missing top-level 'dataset'.")

        output_dir = Path(
            stage3_section.get(
                "output_dir",
                Path(__file__).resolve().parents[4] / "outputs" / "stage3_answers" / dataset,
            )
        ).resolve()

        designers_raw = stage3_section.get("designers") or {}
        if not designers_raw:
            raise ValueError("No designers listed under stage3.designers.")

        designers: List[DesignerRunConfig] = []
        for key, cfg in designers_raw.items():
            provider_dir = Path(cfg["provider_dir"]).resolve()
            question_file = cfg.get("question_file")
            designers.append(
                DesignerRunConfig(
                    key=str(key),
                    designer_model=cfg.get("designer_model", key),
                    provider_dir=provider_dir,
                    question_file=Path(question_file).resolve()
                    if question_file
                    else None,
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
            dataset=dataset,
            output_dir=output_dir,
            designers=designers,
            answer_models=answerers,
            judge_model=judge_model or None,
            strong_models=[str(item) for item in strong_models_raw],
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
    record["visual_condition"] = record.get("visual_condition") or _infer_visual_condition(record)
    record["generation_index"] = record.get("generation_index", generation_index)
    record["language"] = record.get("language") or "en"
    record["declared_difficulty"] = record.get("declared_difficulty") or record.get("difficulty") or ""

    missing = [field for field in REQUIRED_STAGE2_FIELDS if field not in record or record.get(field) in (None, "")]
    if missing:
        LOGGER.warning("Question %s is missing required fields: %s", question_id, ", ".join(sorted(missing)))
    return record


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

            if texts:
                return "\n".join(texts)
            # Fallback: try chat.completions if Responses returned no text
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            chat_resp = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_new_tokens,
            )
            return chat_resp.choices[0].message.content

        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        return response.choices[0].message.content

    async def _call_gemini(
        self,
        cfg: AnswerModelConfig,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int,
    ) -> str:
        if not getattr(self.llm, "gemini_available", False):
            raise RuntimeError("Gemini client is not initialized.")

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
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                part = content.parts[0]
                text = getattr(part, "text", None)
                if text:
                    return text

        # Graceful fallback: return empty string instead of raising, so the call
        # is logged as an incorrect answer rather than stopping the run.
        logging.warning("Gemini response did not include text content; returning empty string.")
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
            system = "You are taking a computer science exam. Choose exactly one correct option."
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
            system = "You are taking a computer science exam. Some questions may have multiple correct options."
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
        system = "You are taking a short-answer computer science exam. Give concise, direct answers."
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
    def score(
        question: Dict[str, Any],
        parsed_answer: Any,
    ) -> Tuple[Optional[bool], Dict[str, Any]]:
        qtype = question.get("question_type", "open_ended")
        gold = question.get("gold_answer", question.get("answer"))
        scoring_method = (question.get("scoring_method") or "").lower()
        if not scoring_method:
            if qtype in {"mcq_single", "mcq_multi"}:
                scoring_method = "exact"
            else:
                scoring_method = "exact"

        details: Dict[str, Any] = {"method": scoring_method}
        if parsed_answer is None or gold is None:
            if scoring_method == "judge":
                details["requires_judge"] = True
                return None, details
            return False, details

        if qtype == "mcq_single":
            result = str(parsed_answer).upper() == str(gold).upper()
            return result, details

        if qtype == "mcq_multi":
            parsed_set = set(parsed_answer) if isinstance(parsed_answer, (list, tuple)) else {parsed_answer}
            gold_set = set(gold) if isinstance(gold, (list, tuple)) else {gold}
            return parsed_set == gold_set, details

        if scoring_method == "numeric":
            pred_value = extract_numeric(parsed_answer if isinstance(parsed_answer, str) else str(parsed_answer))
            gold_value = extract_numeric(gold if isinstance(gold, str) else str(gold))
            if pred_value is None or gold_value is None:
                return False, details
            tolerance = max(1e-6, abs(gold_value) * 1e-4)
            return abs(pred_value - gold_value) <= tolerance, details

        if scoring_method == "judge":
            details["requires_judge"] = True
            return None, details

        if qtype == "open_ended":
            is_correct, reason = score_open_ended(question, str(parsed_answer))
            details["reason"] = reason
            details["method"] = "open_ended_heuristic"
            return is_correct, details

        normalized_pred = normalize_text(parsed_answer if isinstance(parsed_answer, str) else str(parsed_answer))
        normalized_gold = normalize_text(gold if isinstance(gold, str) else str(gold))
        details["normalized_gold"] = normalized_gold
        details["normalized_pred"] = normalized_pred
        return normalized_pred == normalized_gold, details


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
    ):
        self.config = config
        self.dataset = dataset_override or config.dataset
        self.limit_per_designer = limit_per_designer
        self.dry_run = dry_run
        self.skip_existing = skip_existing
        self.judge_model = config.judge_model
        self.strong_models = config.strong_models

        designer_filter_set = {slugify(item) for item in designer_filter or []}
        answerer_filter_set = {slugify(item) for item in answerer_filter or []}

        self.designers = [
            d
            for d in config.designers
            if not designer_filter_set or slugify(d.key) in designer_filter_set or slugify(d.designer_model) in designer_filter_set
        ]
        self.answerers = [
            a
            for a in config.answer_models
            if a.enabled
            and (
                not answerer_filter_set
                or slugify(a.key) in answerer_filter_set
                or slugify(a.display_name) in answerer_filter_set
            )
        ]

        if not self.designers:
            raise ValueError("No designers selected after filtering.")
        if not self.answerers:
            raise ValueError("No answerer models selected after filtering.")

        providers_needed = self._collect_providers(self.answerers)
        self.llm = LLMManager(models_to_init=providers_needed)
        self.caller = AnswerModelCaller(self.llm)
        self.prompt_builder = PromptBuilder()
        self.parser = AnswerParser()

        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        for answerer in self.answerers:
            limit = answerer.max_concurrent_requests or answerer.batch_size
            self._semaphores[answerer.key] = asyncio.Semaphore(max(1, limit))

    @staticmethod
    def _collect_providers(answerers: List[AnswerModelConfig]) -> List[str]:
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
        return sorted(providers)

    async def run(self) -> None:
        total_pairs = len(self.designers) * len(self.answerers)
        LOGGER.info("Starting Stage-3 answering: %d designers x %d answerers = %d pairs.", len(self.designers), len(self.answerers), total_pairs)
        if self.judge_model:
            LOGGER.info("Judge model configured for rare checks: %s", self.judge_model.get("model") or self.judge_model)
        if self.strong_models:
            LOGGER.info("Strong model set for quality checks: %s", ", ".join(self.strong_models))

        for designer in self.designers:
            questions = self._load_questions(designer)
            if self.limit_per_designer:
                questions = questions[: self.limit_per_designer]
            if not questions:
                LOGGER.warning("Designer %s produced no questions; skipping.", designer.designer_model)
                continue

            for answerer in self.answerers:
                await self._run_pair(designer, answerer, questions)

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
        LOGGER.info("Loaded %d questions for %s", len(questions), designer.designer_model)
        return questions

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
        designer: DesignerRunConfig,
        answerer: AnswerModelConfig,
        questions: List[Dict[str, Any]],
    ) -> None:
        designer_slug = slugify(designer.designer_model)
        answerer_slug = slugify(answerer.display_name)
        filename = f"{self.dataset}_stage3_answers_{designer_slug}_vs_{answerer_slug}.jsonl"
        output_path = self.output_dir / filename

        if self.skip_existing and output_path.exists():
            LOGGER.info("Skipping %s vs %s (output exists).", designer.designer_model, answerer.display_name)
            return

        LOGGER.info(
            "Answering %d questions: designer=%s, answerer=%s -> %s",
            len(questions),
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
                records = await self._answer_batch(batch, designer, answerer, call_batch_id, batch_idx)
                for record in records:
                    sink.write(json.dumps(record, ensure_ascii=False))
                    sink.write("\n")

    async def _answer_batch(
        self,
        batch: List[Dict[str, Any]],
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
        is_correct, score_meta = AnswerScorer.score(question, parsed_answer)

        record: Dict[str, Any] = {
            "dataset": self.dataset,
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
            "gold_answer": question.get("gold_answer", question.get("answer")),
            "model_output_raw": raw_response,
            "parsed_answer": parsed_answer,
            "scoring_method": score_meta.get("method"),
            "score_details": score_meta,
            "is_correct": is_correct,
            "answer_time_ms": elapsed_ms,
            "call_batch_id": call_batch_id,
            "call_batch_index": batch_idx,
            "call_batch_size": batch_size,
            "call_index_in_batch": call_index,
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
        return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3 answerer panel runner (csbench).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "config" / "stage3_csbench.yaml",
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
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


# async def answer_all_models(
#     config_path: Optional[Path] = None,
#     *,
#     dataset: Optional[str] = None,
#     dry_run: bool = False,
#     skip_existing: bool = False,
# ) -> None:
#     """
#     Convenience helper to answer every designer drop with every configured model.

#     Args:
#         config_path: Optional override for the stage3 YAML file. Defaults to
#             benchbench/config/stage3_csbench.yaml.
#         dataset: Optional dataset name override for output metadata.
#         dry_run: When True, build prompts and plan batches without calling any APIs.
#         skip_existing: When True, pairs whose JSONL output already exists are skipped.
#     """
#     default_config = Path(__file__).resolve().parents[3] / "config" / "stage3_csbench.yaml"
#     config = Stage3Config.from_yaml((config_path or default_config).resolve())
#     runner = Stage3AnswererPanel(
#         config,
#         dataset_override=dataset,
#         dry_run=dry_run,
#         skip_existing=skip_existing,
#     )
#     await runner.run()


# def answer_all_models_sync(
#     config_path: Optional[Path] = None,
#     *,
#     dataset: Optional[str] = None,
#     dry_run: bool = False,
#     skip_existing: bool = False,
# ) -> None:
#     """Sync wrapper around :func:`answer_all_models`."""
#     asyncio.run(
#         answer_all_models(
#             config_path=config_path,
#             dataset=dataset,
#             dry_run=dry_run,
#             skip_existing=skip_existing,
#         )
#     )


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
    )
    await runner.run()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()

