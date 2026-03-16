#!/usr/bin/env python3
"""
Stage 2 Orchestrator - Run the complete question generation pipeline
"""

import asyncio
import argparse
import sys
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import logging

from question_generator import QuestionGenerator
from validate_questions import QuestionValidator
from prompt_builder import (
    load_domain_card,
    build_batch_user_prompt,
    get_dataset_display_name,
    infer_language,
    infer_modality,
    extract_super_parent_context,
    get_additional_instructions_for_dataset,
    BatchPromptSpec,
    SuperParentContext
)
from coverage_config import get_coverage_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Available datasets
AVAILABLE_DATASETS = [
    "csbench_en",
    "csbench_cn",
    "csbench_fr",
    "csbench_de",
    "tombench_en",
    "tombench_cn",
    "medxpertqa_text",
    "medxpertqa_mm",
    "wemath",
    "wemath_stage2_textonly",
    "wemath_stage2_visualprimed",
    "medxpertqa_mm_stage2_textonly",
    "medxpertqa_mm_stage2_visualprimed",
]


@dataclass
class BatchPlan:
    """Represents a single LLM call plan."""
    spec: BatchPromptSpec
    batch_label: str
    is_top_up: bool = False
    notes: str = ""


@dataclass(frozen=True)
class GenerationProfile:
    """Tunable defaults for batch sizing/top-ups."""
    name: str
    questions_per_call: int
    min_batch_questions: int
    max_batch_questions: int
    max_topup_rounds: int
    max_topup_batch_questions: Optional[int] = None


API_GENERATION_PROFILE = GenerationProfile(
    name="api",
    questions_per_call=40,
    min_batch_questions=10,
    max_batch_questions=40,
    max_topup_rounds=3,
    max_topup_batch_questions=None,
)

LOCAL_GENERATION_PROFILE = GenerationProfile(
    name="local",
    questions_per_call=10,
    min_batch_questions=5,
    max_batch_questions=10,
    max_topup_rounds=2,
    max_topup_batch_questions=5,
)

GENERATION_PROFILES: Dict[str, GenerationProfile] = {
    API_GENERATION_PROFILE.name: API_GENERATION_PROFILE,
    LOCAL_GENERATION_PROFILE.name: LOCAL_GENERATION_PROFILE,
}

LOCAL_MODEL_DEFAULT_PROFILE = {"ollama", "huggingface", "vllm"}


def slugify(value: str) -> str:
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in value)
    return "_".join(filter(None, cleaned.split("_")))


class Stage2Orchestrator:
    """Orchestrate the complete Stage 2 pipeline"""

    def __init__(
        self,
        datasets: List[str],
        model: str = "openai",
        designer_model: str = "gpt-5-mini",
        temperature: float = 0.8,
        output_base: Optional[Path] = None,
        validate: bool = True,
        ollama_model_override: Optional[str] = None,
        ollama_base_url_override: Optional[str] = None,
        huggingface_options: Optional[dict] = None,
        questions_per_call: int = 300,
        min_batch_questions: int = 20,
        max_batch_questions: int = 40,
        max_topup_rounds: int = 3,
        max_topup_batch_questions: Optional[int] = None,
        llm_model_override: Optional[str] = None,
        profile_name: str = API_GENERATION_PROFILE.name,
        vllm_base_url_override: Optional[str] = None,
        vllm_max_tokens_override: Optional[int] = None,
        batch_mode: str = "none",
        batch_prompts_per_file: int = 20,
        batch_max_tokens: int = 8000,
    ):
        """
        Initialize orchestrator.

        Args:
            datasets: List of dataset names to process
            model: LLM to use for generation
            designer_model: Designer model identifier
            temperature: Sampling temperature
            output_base: Base output directory
            validate: Whether to validate after generation
        """
        self.datasets = datasets
        self.model = model
        self.designer_model = designer_model
        self.temperature = temperature
        self.validate = validate
        self.ollama_model_override = ollama_model_override
        self.ollama_base_url_override = ollama_base_url_override
        self.vllm_base_url_override = vllm_base_url_override
        self.vllm_max_tokens_override = (
            int(vllm_max_tokens_override) if vllm_max_tokens_override else None
        )
        self.huggingface_options = huggingface_options or {}
        self.profile_name = profile_name
        self.min_batch_questions = max(1, min_batch_questions)
        self.max_batch_questions = max(self.min_batch_questions, max_batch_questions)
        self.questions_per_call = max(1, questions_per_call)
        self.questions_per_call = max(self.min_batch_questions, self.questions_per_call)
        self.max_topup_rounds = max(0, max_topup_rounds)
        if max_topup_batch_questions is not None:
            self.max_topup_batch_questions = max(1, max_topup_batch_questions)
        else:
            self.max_topup_batch_questions = None
        self.llm_model_override = llm_model_override
        self.batch_counter_per_dataset: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.provider_name = self.model
        self.model_identifier = self._resolve_model_identifier()
        self.safe_provider_name = self._safe_name(self.provider_name)
        self.safe_model_identifier = self._safe_name(self.model_identifier)
        self.safe_designer_name = self._safe_name(self.designer_model)
        self.inter_batch_delay_seconds = 60 if self.model == "anthropic" else 0
        self.batch_mode = batch_mode
        self.batch_prompts_per_file = max(1, batch_prompts_per_file)
        self.batch_max_tokens = max(1, batch_max_tokens)
        self.dataset_extra_instructions: Dict[str, List[str]] = defaultdict(list)

        # Set output base
        if output_base is None:
            project_root = Path(__file__).parents[3]
            output_base = project_root / "outputs" / "stage2_questions"
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Results tracking
        self.results = {}

    @staticmethod
    def _safe_name(value: str) -> str:
        """Sanitize string for filenames."""
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value)

    def _resolve_model_identifier(self) -> str:
        """Determine human-readable model identifier for filenames/metadata."""
        if self.model == "huggingface":
            return self.huggingface_options.get("model_id") or "huggingface"
        if self.model == "ollama" and self.ollama_model_override:
            return self.ollama_model_override
        return self.model

    def _plan_initial_batches(
        self,
        dataset: str,
        dataset_display: str,
        language: str,
        modality: str,
        coverage_config,
        super_context: Dict[str, SuperParentContext],
        extra_instructions: Optional[List[str]] = None
    ) -> List[BatchPlan]:
        batches: List[BatchPlan] = []
        designer_slug = slugify(self.designer_model)
        dataset_slug = slugify(dataset)
        extra_instructions = extra_instructions or []

        dataset_counters = self.batch_counter_per_dataset[dataset]

        for quota in coverage_config.quotas:
            remaining_std = quota.standard_questions
            remaining_adv = quota.adversarial_questions
            batch_idx = 1
            while remaining_std + remaining_adv > 0:
                total_remaining = remaining_std + remaining_adv
                target_size = min(self.questions_per_call, total_remaining)
                if target_size > self.max_batch_questions:
                    target_size = self.max_batch_questions
                if target_size < self.min_batch_questions and total_remaining > self.min_batch_questions:
                    target_size = self.min_batch_questions
                if target_size <= 0:
                    break

                if total_remaining <= self.min_batch_questions:
                    target_size = total_remaining

                std_share = min(
                    remaining_std,
                    max(0, round(target_size * (remaining_std / total_remaining))) if total_remaining else 0
                )
                adv_share = target_size - std_share

                # Adjust if we've over/under allocated
                if adv_share > remaining_adv:
                    adv_share = remaining_adv
                    std_share = target_size - adv_share
                if std_share > remaining_std:
                    std_share = remaining_std
                    adv_share = target_size - std_share

                # If one bucket is empty but there are still counts remaining in the other, fill the gap
                if std_share == 0 and remaining_std > 0:
                    std_share = min(remaining_std, target_size - adv_share)
                    adv_share = target_size - std_share
                if adv_share == 0 and remaining_adv > 0:
                    adv_share = min(remaining_adv, target_size - std_share)
                    std_share = target_size - adv_share

                spec = BatchPromptSpec(
                    dataset_name=dataset,
                    dataset_display_name=dataset_display,
                    designer_model=self.designer_model,
                    super_parents=[quota.super_parent],
                    total_questions=target_size,
                    standard_questions=std_share,
                    adversarial_questions=adv_share,
                    difficulty_targets=self._allocate_targets(
                        coverage_config.difficulty_distribution, target_size, order=["L1", "L2", "L3", "L4", "L5"]
                    ),
                    format_targets=self._allocate_targets(
                        coverage_config.format_distribution, target_size
                    ),
                    id_prefix=f"{designer_slug}_{dataset_slug}_{slugify(quota.super_parent)}_b{batch_idx:02d}",
                    id_start_index=1,
                    language=language,
                    modality=modality,
                    super_parent_context={
                        quota.super_parent: super_context.get(
                            quota.super_parent,
                            SuperParentContext(super_parent=quota.super_parent)
                        )
                    },
                    additional_instructions=list(extra_instructions)
                )

                batches.append(
                    BatchPlan(
                        spec=spec,
                        batch_label=f"{slugify(quota.super_parent)}_b{batch_idx:02d}"
                    )
                )

                remaining_std -= std_share
                remaining_adv -= adv_share
                batch_idx += 1

            dataset_counters[quota.super_parent] = max(dataset_counters[quota.super_parent], batch_idx)

        return batches

    @staticmethod
    def _allocate_targets(distribution: Dict[str, float], total: int, order: Optional[List[str]] = None) -> Dict[str, int]:
        if total <= 0:
            return {k: 0 for k in (order or distribution.keys())}

        keys = order or list(distribution.keys())
        raw = []
        for key in keys:
            pct = distribution.get(key, 0.0)
            raw.append((key, pct * total))

        allocated: Dict[str, int] = {}
        cumulative = 0
        for key, value in raw:
            count = int(value)
            allocated[key] = count
            cumulative += count

        remainder = total - cumulative
        if remainder > 0:
            for key, value in sorted(raw, key=lambda x: x[1] - int(x[1]), reverse=True):
                if remainder <= 0:
                    break
                allocated[key] += 1
                remainder -= 1

        return allocated

    async def _execute_batch_plans(
        self,
        generator: QuestionGenerator,
        batch_plans: List[BatchPlan],
        dataset: str,
        is_top_up: bool = False
    ) -> List[Dict[str, Any]]:
        batch_root = generator.output_dir / "batches"
        batch_root.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for idx, plan in enumerate(batch_plans):
            if idx > 0 and self.inter_batch_delay_seconds > 0:
                logger.info(
                    f"[{dataset}] Waiting {self.inter_batch_delay_seconds}s before next batch to honor {self.model} rate limits."
                )
                await asyncio.sleep(self.inter_batch_delay_seconds)

            user_prompt = build_batch_user_prompt(plan.spec)
            prompt_suffix = "_topup" if is_top_up else ""
            prompt_path = batch_root / f"{plan.batch_label}{prompt_suffix}_prompt.txt"
            prompt_path.write_text(user_prompt, encoding="utf-8")

            raw_filename = f"{plan.batch_label}{prompt_suffix}_raw.txt"
            raw_output_path = batch_root / raw_filename

            questions = await generator.generate_questions_from_prompt(
                user_prompt=user_prompt,
                expected_questions=plan.spec.total_questions,
                model=self.model,
                temperature=self.temperature,
                raw_output_path=raw_output_path
            )

            batch_filename = f"{plan.batch_label}{prompt_suffix}_questions.jsonl"
            batch_output_path = generator.save_questions(
                questions,
                filename=f"batches/{batch_filename}"
            )

            results.append({
                "plan": plan,
                "prompt_path": prompt_path,
                "output_path": batch_output_path,
                "raw_path": raw_output_path,
                "questions": questions,
                "is_top_up": is_top_up
            })

        return results

    async def _execute_batch_plans_batch_api(
        self,
        generator: QuestionGenerator,
        batch_plans: List[BatchPlan],
        dataset: str,
        is_top_up: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Batch API path (OpenAI/Anthropic). Builds a JSONL of requests, submits a batch,
        polls for completion, and parses outputs.
        """
        if self.batch_mode not in {"openai", "anthropic"}:
            raise RuntimeError("batch_mode must be openai or anthropic for batch API execution")

        batch_root = generator.output_dir / "batches"
        batch_root.mkdir(parents=True, exist_ok=True)

        prompt_suffix = "_topup" if is_top_up else ""
        requests: List[Dict[str, Any]] = []
        plan_lookup: Dict[str, BatchPlan] = {}
        for plan in batch_plans:
            user_prompt = build_batch_user_prompt(plan.spec)
            prompt_path = batch_root / f"{plan.batch_label}{prompt_suffix}_prompt.txt"
            prompt_path.write_text(user_prompt, encoding="utf-8")

            custom_id = f"{plan.batch_label}{prompt_suffix}"
            plan_lookup[custom_id] = plan

            if self.batch_mode == "openai":
                requests.append(
                    {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": self._build_openai_batch_body(
                            generator=generator,
                            user_prompt=user_prompt,
                        ),
                    }
                )
            else:  # anthropic
                requests.append(
                    {
                        "custom_id": custom_id,
                        "params": {
                            "model": generator.llm.config.get("anthropic", {}).get("model")
                            or self.llm_model_override
                            or self.model_identifier,
                            "max_tokens": self.batch_max_tokens,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": generator.system_prompt or ""},
                                        {"type": "text", "text": user_prompt},
                                    ],
                                }
                            ],
                        },
                    }
                )

        results: List[Dict[str, Any]] = []

        if not requests:
            return results

        # Split into chunks for safety
        chunks = [
            requests[i : i + self.batch_prompts_per_file]
            for i in range(0, len(requests), self.batch_prompts_per_file)
        ]

        for chunk_idx, chunk_requests in enumerate(chunks):
            logger.info(
                "[%s] Submitting batch chunk %d/%d via %s (%d prompts)",
                dataset,
                chunk_idx + 1,
                len(chunks),
                self.batch_mode,
                len(chunk_requests),
            )
            if self.batch_mode == "openai":
                batch_id = await generator.llm.openai_create_batch(chunk_requests)
                status = await self._poll_openai_batch(batch_id)
                output_file_id = status.get("output_file_id")
                if not output_file_id:
                    raise RuntimeError(f"OpenAI batch {batch_id} completed without output_file_id")
                raw_output = await generator.llm.openai_download_batch_output(output_file_id)
                chunk_results = self._parse_openai_batch_output(raw_output, generator)
            else:
                batch_id = await generator.llm.anthropic_create_batch(chunk_requests)
                status = await self._poll_anthropic_batch(batch_id)
                results_list = await generator.llm.anthropic_download_batch_output(batch_id)
                chunk_results = self._parse_anthropic_batch_output(results_list, generator)

            # Save outputs per request
            for item in chunk_results:
                custom_id = item["custom_id"]
                plan = plan_lookup.get(custom_id)
                if not plan:
                    continue
                questions = item["questions"]
                batch_filename = f"{plan.batch_label}{prompt_suffix}_questions.jsonl"
                batch_output_path = generator.save_questions(
                    questions,
                    filename=f"batches/{batch_filename}"
                )
                results.append(
                    {
                        "plan": plan,
                        "prompt_path": batch_root / f"{plan.batch_label}{prompt_suffix}_prompt.txt",
                        "output_path": batch_output_path,
                        "raw_path": None,
                        "questions": questions,
                        "is_top_up": is_top_up,
                        "batch_id": item.get("batch_id"),
                    }
                )

        return results

    def _build_openai_batch_body(self, generator: QuestionGenerator, user_prompt: str) -> Dict[str, Any]:
        """Build OpenAI batch request body with GPT-5 parameter compatibility."""
        model_name = (
            generator.llm.config.get("openai", {}).get("model", {}).get("text")
            or generator.llm.config.get("openai", {}).get("model")
            or self.llm_model_override
            or self.model_identifier
        )
        is_gpt5 = isinstance(model_name, str) and "gpt-5" in model_name.lower()

        body: Dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": generator.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if is_gpt5:
            # GPT-5 batches: use lower max_completion_tokens to avoid truncation to empty
            body["max_completion_tokens"] = min(self.batch_max_tokens, 2000)
            # GPT-5 batches do not allow temperature overrides; use model default.
        else:
            body["max_tokens"] = self.batch_max_tokens
            body["temperature"] = self.temperature

        return body

    async def _poll_openai_batch(self, batch_id: str, timeout_seconds: int = 7200) -> dict:
        start = asyncio.get_event_loop().time()
        while True:
            status = await self._safe_openai_get_batch(batch_id)
            state = status.get("status")
            if state in {"completed", "failed", "expired", "cancelled"}:
                return status
            if asyncio.get_event_loop().time() - start > timeout_seconds:
                raise TimeoutError(f"OpenAI batch {batch_id} did not complete in time")
            await asyncio.sleep(10)

    async def _safe_openai_get_batch(self, batch_id: str) -> dict:
        try:
            return await self.generator_llm().openai_get_batch(batch_id)
        except Exception as e:
            logger.warning("Failed to fetch OpenAI batch status (%s): %s", batch_id, e)
            return {"status": "unknown"}

    async def _poll_anthropic_batch(self, batch_id: str, timeout_seconds: int = 7200) -> dict:
        start = asyncio.get_event_loop().time()
        while True:
            status = await self.generator_llm().anthropic_get_batch(batch_id)
            state = status.get("status")
            if state in {"completed", "failed", "expired", "cancelled"}:
                return status
            if asyncio.get_event_loop().time() - start > timeout_seconds:
                raise TimeoutError(f"Anthropic batch {batch_id} did not complete in time")
            await asyncio.sleep(10)

    def _parse_openai_batch_output(self, raw_output: str, generator: QuestionGenerator) -> List[Dict[str, Any]]:
        """
        OpenAI batch output file: each line is JSON with custom_id and response.
        """
        results = []
        for line in raw_output.splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            custom_id = obj.get("custom_id")
            response_obj = obj.get("response", {})
            message = response_obj.get("choices", [{}])[0].get("message", {})
            content = message.get("content")
            if isinstance(content, list):
                text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                text = "\n".join(text_parts)
            else:
                text = content
            if not text:
                continue
            questions = generator._parse_response(text)
            results.append({"custom_id": custom_id, "questions": questions, "batch_id": obj.get("id")})
        return results

    def _parse_anthropic_batch_output(self, results_list: List[dict], generator: QuestionGenerator) -> List[Dict[str, Any]]:
        parsed = []
        for item in results_list:
            custom_id = item.get("custom_id")
            result = item.get("result") or {}
            content_blocks = result.get("content") or []
            text_parts = [blk.get("text", "") for blk in content_blocks if blk.get("type") == "text"]
            text = "\n".join(text_parts).strip()
            if not text:
                continue
            questions = generator._parse_response(text)
            parsed.append({"custom_id": custom_id, "questions": questions, "batch_id": result.get("id")})
        return parsed

    def generator_llm(self):
        """Helper to reference the LLM manager from generator in batch polling."""
        # This indirection avoids passing generator everywhere
        # Will be set on the orchestrator when needed
        return self._llm_ref

    def _calculate_deficits(self, questions: List[Dict[str, Any]], coverage_config) -> Dict[str, Any]:
        total_target = coverage_config.total_standard + coverage_config.total_adversarial
        valid_questions: List[Dict[str, Any]] = []
        for idx, q in enumerate(questions):
            if isinstance(q, dict):
                valid_questions.append(q)
            else:
                logger.warning(
                    "[%s] Skipping malformed question #%d of type %s during deficit calculation",
                    dataset,
                    idx,
                    type(q).__name__
                )

        questions = valid_questions
        actual_total = len(questions)

        super_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"standard": 0, "adversarial": 0})
        difficulty_counts: Dict[str, int] = defaultdict(int)
        format_counts: Dict[str, int] = defaultdict(int)

        for q in questions:
            super_parent = q.get("super_parent") or q.get("superParent") or "unknown"
            design_type = (q.get("design_type") or "").lower()
            if design_type == "adversarial":
                super_stats[super_parent]["adversarial"] += 1
            else:
                super_stats[super_parent]["standard"] += 1

            difficulty = q.get("declared_difficulty", "L3")
            difficulty_counts[difficulty] += 1

            q_type = q.get("question_type", "open_ended")
            format_counts[q_type] += 1

        super_deficits: Dict[str, Dict[str, int]] = {}
        needs_topup = False
        for quota in coverage_config.quotas:
            stats = super_stats.get(quota.super_parent, {"standard": 0, "adversarial": 0})
            std_def = max(0, quota.standard_questions - stats["standard"])
            adv_def = max(0, quota.adversarial_questions - stats["adversarial"])
            if std_def > 0 or adv_def > 0:
                needs_topup = True
            super_deficits[quota.super_parent] = {"standard": std_def, "adversarial": adv_def}

        difficulty_targets = self._allocate_targets(
            coverage_config.difficulty_distribution,
            total_target,
            order=["L1", "L2", "L3", "L4", "L5"]
        )
        difficulty_deficits = {
            level: max(0, difficulty_targets.get(level, 0) - difficulty_counts.get(level, 0))
            for level in difficulty_targets
        }
        if any(value > 0 for value in difficulty_deficits.values()):
            needs_topup = True

        format_targets = self._allocate_targets(coverage_config.format_distribution, total_target)
        format_deficits = {
            fmt: max(0, format_targets.get(fmt, 0) - format_counts.get(fmt, 0))
            for fmt in format_targets
        }
        if any(value > 0 for value in format_deficits.values()):
            needs_topup = True

        if actual_total < total_target:
            needs_topup = True

        return {
            "needs_topup": needs_topup,
            "super_parents": super_deficits,
            "difficulty": difficulty_deficits,
            "format": format_deficits,
            "total_target": total_target,
            "current_total": actual_total,
            "remaining_total": max(0, total_target - actual_total)
        }

    def _allocate_from_deficits(
        self,
        deficits_map: Dict[str, int],
        total: int,
        fallback_distribution: Dict[str, float],
        order: Optional[List[str]] = None
    ) -> Dict[str, int]:
        keys = order or list(fallback_distribution.keys())
        allocation = {key: 0 for key in keys}
        remaining = total

        for key in keys:
            need = deficits_map.get(key, 0)
            take = min(need, remaining)
            allocation[key] = take
            deficits_map[key] = max(0, need - take)
            remaining -= take
            if remaining <= 0:
                break

        if remaining > 0:
            fallback = self._allocate_targets(fallback_distribution, remaining, order=keys)
            for key, value in fallback.items():
                allocation[key] += value

        return allocation

    def _plan_topup_batches(
        self,
        deficits: Dict[str, Any],
        dataset: str,
        dataset_display: str,
        language: str,
        modality: str,
        super_context: Dict[str, SuperParentContext],
        coverage_config,
        extra_instructions: Optional[List[str]] = None
    ) -> List[BatchPlan]:
        plans: List[BatchPlan] = []
        remaining_total = deficits["remaining_total"]
        designer_slug = slugify(self.designer_model)
        dataset_slug = slugify(dataset)
        dataset_counters = self.batch_counter_per_dataset[dataset]
        extra_instructions = extra_instructions or []

        for super_parent, values in deficits["super_parents"].items():
            needed = values["standard"] + values["adversarial"]
            if needed <= 0 and remaining_total <= 0:
                continue

            if needed > 0:
                target_total = min(needed, self.questions_per_call)
            else:
                fallback_total = min(self.min_batch_questions, remaining_total) if remaining_total > 0 else self.min_batch_questions
                target_total = min(fallback_total, self.questions_per_call)

            if remaining_total > 0:
                target_total = min(target_total, remaining_total)

            target_total = min(target_total, self.max_batch_questions)
            if self.max_topup_batch_questions is not None:
                target_total = min(target_total, self.max_topup_batch_questions)
            target_total = max(1, target_total)

            std_share = min(values["standard"], target_total)
            adv_share = min(values["adversarial"], target_total - std_share)

            if std_share + adv_share < target_total:
                std_share = target_total - adv_share

            difficulty_targets = self._allocate_from_deficits(
                deficits["difficulty"],
                target_total,
                coverage_config.difficulty_distribution,
                order=["L1", "L2", "L3", "L4", "L5"]
            )
            format_targets = self._allocate_from_deficits(
                deficits["format"],
                target_total,
                coverage_config.format_distribution
            )

            counter = dataset_counters[super_parent]
            dataset_counters[super_parent] = counter + 1

            spec = BatchPromptSpec(
                dataset_name=dataset,
                dataset_display_name=dataset_display,
                designer_model=self.designer_model,
                super_parents=[super_parent],
                total_questions=target_total,
                standard_questions=std_share,
                adversarial_questions=adv_share,
                difficulty_targets=difficulty_targets,
                format_targets=format_targets,
                id_prefix=f"{designer_slug}_{dataset_slug}_{slugify(super_parent)}_topup{counter:02d}",
                id_start_index=1,
                language=language,
                modality=modality,
                super_parent_context={
                    super_parent: super_context.get(
                        super_parent,
                        SuperParentContext(super_parent=super_parent)
                    )
                },
                additional_instructions=[*extra_instructions, "Top-up batch to resolve outstanding coverage/difficulty gaps."]
            )

            plans.append(
                BatchPlan(
                    spec=spec,
                    batch_label=f"{slugify(super_parent)}_topup_{counter:02d}",
                    is_top_up=True
                )
            )

            remaining_total = max(0, remaining_total - target_total)

        return plans

    def _normalize_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        dataset_name = self.datasets[0] if self.datasets else ""

        for question in questions:
            q = dict(question)
            q.setdefault("designer_model", self.designer_model)
            q.setdefault("source_dataset", dataset_name)
            q.setdefault("modality", "text")
            q.setdefault("language", "en")
            q.setdefault("meta_notes", "")
            design_type = (q.get("design_type") or "standard").strip().lower()
            if design_type not in {"standard", "adversarial"}:
                design_type = "standard"
            q["design_type"] = design_type
            q.setdefault("question_type", "open_ended" if not q.get("options") else "mcq_single")
            if q["question_type"] in {"open_ended", "structured"}:
                q["options"] = []
            else:
                q.setdefault("options", [])
            if "uses_visual" not in q or q["uses_visual"] is None:
                q["uses_visual"] = False
            if "visual_instruction" not in q:
                q["visual_instruction"] = None
            normalized.append(q)

        return normalized

    def _deduplicate_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen_ids: Dict[str, int] = {}
        unique_questions: List[Dict[str, Any]] = []

        for question in questions:
            qid = question.get("id")
            if not qid:
                unique_questions.append(question)
                continue

            if qid not in seen_ids:
                seen_ids[qid] = 0
                unique_questions.append(question)
            else:
                seen_ids[qid] += 1
                new_question = dict(question)
                new_question["id"] = f"{qid}_dup{seen_ids[qid]}"
                unique_questions.append(new_question)

        return unique_questions

    def _trim_to_target(self, questions: List[Dict[str, Any]], coverage_config) -> List[Dict[str, Any]]:
        target_total = coverage_config.total_standard + coverage_config.total_adversarial
        if len(questions) <= target_total:
            return questions

        logger.warning(
            f"Generated {len(questions)} questions but target is {target_total}. Trimming extras."
        )
        return questions[:target_total]

    def _run_validation(self, dataset: str, questions_path: Path) -> Tuple[Optional[Path], bool]:
        if not self.validate:
            return None, True

        validator = QuestionValidator(dataset)
        validator.load_questions(questions_path)

        validation_filename = (
            f"validation_report_{self.safe_provider_name}_{self.safe_model_identifier}_{self.safe_designer_name}.json"
        )
        validation_path = questions_path.parent / validation_filename
        validator.save_report(validation_path)

        passed = len(validator.errors) == 0
        if passed:
            logger.info(f"[{dataset}] ✓ Validation complete")
        else:
            logger.warning(f"[{dataset}] Validation found {len(validator.errors)} issues")
        return validation_path, passed

    def _save_metadata(self, output_dir: Path, final_filename: str, total_questions: int, generator=None) -> None:
        metadata = {
            "designer_model": self.designer_model,
            "llm_provider": self.provider_name,
            "llm_model": self.model_identifier,
            "temperature": self.temperature,
            "questions_file": final_filename,
            "total_questions": total_questions
        }

        # Add quantization info for HuggingFace models
        if self.model == "huggingface" and generator and hasattr(generator, '_quantization_method'):
            metadata["precision"] = generator._quantization_method
        elif self.model == "huggingface":
            quantization = self.huggingface_options.get("quantization", "none")
            metadata["precision"] = quantization

        metadata_path = output_dir / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    async def process_dataset(self, dataset: str) -> dict:
        """
        Process a single dataset.

        Args:
            dataset: Dataset name

        Returns:
            Processing result summary
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing dataset: {dataset}")
        logger.info(f"{'='*80}\n")

        result = {
            "dataset": dataset,
            "success": False,
            "error": None,
            "questions_generated": 0,
            "validation_passed": False
        }

        try:
            domain_card = load_domain_card(dataset)
            coverage_config = get_coverage_config(dataset)
            dataset_display = get_dataset_display_name(domain_card, dataset)
            language = infer_language(dataset)
            modality = infer_modality(domain_card, dataset_name=dataset)
            super_context = extract_super_parent_context(domain_card)
            variant_instructions = get_additional_instructions_for_dataset(dataset)
            if variant_instructions:
                self.dataset_extra_instructions[dataset] = variant_instructions

            generator = QuestionGenerator(
                dataset_name=dataset,
                designer_model=self.designer_model,
                output_dir=self.output_base / dataset / self.safe_designer_name,
                ollama_model_override=self.ollama_model_override,
                ollama_base_url_override=self.ollama_base_url_override,
                vllm_base_url_override=self.vllm_base_url_override,
                huggingface_options=self.huggingface_options,
                llm_model_override=self.llm_model_override,
                vllm_max_tokens_override=self.vllm_max_tokens_override
            )
            # expose llm for batch polling helpers
            self._llm_ref = generator.llm

            batch_plans = self._plan_initial_batches(
                dataset=dataset,
                dataset_display=dataset_display,
                language=language,
                modality=modality,
                coverage_config=coverage_config,
                super_context=super_context,
                extra_instructions=self.dataset_extra_instructions.get(dataset, [])
            )

            if self.batch_mode in {"openai", "anthropic"} and self.model in {"openai", "anthropic"}:
                batch_results = await self._execute_batch_plans_batch_api(
                    generator,
                    batch_plans,
                    dataset
                )
            else:
                batch_results = await self._execute_batch_plans(
                    generator,
                    batch_plans,
                    dataset
                )

            all_questions = [
                question
                for batch in batch_results
                for question in batch["questions"]
            ]

            top_up_round = 0
            max_topup_rounds = self.max_topup_rounds or 0
            while top_up_round < max_topup_rounds:
                deficits = self._calculate_deficits(all_questions, coverage_config)
                if not deficits["needs_topup"]:
                    break

                topup_plans = self._plan_topup_batches(
                    deficits,
                    dataset=dataset,
                    dataset_display=dataset_display,
                    language=language,
                    modality=modality,
                    super_context=super_context,
                    coverage_config=coverage_config,
                    extra_instructions=self.dataset_extra_instructions.get(dataset, [])
                )

                if not topup_plans:
                    break

                logger.info(f"[{dataset}] Planning {len(topup_plans)} top-up batch(es) to close gaps")
                top_up_round += 1

                if self.batch_mode in {"openai", "anthropic"} and self.model in {"openai", "anthropic"}:
                    topup_results = await self._execute_batch_plans_batch_api(
                        generator,
                        topup_plans,
                        dataset,
                        is_top_up=True
                    )
                else:
                    topup_results = await self._execute_batch_plans(
                        generator,
                        topup_plans,
                        dataset,
                        is_top_up=True
                    )

                for res in topup_results:
                    batch_results.append(res)
                    all_questions.extend(res["questions"])

            all_questions = self._normalize_questions(all_questions)
            all_questions = self._deduplicate_questions(all_questions)
            all_questions = self._trim_to_target(all_questions, coverage_config)

            final_filename = f"questions_{self.safe_provider_name}_{self.safe_model_identifier}_{self.safe_designer_name}.jsonl"
            final_path = generator.save_questions(all_questions, final_filename)

            validation_path, validation_passed = self._run_validation(dataset, final_path)
            self._save_metadata(generator.output_dir, final_filename, len(all_questions), generator)

            result["success"] = validation_passed or not self.validate
            result["questions_generated"] = len(all_questions)
            result["questions_file"] = str(final_path)
            result["validation_file"] = str(validation_path) if validation_path else None
            result["validation_passed"] = validation_passed

        except Exception as e:
            logger.error(f"[{dataset}] ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)

        return result

    async def run(self) -> dict:
        """
        Run the complete Stage 2 pipeline for all datasets.

        Returns:
            Overall summary
        """
        logger.info("="*80)
        logger.info("STAGE 2: QUESTION GENERATION PIPELINE")
        logger.info("="*80)
        logger.info(f"Datasets: {', '.join(self.datasets)}")
        logger.info(f"Model: {self.model}")
        logger.info(f"Temperature: {self.temperature}")
        logger.info(f"Validation: {'Enabled' if self.validate else 'Disabled'}")
        logger.info("="*80 + "\n")

        # Process each dataset
        for dataset in self.datasets:
            result = await self.process_dataset(dataset)
            self.results[dataset] = result

        # Generate summary
        summary = self._generate_summary()

        # Save per-dataset summary files
        self._save_dataset_summaries(summary)

        return summary

    def _generate_summary(self) -> dict:
        """Generate processing summary"""
        total = len(self.datasets)
        successful = sum(1 for r in self.results.values() if r["success"])
        failed = total - successful

        total_questions = sum(r.get("questions_generated", 0) for r in self.results.values())

        if self.validate:
            validated = sum(1 for r in self.results.values() if r.get("validation_passed"))
        else:
            validated = None

        summary = {
            "stage": "Stage 2: Question Generation",
            "llm_provider": self.provider_name,
            "llm_model_identifier": self.model_identifier,
            "model": self.model,
            "designer_model": self.designer_model,
            "temperature": self.temperature,
            "datasets_processed": total,
            "datasets_successful": successful,
            "datasets_failed": failed,
            "total_questions_generated": total_questions,
            "validation_enabled": self.validate,
            "datasets_validated": validated,
            "results": self.results
        }

        # Print summary
        print("\n" + "="*80)
        print("STAGE 2 SUMMARY")
        print("="*80)
        print(f"Datasets Processed: {total}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"Total Questions Generated: {total_questions}")
        if self.validate:
            print(f"Datasets Validated (no errors): {validated}")
        print("="*80)

        # Print per-dataset results
        print("\nPer-Dataset Results:")
        print("-" * 80)
        for dataset, result in self.results.items():
            status = "✓" if result["success"] else "✗"
            questions = result.get("questions_generated", 0)
            print(f"{status} {dataset}: {questions} questions", end="")
            if self.validate and result.get("validation_passed"):
                print(" (validated)")
            elif self.validate:
                print(" (validation failed)")
            else:
                print()
        print("="*80)

        return summary

    def _save_dataset_summaries(self, summary: dict) -> None:
        """Save summary inside each dataset folder with model/designer-specific name."""
        filename = f"stage2_summary_{self.safe_provider_name}_{self.safe_designer_name}.json"

        for dataset, result in self.results.items():
            dataset_dir = self.output_base / dataset / self.safe_designer_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            dataset_summary = {
                "stage": summary["stage"],
                "dataset": dataset,
                "provider": self.provider_name,
                "model": self.model_identifier,
                "designer_model": summary["designer_model"],
                "temperature": summary["temperature"],
                "questions_generated": result.get("questions_generated", 0),
                "success": result.get("success", False),
                "validation_passed": result.get("validation_passed"),
                "error": result.get("error"),
                "result_details": result,
                "overall_summary": summary
            }

            summary_path = dataset_dir / filename
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(dataset_summary, f, indent=2)

            logger.info(f"\n✓ Summary saved to {summary_path}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Stage 2 question generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets:
  {', '.join(AVAILABLE_DATASETS)}

Examples:
  # Generate for CSBench English
  python run_stage2.py csbench_en

  # Generate for multiple datasets
  python run_stage2.py csbench_en tombench_en medxpertqa_text

  # Generate all datasets
  python run_stage2.py --all

  # Use different model
  python run_stage2.py csbench_en --model gemini

  # Use local Ollama with deepseek-r1
  python run_stage2.py csbench_en --model ollama

  # Skip validation
  python run_stage2.py csbench_en --no-validate
"""
    )

    parser.add_argument(
        "datasets",
        nargs="*",
        help="Dataset names to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available datasets"
    )
    parser.add_argument(
        "--model",
        default="openai",
        choices=["openai", "gemini", "anthropic", "grok", "deepseek", "doubao", "qwen", "llama", "ollama", "huggingface", "vllm"],
        help="LLM to use for generation (default: openai). Use 'ollama' for local Ollama, 'huggingface' for local transformers, or 'vllm' to call an OpenAI-compatible vLLM server."
    )
    parser.add_argument(
        "--batch-mode",
        choices=["none", "openai", "anthropic"],
        default="none",
        help="Use provider batch API instead of per-call requests (default: none)."
    )
    parser.add_argument(
        "--batch-prompts-per-file",
        type=int,
        default=20,
        help="How many prompts to pack into a single batch input (batch mode only)."
    )
    parser.add_argument(
        "--batch-max-tokens",
        type=int,
        default=8000,
        help="Max tokens per request in batch mode."
    )
    parser.add_argument(
        "--generation-profile",
        choices=sorted(GENERATION_PROFILES.keys()),
        default=None,
        help="Batch sizing preset: 'api' (default for hosted APIs) or 'local' (default for local engines)."
    )
    parser.add_argument(
        "--ollama-model",
        default=None,
        help="Override Ollama model name for this run (default: config file value)"
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Override Ollama base URL (include protocol, optional /v1 suffix)"
    )
    parser.add_argument(
        "--vllm-base-url",
        default=None,
        help="Override vLLM base URL (include protocol, optional /v1 suffix)"
    )
    parser.add_argument(
        "--vllm-max-tokens",
        type=int,
        default=None,
        help="Override max tokens per vLLM completion (default 16000)."
    )
    designer_default = "gpt-5-mini"
    parser.add_argument(
        "--designer-model",
        default=designer_default,
        help=f"Designer model identifier for tracking (default: {designer_default})"
    )
    parser.add_argument(
        "--hf-model-id",
        default=None,
        help="Hugging Face model id for --model huggingface (default: deepseek-ai/DeepSeek-R1)"
    )
    parser.add_argument(
        "--hf-device-map",
        default="auto",
        help="Device map passed to AutoModel.from_pretrained (e.g., 'auto', 'cuda:0', 'balanced', 'sequential')"
    )
    parser.add_argument(
        "--hf-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for Hugging Face models"
    )
    parser.add_argument(
        "--hf-max-new-tokens",
        type=int,
        default=2048,
        help="Max new tokens when generating with Hugging Face models"
    )
    parser.add_argument(
        "--hf-top-p",
        type=float,
        default=0.95,
        help="Top-p sampling for Hugging Face models"
    )
    parser.add_argument(
        "--hf-top-k",
        type=int,
        default=None,
        help="Top-k sampling cutoff for Hugging Face models"
    )
    parser.add_argument(
        "--hf-trust-remote-code",
        action="store_true",
        help="Allow Hugging Face models to execute custom code (trust_remote_code=True)"
    )
    parser.add_argument(
        "--hf-quantization",
        default=None,
        choices=["none", "int4_awq", "int4_gptq", "int4_bnb", "int8"],
        help="Quantization method: none (FP16/BF16), int4_awq (default for 70B+), int4_gptq, int4_bnb (bitsandbytes NF4), int8"
    )
    parser.add_argument(
        "--hf-load-in-8bit",
        action="store_true",
        help="[Deprecated] Load in 8-bit. Use --hf-quantization int8 instead"
    )
    parser.add_argument(
        "--hf-load-in-4bit",
        action="store_true",
        help="[Deprecated] Load in 4-bit BNB. Use --hf-quantization int4_bnb instead"
    )
    parser.add_argument(
        "--hf-attn-impl",
        default=None,
        help="Override attn_implementation when loading Hugging Face models (e.g., flash_attention_2)"
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Cache directory for Hugging Face models (default: use HF_HOME env var or ~/.cache/huggingface)"
    )
    # parser.add_argument(
    #     "--hf-max-memory-per-gpu",
    #     default=None,
    #     help="Max memory per GPU in GiB (e.g., '40GiB' for A6000 to leave headroom)"
    # )
    parser.add_argument(
        "--questions-per-call",
        type=int,
        default=None,
        help="Override questions per LLM call; defaults come from the selected generation profile."
    )
    parser.add_argument(
        "--min-batch-questions",
        type=int,
        default=None,
        help="Override minimum questions per batch (profile default otherwise)."
    )
    parser.add_argument(
        "--max-batch-questions",
        type=int,
        default=None,
        help="Override maximum questions per batch (profile default otherwise)."
    )
    parser.add_argument(
        "--max-topup-rounds",
        type=int,
        default=None,
        help="Override maximum number of top-up passes (profile default otherwise)."
    )
    parser.add_argument(
        "--llm-model-name",
        default=None,
        help="Override provider-specific model name for this run (e.g., gpt-4o, deepseek-reasoner)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: outputs/stage2_questions)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation after generation"
    )

    args = parser.parse_args()

    if args.llm_model_name:
        if args.designer_model != args.llm_model_name:
            logger.info(f"Aligning designer_model to LLM override: {args.llm_model_name}")
        args.designer_model = args.llm_model_name

    if args.model == "huggingface":
        # Handle backward compatibility for deprecated flags
        quantization = args.hf_quantization
        if args.hf_load_in_8bit and not quantization:
            quantization = "int8"
        elif args.hf_load_in_4bit and not quantization:
            quantization = "int4_bnb"

        # Auto-detect quantization based on model size if not specified
        if not quantization:
            model_id = args.hf_model_id or "deepseek-ai/DeepSeek-R1"
            if "70b" in model_id.lower() or "72b" in model_id.lower():
                quantization = "int4_awq"
                logger.info(f"Auto-detected 70B+ model, using quantization: {quantization}")
            elif "32b" in model_id.lower() or "30b" in model_id.lower():
                quantization = "none"
                logger.info(f"Auto-detected 32B model, using full precision (BF16)")
            else:
                quantization = "int4_awq"  # Safe default
                logger.info(f"Using default quantization: {quantization}")

        huggingface_options = {
            "model_id": args.hf_model_id,
            "device_map": args.hf_device_map,
            "dtype": args.hf_dtype,
            "quantization": quantization,
            "max_new_tokens": args.hf_max_new_tokens,
            "top_p": args.hf_top_p,
            "top_k": args.hf_top_k,
            "trust_remote_code": args.hf_trust_remote_code,
            "load_in_8bit": args.hf_load_in_8bit,  # Keep for backward compat
            "load_in_4bit": args.hf_load_in_4bit,  # Keep for backward compat
            "attn_implementation": args.hf_attn_impl,
            "cache_dir": args.hf_cache_dir,
            # "max_memory_per_gpu": args.hf_max_memory_per_gpu
        }
    else:
        huggingface_options = None

    # Determine datasets to process
    if args.all:
        datasets = AVAILABLE_DATASETS
    elif args.datasets:
        datasets = args.datasets
    else:
        parser.error("Must specify datasets or use --all")

    # Validate dataset names
    invalid = [d for d in datasets if d not in AVAILABLE_DATASETS]
    if invalid:
        parser.error(f"Invalid datasets: {', '.join(invalid)}\n"
                    f"Available: {', '.join(AVAILABLE_DATASETS)}")

    default_profile_name = (
        LOCAL_GENERATION_PROFILE.name if args.model in LOCAL_MODEL_DEFAULT_PROFILE else API_GENERATION_PROFILE.name
    )
    selected_profile_name = args.generation_profile or default_profile_name
    profile = GENERATION_PROFILES[selected_profile_name]

    questions_per_call = args.questions_per_call if args.questions_per_call is not None else profile.questions_per_call
    min_batch_questions = (
        args.min_batch_questions if args.min_batch_questions is not None else profile.min_batch_questions
    )
    max_batch_questions = (
        args.max_batch_questions if args.max_batch_questions is not None else profile.max_batch_questions
    )
    max_topup_rounds = args.max_topup_rounds if args.max_topup_rounds is not None else profile.max_topup_rounds
    max_topup_batch_questions = profile.max_topup_batch_questions

    if max_batch_questions < min_batch_questions:
        logger.warning(
            "max_batch_questions (%s) is less than min_batch_questions (%s); adjusting to match.",
            max_batch_questions,
            min_batch_questions,
        )
        max_batch_questions = min_batch_questions

    logger.info(
        "Using generation profile '%s': questions_per_call=%s, min_batch=%s, max_batch=%s, max_topup_rounds=%s",
        profile.name,
        questions_per_call,
        min_batch_questions,
        max_batch_questions,
        max_topup_rounds,
    )

    # Create orchestrator
    orchestrator = Stage2Orchestrator(
        datasets=datasets,
        model=args.model,
        designer_model=args.designer_model,
        temperature=args.temperature,
        output_base=args.output_dir,
        validate=not args.no_validate,
        ollama_model_override=args.ollama_model,
        ollama_base_url_override=args.ollama_base_url,
        huggingface_options=huggingface_options,
        questions_per_call=questions_per_call,
        min_batch_questions=min_batch_questions,
        max_batch_questions=max_batch_questions,
        max_topup_rounds=max_topup_rounds,
        max_topup_batch_questions=max_topup_batch_questions,
        llm_model_override=args.llm_model_name,
        profile_name=profile.name,
        vllm_base_url_override=args.vllm_base_url,
        vllm_max_tokens_override=args.vllm_max_tokens,
        batch_mode=args.batch_mode,
        batch_prompts_per_file=args.batch_prompts_per_file,
        batch_max_tokens=args.batch_max_tokens,
    )

    # Run pipeline
    try:
        summary = await orchestrator.run()

        # Exit with error if any dataset failed
        if summary["datasets_failed"] > 0:
            logger.error(f"\n{summary['datasets_failed']} dataset(s) failed")
            sys.exit(1)

        logger.info("\n✓ Stage 2 completed successfully!")
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())