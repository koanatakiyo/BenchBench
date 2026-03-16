#!/usr/bin/env python3
"""
Stage 3 post-hoc cleanup + targeted top-up generation for Stage 2 outputs.

Typical workflow for csbench_en:
1. Drop duplicate stems (keep earliest instance).
2. Trim oversubscribed super_parent/design_type buckets.
3. Compute deficits and, when requested, run small targeted top-ups that
   explicitly address the missing coverage (e.g., adversarial DS&A).
4. Save a cleaned JSONL file plus an audit report summarizing all actions.

Run per provider directory, e.g.:
python posthoc_clean_topup.py \
    --dataset csbench_en \
    --provider-dir /abs/path/to/outputs/stage2_questions/csbench_en/gpt-5_1 \
    --designer-model gpt-5.1 \
    --llm-provider openai \
    --llm-model-override gpt-5.1 \
    --temperature 0.6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# if __package__ in (None, ""):
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
STAGE2_ROOT = PACKAGE_ROOT / "stage2"

sys.path.insert(0, str(STAGE2_ROOT))
sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))

from stage2.coverage_config import get_coverage_config
from stage2.prompt_builder import (
    BatchPromptSpec,
    extract_super_parent_context,
    get_dataset_display_name,
    infer_language,
    infer_modality,
    load_domain_card,
    build_batch_user_prompt,
)
from stage2.question_generator import QuestionGenerator
# else:
#     from ..stage2.coverage_config import get_coverage_config
#     from ..stage2.prompt_builder import (
#         BatchPromptSpec,
#         extract_super_parent_context,
#         get_dataset_display_name,
#         infer_language,
#         infer_modality,
#         load_domain_card,
#         build_batch_user_prompt,
#     )
#     from ..stage2.question_generator import QuestionGenerator

logger = logging.getLogger("stage3.cleanup")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


DIFFICULTY_ORDER = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5}


def slugify(value: str) -> str:
    """Utility to produce filesystem-safe slugs."""
    cleaned = "".join(c.lower() if c.isalnum() else "_" for c in value)
    return "_".join(filter(None, cleaned.split("_")))


@dataclass
class TopUpRequest:
    """Represents a targeted deficit to fill."""

    super_parent: str
    design_type: str
    count: int


class Stage3PostHocCleaner:
    """Encapsulates duplicate trimming + targeted top-ups for a single provider output."""

    def __init__(
        self,
        dataset: str,
        provider_dir: Path,
        designer_model: str,
        llm_provider: str = "openai",
        question_file: Optional[Path] = None,
        output_file: Optional[Path] = None,
        report_file: Optional[Path] = None,
        temperature: float = 0.6,
        llm_model_override: Optional[str] = None,
        skip_topup: bool = False,
        dry_run: bool = False,
        max_avoid_stems: int = 12,
    ) -> None:
        self.dataset = dataset
        self.provider_dir = provider_dir.resolve()
        self.designer_model = designer_model
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.llm_model_override = llm_model_override
        self.skip_topup = skip_topup
        self.dry_run = dry_run
        self.max_avoid_stems = max(0, max_avoid_stems)

        self.coverage_config = get_coverage_config(dataset)
        self.domain_card = load_domain_card(dataset)
        self.dataset_display = get_dataset_display_name(self.domain_card, dataset)
        self.language = infer_language(dataset)
        self.modality = infer_modality(self.domain_card)
        self.super_parent_context = extract_super_parent_context(self.domain_card)

        self.questions_path = (
            question_file.resolve()
            if question_file
            else self._infer_question_file()
        )
        output_default = self.questions_path.with_name(
            f"{self.questions_path.stem}_stage3_clean.jsonl"
        )
        self.output_path = output_file.resolve() if output_file else output_default
        report_default = self.provider_dir / "stage3_posthoc_report.json"
        self.report_path = report_file.resolve() if report_file else report_default
        self.topup_dir = self.provider_dir / "stage3_topups"
        self.topup_dir.mkdir(parents=True, exist_ok=True)

        self.questions: List[Dict[str, Any]] = []
        self.initial_count = 0
        self.duplicate_removals: List[Dict[str, Any]] = []
        self.trimmed_removals: List[Dict[str, Any]] = []
        self.recently_removed_stems: List[str] = []
        self.deficits_before: Optional[Dict[str, Any]] = None
        self.deficits_after: Optional[Dict[str, Any]] = None
        self.topup_records: List[Dict[str, Any]] = []
        self.generator: Optional[QuestionGenerator] = None

    # ---------------------------------------------------------------------#
    # Public API
    # ---------------------------------------------------------------------#
    async def run(self) -> None:
        """Execute the full cleanup flow."""
        logger.info("Loading questions from %s", self.questions_path)
        self._load_questions()
        self._drop_duplicate_stems()
        self._trim_oversubscribed_buckets()

        self.deficits_before = self._calculate_deficits(self.questions)
        logger.info(
            "Post-clean count: %d (needs_topup=%s)",
            len(self.questions),
            self.deficits_before["needs_topup"],
        )

        if not self.skip_topup and self.deficits_before["needs_topup"]:
            if self.dry_run:
                logger.info("Dry-run mode: building top-up plan only (no LLM calls).")
                self.topup_records = self._preview_topups(self.deficits_before)
            else:
                logger.info("Running targeted top-ups to close deficits...")
                self.topup_records = await self._run_topups(self.deficits_before)
                if self.topup_records:
                    self.deficits_after = self._calculate_deficits(self.questions)
                else:
                    self.deficits_after = self.deficits_before
        else:
            self.deficits_after = self.deficits_before

        if not self.dry_run:
            logger.info("Saving cleaned questions to %s", self.output_path)
            self._save_questions(self.output_path)

        self._write_report()
        logger.info("Stage 3 cleanup complete.")

    # ---------------------------------------------------------------------#
    # Loading / Saving helpers
    # ---------------------------------------------------------------------#
    def _infer_question_file(self) -> Path:
        candidates = sorted(self.provider_dir.glob("questions_*.jsonl"))
        if not candidates:
            raise FileNotFoundError(
                f"No questions_*.jsonl files found under {self.provider_dir}"
            )
        if len(candidates) > 1:
            logger.warning(
                "Multiple question files found; defaulting to %s. "
                "Use --question-file to override.",
                candidates[0],
            )
        return candidates[0].resolve()

    def _load_questions(self) -> None:
        with open(self.questions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.questions.append(json.loads(line))
        self.initial_count = len(self.questions)

    def _save_questions(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for question in self.questions:
                f.write(json.dumps(question, ensure_ascii=False))
                f.write("\n")

    # ---------------------------------------------------------------------#
    # Cleaning steps
    # ---------------------------------------------------------------------#
    def _drop_duplicate_stems(self) -> None:
        seen: Dict[str, Dict[str, Any]] = {}
        unique_questions: List[Dict[str, Any]] = []

        for question in self.questions:
            stem = (question.get("question_stem") or "").strip()
            normalized = self._normalize_stem(stem)
            if not normalized:
                unique_questions.append(question)
                continue

            if normalized in seen:
                record = {
                    "question_id": question.get("id"),
                    "reason": "duplicate_stem",
                    "stem": stem,
                }
                self.duplicate_removals.append(record)
                self.recently_removed_stems.append(stem)
            else:
                seen[normalized] = question
                unique_questions.append(question)

        removed = len(self.questions) - len(unique_questions)
        if removed:
            logger.info("Removed %d duplicate stems.", removed)
        self.questions = unique_questions

    def _trim_oversubscribed_buckets(self) -> None:
        if not self.questions:
            return

        counts = self._count_by_super_parent(self.questions)
        removal_handles = set()

        for quota in self.coverage_config.quotas:
            for design_type in ("standard", "adversarial"):
                required = getattr(quota, f"{design_type}_questions")
                actual = counts[quota.super_parent][design_type]
                if actual <= required:
                    continue

                drop_count = actual - required
                trimmed = self._select_questions_to_trim(
                    quota.super_parent,
                    design_type,
                    drop_count,
                )
                for item in trimmed:
                    removal_handles.add(id(item))
                    self.trimmed_removals.append(
                        {
                            "question_id": item.get("id"),
                            "super_parent": quota.super_parent,
                            "design_type": design_type,
                            "declared_difficulty": item.get("declared_difficulty"),
                            "reason": f"over_quota_{design_type}",
                        }
                    )
                    stem = (item.get("question_stem") or "").strip()
                    if stem:
                        self.recently_removed_stems.append(stem)

        if removal_handles:
            before = len(self.questions)
            self.questions = [
                q for q in self.questions if id(q) not in removal_handles
            ]
            logger.info(
                "Trimmed %d oversubscribed questions.",
                before - len(self.questions),
            )

    # ---------------------------------------------------------------------#
    # Deficit analysis
    # ---------------------------------------------------------------------#
    def _calculate_deficits(self, questions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        coverage_counts = self._count_by_super_parent(questions)
        difficulty_counts = defaultdict(int)
        format_counts = defaultdict(int)

        for question in questions:
            difficulty_counts[question.get("declared_difficulty", "L3")] += 1
            format_counts[question.get("question_type", "open_ended")] += 1

        total_target = (
            self.coverage_config.total_standard
            + self.coverage_config.total_adversarial
        )
        current_total = len(list(questions))

        super_deficits: Dict[str, Dict[str, int]] = {}
        needs_topup = current_total < total_target

        for quota in self.coverage_config.quotas:
            stats = coverage_counts.get(quota.super_parent, {"standard": 0, "adversarial": 0})
            std_def = max(0, quota.standard_questions - stats["standard"])
            adv_def = max(0, quota.adversarial_questions - stats["adversarial"])
            if std_def > 0 or adv_def > 0:
                needs_topup = True
            super_deficits[quota.super_parent] = {
                "standard": std_def,
                "adversarial": adv_def,
            }

        difficulty_targets = self._allocate_distribution(
            self.coverage_config.difficulty_distribution,
            total_target,
            order=["L1", "L2", "L3", "L4", "L5"],
        )
        difficulty_deficits = {
            level: max(0, difficulty_targets[level] - difficulty_counts.get(level, 0))
            for level in difficulty_targets
        }
        if any(value > 0 for value in difficulty_deficits.values()):
            needs_topup = True

        format_targets = self._allocate_distribution(
            self.coverage_config.format_distribution,
            total_target,
        )
        format_deficits = {
            fmt: max(0, format_targets[fmt] - format_counts.get(fmt, 0))
            for fmt in format_targets
        }
        if any(value > 0 for value in format_deficits.values()):
            needs_topup = True

        return {
            "needs_topup": needs_topup,
            "super_parents": super_deficits,
            "difficulty": difficulty_deficits,
            "format": format_deficits,
            "total_target": total_target,
            "current_total": current_total,
            "remaining_total": max(0, total_target - current_total),
        }

    # ---------------------------------------------------------------------#
    # Top-up orchestration
    # ---------------------------------------------------------------------#
    def _preview_topups(self, deficits: Dict[str, Any]) -> List[Dict[str, Any]]:
        requests = self._build_topup_requests(deficits)
        previews: List[Dict[str, Any]] = []
        for idx, request in enumerate(requests, 1):
            difficulty_targets = self._allocate_difficulty_targets(
                request.count,
                prefer_hard=request.design_type == "adversarial",
                deficits_snapshot=deficits["difficulty"],
            )
            spec = self._build_prompt_spec(request, idx, difficulty_targets)
            prompt = build_batch_user_prompt(spec)
            previews.append(
                {
                    "request": request.__dict__,
                    "prompt_head": prompt[:750],
                    "difficulty_targets": difficulty_targets,
                    "format_targets": spec.format_targets,
                }
            )
        return previews

    async def _run_topups(self, deficits: Dict[str, Any]) -> List[Dict[str, Any]]:
        requests = self._build_topup_requests(deficits)
        if not requests:
            logger.info("No actionable deficits detected; skipping top-ups.")
            return []

        generator = self._get_generator()
        records: List[Dict[str, Any]] = []

        for idx, request in enumerate(requests, 1):
            difficulty_targets = self._allocate_difficulty_targets(
                request.count,
                prefer_hard=request.design_type == "adversarial",
                deficits_snapshot=deficits["difficulty"],
            )
            spec = self._build_prompt_spec(request, idx, difficulty_targets)
            prompt = build_batch_user_prompt(spec)

            prompt_file = self.topup_dir / f"{spec.id_prefix}_prompt.txt"
            prompt_file.write_text(prompt, encoding="utf-8")

            batch_questions = await generator.generate_questions_from_prompt(
                user_prompt=prompt,
                expected_questions=request.count,
                model=self.llm_provider,
                temperature=self.temperature,
            )
            normalized_questions = self._ensure_question_dicts(batch_questions)
            if not normalized_questions:
                logger.warning(
                    "Top-up request %s returned no valid questions.", spec.id_prefix
                )
                continue

            output_file = self.topup_dir / f"{spec.id_prefix}_questions.jsonl"
            self._save_questions_list(normalized_questions, output_file)
            self.questions.extend(normalized_questions)

            records.append(
                {
                    "request": request.__dict__,
                    "prompt_file": str(prompt_file),
                    "questions_file": str(output_file),
                    "generated": len(normalized_questions),
                    "difficulty_targets": difficulty_targets,
                    "format_targets": spec.format_targets,
                }
            )

        return records

    def _get_generator(self) -> QuestionGenerator:
        if self.generator is None:
            self.generator = QuestionGenerator(
                dataset_name=self.dataset,
                designer_model=self.designer_model,
                output_dir=self.provider_dir,
                llm_model_override=self.llm_model_override,
            )
        return self.generator

    # ---------------------------------------------------------------------#
    # Reporting
    # ---------------------------------------------------------------------#
    def _write_report(self) -> None:
        report = {
            "dataset": self.dataset,
            "designer_model": self.designer_model,
            "provider_dir": str(self.provider_dir),
            "input_file": str(self.questions_path),
            "output_file": str(self.output_path) if not self.dry_run else None,
            "dry_run": self.dry_run,
            "skip_topup": self.skip_topup,
            "initial_count": self.initial_count,
            "final_count": len(self.questions),
            "duplicate_removals": self.duplicate_removals,
            "trimmed_removals": self.trimmed_removals,
            "recently_removed_stems": self.recently_removed_stems[: self.max_avoid_stems],
            "deficits_before": self.deficits_before,
            "deficits_after": self.deficits_after,
            "topup_records": self.topup_records,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Wrote report to %s", self.report_path)

    # ---------------------------------------------------------------------#
    # Internal helpers
    # ---------------------------------------------------------------------#
    @staticmethod
    def _normalize_stem(stem: str) -> str:
        return " ".join(stem.lower().split())

    @staticmethod
    def _normalize_design_type(design_type: str) -> str:
        return "adversarial" if (design_type or "").lower() == "adversarial" else "standard"

    def _count_by_super_parent(
        self,
        questions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Dict[str, int]]:
        counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"standard": 0, "adversarial": 0})
        for question in questions:
            super_parent = question.get("super_parent") or "unknown"
            design_type = self._normalize_design_type(question.get("design_type", "standard"))
            counts[super_parent][design_type] += 1
        return counts

    def _select_questions_to_trim(
        self,
        super_parent: str,
        design_type: str,
        drop_count: int,
    ) -> List[Dict[str, Any]]:
        candidates: List[tuple] = []
        for question in self.questions:
            if question.get("super_parent") != super_parent:
                continue
            if self._normalize_design_type(question.get("design_type")) != design_type:
                continue
            difficulty = question.get("declared_difficulty", "L3")
            difficulty_rank = DIFFICULTY_ORDER.get(difficulty, 3)
            est_time = question.get("estimated_time_sec") or 0
            candidates.append(
                (difficulty_rank, est_time, question.get("id", ""), question)
            )

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in candidates[:drop_count]]

    def _build_topup_requests(self, deficits: Dict[str, Any]) -> List[TopUpRequest]:
        requests: List[TopUpRequest] = []
        for super_parent, slot in deficits["super_parents"].items():
            for design_type in ("standard", "adversarial"):
                needed = slot.get(design_type, 0)
                if needed > 0:
                    requests.append(
                        TopUpRequest(
                            super_parent=super_parent,
                            design_type=design_type,
                            count=needed,
                        )
                    )

        # Prioritize adversarial deficits first, then larger counts.
        requests.sort(
            key=lambda req: (
                0 if req.design_type == "adversarial" else 1,
                -req.count,
                req.super_parent,
            )
        )
        return requests

    def _allocate_difficulty_targets(
        self,
        count: int,
        prefer_hard: bool,
        deficits_snapshot: Dict[str, int],
    ) -> Dict[str, int]:
        allocation = {level: 0 for level in ["L1", "L2", "L3", "L4", "L5"]}
        remaining = count
        ordering = ["L5", "L4", "L3", "L2", "L1"] if prefer_hard else ["L3", "L2", "L4", "L1", "L5"]

        for level in ordering:
            needed = max(0, deficits_snapshot.get(level, 0))
            if needed <= 0:
                continue
            take = min(needed, remaining)
            if take > 0:
                allocation[level] += take
                deficits_snapshot[level] = max(0, needed - take)
                remaining -= take
            if remaining <= 0:
                break

        if remaining > 0:
            fallback = self._allocate_distribution(
                self.coverage_config.difficulty_distribution,
                remaining,
                order=["L1", "L2", "L3", "L4", "L5"],
            )
            for level, value in fallback.items():
                allocation[level] += value

        return allocation

    def _allocate_format_targets(self, count: int) -> Dict[str, int]:
        return self._allocate_distribution(
            self.coverage_config.format_distribution,
            count,
        )

    @staticmethod
    def _allocate_distribution(distribution: Dict[str, float], total: int, order: Optional[List[str]] = None) -> Dict[str, int]:
        if total <= 0:
            return {key: 0 for key in (order or distribution.keys())}

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

    def _build_prompt_spec(
        self,
        request: TopUpRequest,
        request_index: int,
        difficulty_targets: Dict[str, int],
    ) -> BatchPromptSpec:
        designer_slug = slugify(self.designer_model)
        dataset_slug = slugify(self.dataset)
        super_slug = slugify(request.super_parent)
        id_prefix = f"{designer_slug}_{dataset_slug}_{super_slug}_stage3_topup_{request_index:02d}"

        format_targets = self._allocate_format_targets(request.count)
        additional_instructions = [
            f"All questions MUST have design_type='{request.design_type}'.",
            (
                "These should be high-discrimination, edge-case items."
                if request.design_type == "adversarial"
                else "Keep questions fair but still original."
            ),
            f"Stay strictly within super_parent '{request.super_parent}'.",
        ]

        avoid_clause = self._build_avoid_instruction()
        if avoid_clause:
            additional_instructions.append(avoid_clause)

        return BatchPromptSpec(
            dataset_name=self.dataset,
            dataset_display_name=self.dataset_display,
            designer_model=self.designer_model,
            super_parents=[request.super_parent],
            total_questions=request.count,
            standard_questions=request.count if request.design_type == "standard" else 0,
            adversarial_questions=request.count if request.design_type == "adversarial" else 0,
            difficulty_targets=difficulty_targets,
            format_targets=format_targets,
            id_prefix=id_prefix,
            id_start_index=1,
            language=self.language,
            modality=self.modality,
            super_parent_context=self.super_parent_context,
            additional_instructions=additional_instructions,
        )

    def _build_avoid_instruction(self) -> Optional[str]:
        if not self.recently_removed_stems or self.max_avoid_stems <= 0:
            return None
        stems = [
            stem.replace("\n", " ")[:160]
            for stem in self.recently_removed_stems[: self.max_avoid_stems]
        ]
        return "Do not repeat any of these stems verbatim or conceptually: " + " | ".join(stems)

    @staticmethod
    def _ensure_question_dicts(payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, list):
            logger.warning("Expected list of questions but received %s", type(payload).__name__)
            return []

        normalized: List[Dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                normalized.append(item)
        return normalized

    @staticmethod
    def _save_questions_list(questions: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for question in questions:
                f.write(json.dumps(question, ensure_ascii=False))
                f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc Stage 3 cleanup/top-up helper.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g., csbench_en.")
    parser.add_argument("--provider-dir", required=True, type=Path, help="Path to the Stage 2 provider folder.")
    parser.add_argument("--designer-model", required=True, help="Designer model identifier (for metadata + prompts).")
    parser.add_argument("--llm-provider", default="openai", help="LLM provider key used by QuestionGenerator.")
    parser.add_argument("--llm-model-override", default=None, help="LLM model override passed to QuestionGenerator.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for top-ups.")
    parser.add_argument("--question-file", type=Path, help="Explicit path to questions JSONL (default: auto-detect).")
    parser.add_argument("--output-file", type=Path, help="Where to write cleaned JSONL (default: *_stage3_clean.jsonl).")
    parser.add_argument("--report-file", type=Path, help="Where to write the summary report (default: provider_dir/stage3_posthoc_report.json).")
    parser.add_argument("--max-avoid-stems", type=int, default=12, help="How many removed stems to include in 'avoid duplicates' instruction.")
    parser.add_argument("--skip-topup", action="store_true", help="Skip LLM calls even if deficits remain.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write outputs; only emit the report with plans.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cleaner = Stage3PostHocCleaner(
        dataset=args.dataset,
        provider_dir=args.provider_dir,
        designer_model=args.designer_model,
        llm_provider=args.llm_provider,
        question_file=args.question_file,
        output_file=args.output_file,
        report_file=args.report_file,
        temperature=args.temperature,
        llm_model_override=args.llm_model_override,
        skip_topup=args.skip_topup,
        dry_run=args.dry_run,
        max_avoid_stems=args.max_avoid_stems,
    )
    await cleaner.run()


if __name__ == "__main__":
    asyncio.run(main())

