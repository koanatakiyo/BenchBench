"""
Stage 2: Question Generation

This stage generates 300 high-quality benchmark questions per dataset based on
domain cards from Stage 1.

Key Features:
- 225 standard questions + 75 adversarial questions
- Explicit domain coverage quotas
- Difficulty distribution (L1-L5)
- Format diversity (MCQ, open-ended, structured, etc.)
- Multimodal support (text, text+image, imagined visuals)
- Rich JSON schema compatible with Stage 3

Modules:
- question_schema: Unified question schema definition
- coverage_config: Domain coverage quotas and distributions
- prompt_builder: Structured prompt generation for LLMs
- question_generator: LLM-based question generation
- validate_questions: Quality validation and analysis
- run_stage2: Orchestrator script

Usage:
    python run_stage2.py csbench_en --model openai
"""

from .question_schema import Question, EXAMPLE_QUESTION
from .coverage_config import (
    get_coverage_config,
    DatasetCoverageConfig,
    CoverageQuota,
    DATASET_COVERAGE_MAP
)
from .prompt_builder import (
    build_system_prompt,
    build_batch_user_prompt,
    load_domain_card,
    BatchPromptSpec,
    extract_super_parent_context
)
from .question_generator import QuestionGenerator
from .validate_questions import QuestionValidator

__all__ = [
    "Question",
    "EXAMPLE_QUESTION",
    "get_coverage_config",
    "DatasetCoverageConfig",
    "CoverageQuota",
    "DATASET_COVERAGE_MAP",
    "build_system_prompt",
    "build_batch_user_prompt",
    "load_domain_card",
    "BatchPromptSpec",
    "extract_super_parent_context",
    "QuestionGenerator",
    "QuestionValidator",
]
