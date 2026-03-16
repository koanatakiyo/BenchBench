#!/usr/bin/env python3
"""
Question Schema for Stage 2 - Unified schema for all generated questions
"""

from typing import List, Optional, Literal, Dict, Any
from dataclasses import dataclass, asdict
import json


# Type definitions
DesignType = Literal["standard", "adversarial"]
Modality = Literal["text", "text+existing_image", "text+imagined_image"]
Language = Literal["en", "zh", "fr", "de"]
QuestionType = Literal[
    "mcq_single",       # Single correct answer MCQ
    "mcq_multi",        # Multiple correct answers MCQ
    "open_ended",       # Free-form answer
    "cloze",            # Fill in the blank
    "structured",       # Structured format (e.g., table, list)
    "judgment",         # True/False with explanation
]
DifficultyLevel = Literal["L1", "L2", "L3", "L4", "L5"]


@dataclass
class Question:
    """
    Unified question schema for Stage 2 output.

    This schema is consistent with Stage 3 requirements and includes:
    - Design metadata (model, dataset, domain)
    - Question type and modality
    - Content (stem, options, answer, explanation)
    - Difficulty and design rationale
    - Visual information (if applicable)
    """

    # === Identity & Metadata ===
    id: str                             # e.g., "gpt5_csbench_en_q001"
    designer_model: str                 # Model that generated this question
    source_dataset: str                 # e.g., "csbench_en", "tombench_cn"
    super_parent: str                   # High-level domain (e.g., "CS.MemoryManagement")
    subdomain: str                      # More specific subdomain

    # === Design Classification ===
    design_type: DesignType             # "standard" or "adversarial"
    modality: Modality                  # "text", "text+existing_image", etc.
    language: Language                  # "en", "zh", "fr", "de"
    question_type: QuestionType         # "mcq_single", "open_ended", etc.

    # === Question Content ===
    question_stem: str                  # The main question text
    options: List[str]                  # List of options (empty for open-ended)
    answer: str | List[str]             # Answer key(s): "B", ["B", "C"], or text
    answer_explanation: str             # Why this answer is correct

    # === Difficulty & Design ===
    declared_difficulty: DifficultyLevel  # L1-L5
    estimated_time_sec: Optional[int]   # Optional time estimate
    design_rationale: str               # Why this question was designed

    # === Visual Information ===
    uses_visual: bool                   # Does this question use visuals?
    visual_instruction: Optional[str]   # Description of visual (if applicable)

    # === Additional Metadata ===
    meta_notes: Optional[str] = None    # Any additional notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        """Create Question from dictionary"""
        return cls(**data)

    def validate(self) -> List[str]:
        """
        Validate the question and return list of validation errors.
        Returns empty list if valid.
        """
        errors = []

        # Check required fields
        if not self.id:
            errors.append("id is required")
        if not self.question_stem or len(self.question_stem.strip()) < 10:
            errors.append("question_stem must be at least 10 characters")
        if not self.answer:
            errors.append("answer is required")
        explanation = (self.answer_explanation or "").strip()
        if not explanation:
            errors.append("answer_explanation must not be empty")
        elif len(explanation) < 10:
            errors.append("answer_explanation must be at least 10 characters")
        if not self.design_rationale or len(self.design_rationale.strip()) < 15:
            errors.append("design_rationale must be at least 15 characters")

        # Validate MCQ questions
        if self.question_type in ["mcq_single", "mcq_multi"]:
            if not self.options or len(self.options) < 2:
                errors.append(f"{self.question_type} requires at least 2 options")
            if self.question_type == "mcq_single":
                if not isinstance(self.answer, str):
                    errors.append("mcq_single answer must be a single string")
                elif self.answer not in ["A", "B", "C", "D", "E", "F"]:
                    errors.append(f"Invalid answer key: {self.answer}")
            elif self.question_type == "mcq_multi":
                if not isinstance(self.answer, list):
                    errors.append("mcq_multi answer must be a list")
                elif not all(a in ["A", "B", "C", "D", "E", "F"] for a in self.answer):
                    errors.append(f"Invalid answer keys: {self.answer}")

        # Validate open-ended questions
        elif self.question_type in ["open_ended", "cloze", "structured", "judgment"]:
            if self.options:
                errors.append(f"{self.question_type} should not have options")

        # Validate visual fields
        if self.uses_visual and not self.visual_instruction:
            errors.append("uses_visual is True but visual_instruction is missing")
        if not self.uses_visual and self.visual_instruction:
            errors.append("visual_instruction provided but uses_visual is False")

        # Validate difficulty
        if self.declared_difficulty not in ["L1", "L2", "L3", "L4", "L5"]:
            errors.append(f"Invalid difficulty: {self.declared_difficulty}")

        # Validate time estimate
        if self.estimated_time_sec is not None:
            if self.estimated_time_sec < 10 or self.estimated_time_sec > 600:
                errors.append(f"estimated_time_sec should be between 10-600 seconds")

        return errors


# Example question for reference
EXAMPLE_QUESTION = Question(
    id="gpt5_csbench_en_q001",
    designer_model="gpt-5-mini",
    source_dataset="csbench_en",
    super_parent="CS.Core.OperatingSystem",
    subdomain="memory_management",
    design_type="standard",
    modality="text",
    language="en",
    question_type="mcq_single",
    question_stem="What happens when a TLB miss occurs in a paged virtual memory system?",
    options=[
        "The CPU immediately raises a page fault and the OS terminates the process.",
        "The MMU looks up the page table entry in main memory and updates the TLB.",
        "The disk controller loads the missing page table entry from swap.",
        "The TLB entries are flushed and rebuilt from scratch."
    ],
    answer="B",
    answer_explanation="On a TLB miss, the hardware (or OS) walks the page table in memory and inserts the translation into the TLB; only if the page is not present does a page fault occur.",
    declared_difficulty="L3",
    estimated_time_sec=45,
    design_rationale="Tests knowledge of TLB vs page fault; common confusion point.",
    uses_visual=False,
    visual_instruction=None,
    meta_notes="Based on memory_management subdomains from domain card."
)


if __name__ == "__main__":
    # Test the schema
    print("Example Question Schema:")
    print(EXAMPLE_QUESTION.to_json())
    print("\nValidation:")
    errors = EXAMPLE_QUESTION.validate()
    if errors:
        print("Validation errors:", errors)
    else:
        print("✓ Question is valid")
