#!/usr/bin/env python3
"""
Prompt Builder for Stage 2 - Generate structured prompts for question generation
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from textwrap import dedent

from coverage_config import (
    get_coverage_config,
    format_coverage_table,
    format_difficulty_distribution,
    format_format_distribution
)
from question_schema import EXAMPLE_QUESTION


# Language-specific instructions
LANGUAGE_INSTRUCTIONS = {
    "en": "Write all questions in English, using clear academic style.",
    "zh": "所有问题使用中文，语言需简洁、专业、学术化。",
    "fr": "Rédige toutes les questions en français dans un style académique clair.",
    "de": "Formuliere alle Fragen auf Deutsch in einem klaren, akademischen Stil.",
}

# Modality-specific instructions
MODALITY_INSTRUCTIONS = {
    "text": "Use text-only questions: modality='text', uses_visual=false, visual_instruction=null.",
    "text+existing_image": (
        "Mainly text questions; you MAY reference existing images: for those, set modality='text+existing_image', "
        "uses_visual=true and give a 1–2 sentence visual_instruction."
    ),
    "multimodal": (
        "Mix text-only and visual questions (aim ~60–70% visual). For visual ones use modality='text+existing_image' "
        "or 'text+imagined_image', set uses_visual=true and provide a clear 1–2 sentence visual_instruction."
    ),
}

# Dataset-specific variant definitions for Stage 2 ablations
DATASET_VARIANTS: Dict[str, Dict[str, Any]] = {
    # WeMath variants
    "wemath_stage2_textonly": {
        "base_dataset": "wemath",
        "language_override": "en",
        "modality_override": "text",
        "display_suffix": "Stage 2 Text-only",
        "additional_instructions": [
            "Text-only condition: design questions that can be answered without any images or diagrams.",
            "Do NOT mention images, figures, diagrams, coordinate planes, or visuals; avoid phrases like 'as shown'.",
            "Describe geometry or spatial setups verbally so the problem is solvable from text alone."
        ],
    },
    "wemath_stage2_visualprimed": {
        "base_dataset": "wemath",
        "language_override": "en",
        "modality_override": "multimodal",
        "display_suffix": "Stage 2 Visual-primed",
        "additional_instructions": [
            "Multimodal condition: you may reference imagined visuals (diagrams, geometry pictures, coordinate planes, real-world scene photos).",
            "Only add a visual reference when the question genuinely needs a diagram or picture; keep descriptions concise.",
            "Do not fabricate pixel-level details—describe visuals at a high level (e.g., 'a triangle with sides labeled 3,4,5').",
            "Output remains text-only; images will be paired later."
        ],
    },
    # MedXpertQA-MM variants
    "medxpertqa_mm_stage2_textonly": {
        "base_dataset": "medxpertqa_mm",
        "language_override": "en",
        "modality_override": "text",
        "display_suffix": "Stage 2 Text-only",
        "additional_instructions": [
            "Text-only condition: write clinical questions answerable from text alone.",
            "Do NOT mention images, scans, X-rays, CT/MRI, derm photos, pathology slides, ECG strips, or figures.",
            "If you need to convey findings, describe them verbally (e.g., 'crackles at the right base') without referencing visuals."
        ],
    },
    "medxpertqa_mm_stage2_visualprimed": {
        "base_dataset": "medxpertqa_mm",
        "language_override": "en",
        "modality_override": "multimodal",
        "display_suffix": "Stage 2 Visual-primed",
        "additional_instructions": [
            "Multimodal condition: you may reference imagined visuals (radiology images, pathology slides, derm photos, ECGs, bedside ultrasound).",
            "Only include a visual reference when the question truly requires that image or chart; keep descriptions short and clinically plausible.",
            "Describe visuals at a high level (e.g., 'chest X-ray showing right lower lobe opacity'); do not invent pixel-level specifics.",
            "Output remains text-only; images will be paired later."
        ],
    },
}


@dataclass
class SuperParentContext:
    """Contextual data for a single super parent domain."""
    super_parent: str
    subdomains: List[str] = field(default_factory=list)
    topic_hints: List[str] = field(default_factory=list)
    sample_items: List[str] = field(default_factory=list)


@dataclass
class BatchPromptSpec:
    """Specification for generating a per-call user prompt."""
    dataset_name: str
    dataset_display_name: str
    designer_model: str
    super_parents: List[str]
    total_questions: int
    standard_questions: int
    adversarial_questions: int
    difficulty_targets: Dict[str, int]
    format_targets: Dict[str, int]
    id_prefix: str
    id_start_index: int = 1
    language: str = "en"
    modality: str = "text"
    super_parent_context: Dict[str, SuperParentContext] = field(default_factory=dict)
    additional_instructions: List[str] = field(default_factory=list)
def load_domain_card(dataset_name: str) -> Dict[str, Any]:
    """Load domain card YAML for a dataset"""
    base_dataset = resolve_base_dataset(dataset_name)
    # Construct path to domain card
    project_root = Path(__file__).parents[3]
    card_path = project_root / "domain_cards" / f"{base_dataset}_domain_card.yaml"

    if not card_path.exists():
        raise FileNotFoundError(f"Domain card not found: {card_path}")

    with open(card_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_dataset_display_name(domain_card: Dict[str, Any], dataset_fallback: str) -> str:
    meta = domain_card.get("meta", {})
    base_display = meta.get("dataset", dataset_fallback)
    variant = DATASET_VARIANTS.get(dataset_fallback)
    if variant:
        suffix = variant.get("display_suffix") or variant.get("label")
        if suffix:
            return f"{base_display} ({suffix})"
    return base_display


def resolve_base_dataset(dataset_name: str) -> str:
    """Return the base dataset name for aliases/variants."""
    return DATASET_VARIANTS.get(dataset_name, {}).get("base_dataset", dataset_name)


def infer_language(dataset_name: str) -> str:
    variant = DATASET_VARIANTS.get(dataset_name)
    if variant and variant.get("language_override"):
        return variant["language_override"]

    base = resolve_base_dataset(dataset_name)
    if "_cn" in base or base.endswith("_zh") or base == "wemath":
        return "zh"
    if "_fr" in base:
        return "fr"
    if "_de" in base:
        return "de"
    return "en"


def infer_modality(domain_card: Dict[str, Any], dataset_name: Optional[str] = None) -> str:
    if dataset_name:
        variant = DATASET_VARIANTS.get(dataset_name)
        if variant and variant.get("modality_override"):
            return variant["modality_override"]

    meta = domain_card.get("meta", {})
    modality_info = meta.get("modality", {})
    if modality_info.get("multimodal"):
        return "multimodal"
    if modality_info.get("text") and modality_info.get("existing_images"):
        return "text+existing_image"
    return "text"


def extract_super_parent_context(domain_card: Dict[str, Any]) -> Dict[str, SuperParentContext]:
    context: Dict[str, SuperParentContext] = {}

    ontology = domain_card.get("ontology", [])
    for entry in ontology:
        super_parent = entry.get("super_parent")
        if not super_parent:
            continue
        mid_levels = [
            ml.get("label", "").strip()
            for ml in entry.get("mid_level_parents", [])
            if ml.get("label")
        ]
        context.setdefault(super_parent, SuperParentContext(super_parent=super_parent)).subdomains = mid_levels

    glossary = domain_card.get("glossary", [])
    for entry in glossary:
        super_parent = entry.get("super_parent")
        if not super_parent:
            continue
        terms = entry.get("typical_terms", [])
        ctx = context.setdefault(super_parent, SuperParentContext(super_parent=super_parent))
        if terms:
            ctx.topic_hints = terms[:15]

    samples = domain_card.get("samples", [])
    for entry in samples:
        super_parent = entry.get("super_parent")
        if not super_parent:
            continue
        examples = entry.get("examples", [])
        ctx = context.setdefault(super_parent, SuperParentContext(super_parent=super_parent))
        ctx.sample_items = [
            f"[{ex.get('item_id', '')}] {ex.get('question', '')}".strip()
            for ex in examples[:2]
        ]

    return context


def format_domain_card_summary(domain_card: Dict[str, Any]) -> str:
    """Format domain card into a concise summary for the prompt"""
    lines = ["=== DOMAIN CARD SUMMARY ===\n"]
    meta = domain_card.get("meta", {})
    lines.append(f"Dataset: {meta.get('dataset', 'N/A')}")
    lines.append(f"Total items: {meta.get('total_items', 'N/A')}")
    modality = meta.get("modality", {})
    lines.append(f"Modality: Text={'Yes' if modality.get('text') else 'No'}, "
                 f"Multimodal={'Yes' if modality.get('multimodal') else 'No'}")
    lines.append("")
    return "\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = """
You are an expert exam-question writer for {dataset_display} ({dataset_name}).

Your ONLY job: given a user message describing domains, counts and distributions,
reply with ONE JSON array of question objects and NOTHING else.

Schema (field names and allowed values):

{{id, designer_model, source_dataset, super_parent, subdomain,
  design_type ∈ ["standard","adversarial"],
  modality, language,
  question_type ∈ ["mcq_single","mcq_multi","open_ended","structured"],
  question_stem, options, answer, answer_explanation,
  declared_difficulty ∈ ["L1","L2","L3","L4","L5"],
  estimated_time_sec (int), design_rationale,
  uses_visual (bool), visual_instruction, meta_notes}}

Every question object MUST include all of these fields.

Conventions:
- designer_model = "{designer_model}"
- source_dataset = "{dataset_name}"
- Defaults if not specified by the user:
  • modality = "text", uses_visual = false, visual_instruction = null
  • language = dataset language (English by default)
- Difficulty: L1..L5 go from very easy recall (L1) to very hard multi-step reasoning (L5, ~90–120s).
- Question types:
  • mcq_single: 4–5 options, exactly 1 correct; answer is a single letter.
  • mcq_multi: 4–5 options, usually 2–3 correct; answer is an array of letters.
  • open_ended / structured: options = []; answer is a short gold text; still provide explanation and rationale.
- design_type:
  • "standard" = normal fair exam questions.
  • "adversarial" = harder / edge-case but still precise and unambiguous.

Quality rules (for EVERY question):
- Original (no verbatim copying from examples), domain-correct, solvable from standard knowledge.
- Clear, unambiguous stem and options; plausible distractors, no nonsense.
- Avoid pure trivia and obscure implementation details unless canonical.

Output:
- Return exactly the requested number of questions in a single JSON array.
- No markdown, comments, or prose outside the array.
""".strip()


def build_system_prompt(dataset_name: str, designer_model: str) -> str:
    dataset_display = dataset_name.replace("_", " ").title()
    return dedent(SYSTEM_PROMPT_TEMPLATE).format(
        dataset_display=dataset_display,
        dataset_name=dataset_name,
        designer_model=designer_model,
    )


def _format_super_parent_block(context: SuperParentContext) -> str:
    parts = [f"- {context.super_parent}"]
    if context.subdomains:
        parts.append("  Subdomains: " + ", ".join(context.subdomains[:8]))
    if context.topic_hints:
        parts.append("  Topics: " + ", ".join(context.topic_hints[:10]))
    if context.sample_items:
        parts.append("  Example items: " + " | ".join(context.sample_items[:2]))
    return "\n".join(parts)


def build_batch_user_prompt(spec: BatchPromptSpec) -> str:
    """Build the per-call user prompt using the compact template."""
    language_instr = LANGUAGE_INSTRUCTIONS.get(spec.language, LANGUAGE_INSTRUCTIONS["en"])
    modality_instr = MODALITY_INSTRUCTIONS.get(spec.modality, MODALITY_INSTRUCTIONS["text"])

    id_start = spec.id_start_index
    id_end = spec.id_start_index + spec.total_questions - 1

    lines: List[str] = []

    lines.append(f"Task: generate {spec.total_questions} questions for dataset '{spec.dataset_display_name}'.")
    lines.append("")

    lines.append("Target super_parent domains (choose exactly one per question):")
    for sp in spec.super_parents:
        ctx = spec.super_parent_context.get(sp)
        if ctx:
            lines.append(_format_super_parent_block(ctx))
        else:
            lines.append(f"- {sp}")
    lines.append("")

    lines.append(f"Language: {language_instr} Apply this to ALL text fields.")
    lines.append(f"Modality: {modality_instr}")
    lines.append("")

    lines.append("Quantitative targets (match as closely as possible):")
    lines.append(f"- standard_questions: {spec.standard_questions}")
    lines.append(f"- adversarial_questions: {spec.adversarial_questions}")
    lines.append(f"- difficulty_targets: {spec.difficulty_targets}")
    lines.append(f"- question_type_targets: {spec.format_targets}")
    lines.append("")

    lines.append("ID rules:")
    lines.append(f'- Use IDs of the form "{spec.id_prefix}_qXXX" with zero-padded index.')
    lines.append(f"- Use sequential IDs from {spec.id_prefix}_q{id_start:03d} to {spec.id_prefix}_q{id_end:03d} with no gaps.")
    lines.append("")

    lines.append("Per-question fixed fields:")
    lines.append(f'- designer_model = "{spec.designer_model}"')
    lines.append(f'- source_dataset = "{spec.dataset_name}"')
    lines.append("")

    if spec.additional_instructions:
        lines.append("Extra guidance:")
        for note in spec.additional_instructions:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("Output:")
    lines.append(f"- Return only one single JSON array with exactly {spec.total_questions} question objects and nothing else.")
    lines.append("- Output only the JSON array; do not include commentary, summaries, or extra text before/after it.")

    return "\n".join(lines)


def get_additional_instructions_for_dataset(dataset_name: str) -> List[str]:
    """Return dataset-level extra guidance for prompts."""
    return DATASET_VARIANTS.get(dataset_name, {}).get("additional_instructions", [])

