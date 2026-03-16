#!/usr/bin/env python3
"""
Coverage Configuration for Stage 2 - Domain quotas and difficulty distribution
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class CoverageQuota:
    """Coverage requirements for a super_parent domain"""
    super_parent: str
    standard_questions: int
    adversarial_questions: int


@dataclass
class DatasetCoverageConfig:
    """Complete coverage configuration for a dataset"""
    dataset_name: str
    total_standard: int
    total_adversarial: int
    quotas: List[CoverageQuota]
    difficulty_distribution: Dict[str, float]  # L1-L5 percentages
    format_distribution: Dict[str, float]      # question_type percentages


# ============================================================================
# CS Bench Coverage Configurations
# ============================================================================

CSBENCH_EN_COVERAGE = DatasetCoverageConfig(
    dataset_name="csbench_en",
    total_standard=225,
    total_adversarial=75,
    quotas=[
        CoverageQuota("CS.Core.DataStructuresAlgorithms", 60, 20),
        CoverageQuota("CS.Core.OperatingSystem", 50, 15),
        CoverageQuota("CS.Core.ComputerNetwork", 40, 12),
        CoverageQuota("CS.Core.ComputerOrganization", 30, 10),
        CoverageQuota("CS.Systems.FileSystemAndIO", 20, 8),
        CoverageQuota("CS.Systems.ProcessesAndThreads", 15, 5),
        CoverageQuota("CS.Foundations.OverviewAndDefinitions", 10, 5),
    ],
    difficulty_distribution={
        "L1": 0.15,  # 15% - very easy
        "L2": 0.25,  # 25% - basic
        "L3": 0.30,  # 30% - moderate
        "L4": 0.20,  # 20% - advanced
        "L5": 0.10,  # 10% - very hard
    },
    format_distribution={
        "mcq_single": 0.60,      # 60% single-answer MCQ
        "mcq_multi": 0.10,       # 10% multi-answer MCQ
        "open_ended": 0.20,      # 20% short answer
        "structured": 0.10,      # 10% structured (table, list, etc.)
    }
)

CSBENCH_CN_COVERAGE = DatasetCoverageConfig(
    dataset_name="csbench_cn",
    total_standard=225,
    total_adversarial=75,
    quotas=[
        CoverageQuota("CS.Core.DataStructuresAlgorithms", 60, 20),
        CoverageQuota("CS.Core.OperatingSystem", 50, 15),
        CoverageQuota("CS.Core.ComputerNetwork", 40, 12),
        CoverageQuota("CS.Core.ComputerOrganization", 30, 10),
        CoverageQuota("CS.Systems.FileSystemAndIO", 20, 8),
        CoverageQuota("CS.Systems.ProcessesAndThreads", 15, 5),
        CoverageQuota("CS.Foundations.OverviewAndDefinitions", 10, 5),
    ],
    difficulty_distribution={
        "L1": 0.15,
        "L2": 0.25,
        "L3": 0.30,
        "L4": 0.20,
        "L5": 0.10,
    },
    format_distribution={
        "mcq_single": 0.60,
        "mcq_multi": 0.10,
        "open_ended": 0.20,
        "structured": 0.10,
    }
)

# ============================================================================
# ToMBench Coverage Configurations
# ============================================================================

TOMBENCH_EN_COVERAGE = DatasetCoverageConfig(
    dataset_name="tombench_en",
    total_standard=225,
    total_adversarial=75,
    quotas=[
        CoverageQuota("ToM.BeliefReasoning", 50, 15),
        CoverageQuota("ToM.EmotionRecognition", 40, 12),
        CoverageQuota("ToM.SocialInteraction", 35, 12),
        CoverageQuota("ToM.NonLiteralCommunication", 30, 10),
        CoverageQuota("ToM.IntentionReasoning", 30, 10),
        CoverageQuota("ToM.ObjectLocation", 25, 8),
        CoverageQuota("ToM.MoralJudgment", 15, 8),
    ],
    difficulty_distribution={
        "L1": 0.15,
        "L2": 0.25,
        "L3": 0.30,
        "L4": 0.20,
        "L5": 0.10,
    },
    format_distribution={
        "mcq_single": 0.65,      # 65% - narrative-based MCQ
        "mcq_multi": 0.05,       # 5% - multi-answer
        "open_ended": 0.20,      # 20% - explain reasoning
        "judgment": 0.10,        # 10% - judgment with explanation
    }
)

TOMBENCH_CN_COVERAGE = DatasetCoverageConfig(
    dataset_name="tombench_cn",
    total_standard=225,
    total_adversarial=75,
    quotas=[
        CoverageQuota("ToM.BeliefReasoning", 50, 15),
        CoverageQuota("ToM.EmotionRecognition", 40, 12),
        CoverageQuota("ToM.SocialInteraction", 35, 12),
        CoverageQuota("ToM.NonLiteralCommunication", 30, 10),
        CoverageQuota("ToM.IntentionReasoning", 30, 10),
        CoverageQuota("ToM.ObjectLocation", 25, 8),
        CoverageQuota("ToM.MoralJudgment", 15, 8),
    ],
    difficulty_distribution={
        "L1": 0.15,
        "L2": 0.25,
        "L3": 0.30,
        "L4": 0.20,
        "L5": 0.10,
    },
    format_distribution={
        "mcq_single": 0.65,
        "mcq_multi": 0.05,
        "open_ended": 0.20,
        "judgment": 0.10,
    }
)

# ============================================================================
# MedXpertQA Coverage Configurations
# ============================================================================

MEDXPERTQA_TEXT_COVERAGE = DatasetCoverageConfig(
    dataset_name="medxpertqa_text",
    total_standard=225,
    total_adversarial=75,
    quotas=[
        CoverageQuota("Medicine.Neurology", 25, 8),
        CoverageQuota("Medicine.EmergencyMedicine", 25, 8),
        CoverageQuota("Medicine.DigestiveSystem", 20, 7),
        CoverageQuota("Medicine.Cardiovascular", 20, 7),
        CoverageQuota("Medicine.MusculoskeletalDisorders", 20, 7),
        CoverageQuota("Medicine.InfectiousDiseases", 20, 7),
        CoverageQuota("Medicine.PharmacologicTherapy", 20, 7),
        CoverageQuota("Medicine.ReproductiveHealth", 15, 5),
        CoverageQuota("Medicine.ClinicalManagement", 15, 5),
        CoverageQuota("Medicine.DifferentialDiagnosis", 15, 5),
        CoverageQuota("Medicine.LaboratoryTesting", 15, 5),
        CoverageQuota("Medicine.PhysicalExamination", 15, 4),
    ],
    difficulty_distribution={
        "L1": 0.10,  # Less basic questions for medical
        "L2": 0.20,
        "L3": 0.35,
        "L4": 0.25,
        "L5": 0.10,
    },
    format_distribution={
        "mcq_single": 0.70,      # 70% - clinical case MCQ
        "mcq_multi": 0.05,       # 5% - multi-answer
        "open_ended": 0.15,      # 15% - diagnosis/management
        "structured": 0.10,      # 10% - differential diagnosis list
    }
)

MEDXPERTQA_MM_COVERAGE = DatasetCoverageConfig(
    dataset_name="medxpertqa_mm",
    total_standard=225,
    total_adversarial=75,
    quotas=[
        CoverageQuota("Medicine.DiagnosticImaging", 40, 15),
        CoverageQuota("Medicine.Histopathology", 30, 10),
        CoverageQuota("Medicine.RadiationOncology", 25, 8),
        CoverageQuota("Medicine.OrthopedicSurgery", 25, 8),
        CoverageQuota("Medicine.Neurology", 20, 7),
        CoverageQuota("Medicine.Cardiovascular", 20, 7),
        CoverageQuota("Medicine.PediatricPulmonology", 15, 5),
        CoverageQuota("Medicine.CardiacElectrophysiology", 15, 5),
        CoverageQuota("Medicine.EmergencyMedicine", 15, 5),
        CoverageQuota("Medicine.ClinicalManagement", 20, 5),
    ],
    difficulty_distribution={
        "L1": 0.10,
        "L2": 0.20,
        "L3": 0.35,
        "L4": 0.25,
        "L5": 0.10,
    },
    format_distribution={
        "mcq_single": 0.65,      # 65% - image interpretation MCQ
        "mcq_multi": 0.10,       # 10% - multi-answer
        "open_ended": 0.15,      # 15% - describe findings
        "structured": 0.10,      # 10% - report structure
    }
)

# ============================================================================
# WeMath Coverage Configuration
# ============================================================================

WEMATH_COVERAGE = DatasetCoverageConfig(
    dataset_name="wemath",
    total_standard=225,
    total_adversarial=75,
    quotas=[
        CoverageQuota("Math.Geometry.PlaneFigures", 60, 20),
        CoverageQuota("Math.Geometry.SolidFigures", 50, 15),
        CoverageQuota("Math.Geometry.TransformAndMotion", 40, 12),
        CoverageQuota("Math.Geometry.Measurement", 40, 15),
        CoverageQuota("Math.Geometry.PositionDirection", 35, 13),
    ],
    difficulty_distribution={
        "L1": 0.15,
        "L2": 0.25,
        "L3": 0.30,
        "L4": 0.20,
        "L5": 0.10,
    },
    format_distribution={
        "mcq_single": 0.50,      # 50% - visual MCQ
        "mcq_multi": 0.10,       # 10% - multi-answer
        "open_ended": 0.30,      # 30% - calculation/proof
        "structured": 0.10,      # 10% - step-by-step solution
    }
)

# ============================================================================
# Lookup Dictionary
# ============================================================================

DATASET_COVERAGE_MAP: Dict[str, DatasetCoverageConfig] = {
    "csbench_en": CSBENCH_EN_COVERAGE,
    "csbench_cn": CSBENCH_CN_COVERAGE,
    "csbench_fr": CSBENCH_EN_COVERAGE,  # Use same as EN
    "csbench_de": CSBENCH_EN_COVERAGE,  # Use same as EN
    "tombench_en": TOMBENCH_EN_COVERAGE,
    "tombench_cn": TOMBENCH_CN_COVERAGE,
    "medxpertqa_text": MEDXPERTQA_TEXT_COVERAGE,
    "medxpertqa_mm": MEDXPERTQA_MM_COVERAGE,
    "medxpertqa_mm_stage2_textonly": MEDXPERTQA_MM_COVERAGE,
    "medxpertqa_mm_stage2_visualprimed": MEDXPERTQA_MM_COVERAGE,
    "wemath": WEMATH_COVERAGE,
    "wemath_stage2_textonly": WEMATH_COVERAGE,
    "wemath_stage2_visualprimed": WEMATH_COVERAGE,
}


def get_coverage_config(dataset_name: str) -> DatasetCoverageConfig:
    """Get coverage configuration for a dataset"""
    if dataset_name not in DATASET_COVERAGE_MAP:
        raise ValueError(f"No coverage config for dataset: {dataset_name}")
    return DATASET_COVERAGE_MAP[dataset_name]


def format_coverage_table(config: DatasetCoverageConfig) -> str:
    """Format coverage quotas as a markdown table for prompt"""
    lines = [
        "| Super Parent | Standard Questions | Adversarial Questions |",
        "|--------------|-------------------|----------------------|"
    ]
    for quota in config.quotas:
        lines.append(f"| {quota.super_parent} | {quota.standard_questions} | {quota.adversarial_questions} |")

    total_std = sum(q.standard_questions for q in config.quotas)
    total_adv = sum(q.adversarial_questions for q in config.quotas)
    lines.append(f"| **TOTAL** | **{total_std}** | **{total_adv}** |")

    return "\n".join(lines)


def format_difficulty_distribution(config: DatasetCoverageConfig) -> str:
    """Format difficulty distribution for prompt"""
    lines = ["Difficulty distribution (across all 300 questions):"]
    for level, percentage in sorted(config.difficulty_distribution.items()):
        count = int(300 * percentage)
        lines.append(f"  - {level}: {percentage*100:.0f}% (~{count} questions)")
    return "\n".join(lines)


def format_format_distribution(config: DatasetCoverageConfig) -> str:
    """Format question format distribution for prompt"""
    lines = ["Question format distribution (across all 300 questions):"]
    for fmt, percentage in sorted(config.format_distribution.items()):
        count = int(300 * percentage)
        lines.append(f"  - {fmt}: {percentage*100:.0f}% (~{count} questions)")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test coverage configurations
    print("=== CSBench EN Coverage ===")
    config = get_coverage_config("csbench_en")
    print(format_coverage_table(config))
    print()
    print(format_difficulty_distribution(config))
    print()
    print(format_format_distribution(config))
