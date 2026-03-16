#!/usr/bin/env python3
"""
Question Validator for Stage 2 - Validate and analyze generated questions
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict
import logging

from question_schema import Question
from coverage_config import get_coverage_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuestionValidator:
    """Validate and analyze generated questions"""

    def __init__(self, dataset_name: str):
        """
        Initialize validator.

        Args:
            dataset_name: Name of the dataset
        """
        self.dataset_name = dataset_name
        self.coverage_config = get_coverage_config(dataset_name)
        self.questions: List[Question] = []
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.question_validity: List[bool] = []

    def load_questions(self, questions_path: Path) -> int:
        """
        Load questions from JSONL file.

        Args:
            questions_path: Path to questions JSONL file

        Returns:
            Number of questions loaded
        """
        logger.info(f"Loading questions from {questions_path}...")

        self.questions = []
        with open(questions_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    question = Question.from_dict(data)
                    self.questions.append(question)
                except Exception as e:
                    self.errors.append({
                        "line": i,
                        "type": "load_error",
                        "message": f"Failed to load question: {e}"
                    })

        logger.info(f"Loaded {len(self.questions)} questions")
        return len(self.questions)

    def validate_all(self) -> Tuple[int, int]:
        """
        Validate all loaded questions.

        Returns:
            (valid_count, invalid_count)
        """
        logger.info("Validating questions...")

        valid_count = 0
        invalid_count = 0
        self.question_validity = []

        for i, question in enumerate(self.questions, 1):
            errors = question.validate()
            # Hard check: subdomain must be present and non-empty
            if not question.subdomain:
                errors = errors or []
                errors.append("Missing required field: subdomain")
            if errors:
                invalid_count += 1
                self.question_validity.append(False)
                for error in errors:
                    self.errors.append({
                        "question_id": question.id,
                        "question_index": i,
                        "type": "validation_error",
                        "message": error
                    })
            else:
                valid_count += 1
                self.question_validity.append(True)

        logger.info(f"Validation: {valid_count} valid, {invalid_count} invalid")
        return valid_count, invalid_count

    def analyze_coverage(self) -> Dict[str, Any]:
        """
        Analyze domain coverage against requirements.

        Returns:
            Coverage analysis report
        """
        logger.info("Analyzing coverage...")

        # Count by super_parent and design_type
        coverage_counts = defaultdict(lambda: {"standard": 0, "adversarial": 0})

        for question in self.questions:
            coverage_counts[question.super_parent][question.design_type] += 1

        # Compare with requirements
        coverage_analysis = {
            "total_questions": len(self.questions),
            "total_standard": sum(c["standard"] for c in coverage_counts.values()),
            "total_adversarial": sum(c["adversarial"] for c in coverage_counts.values()),
            "by_super_parent": {}
        }

        # Check each required domain
        for quota in self.coverage_config.quotas:
            parent = quota.super_parent
            actual = coverage_counts.get(parent, {"standard": 0, "adversarial": 0})

            analysis = {
                "required_standard": quota.standard_questions,
                "actual_standard": actual["standard"],
                "required_adversarial": quota.adversarial_questions,
                "actual_adversarial": actual["adversarial"],
                "standard_diff": actual["standard"] - quota.standard_questions,
                "adversarial_diff": actual["adversarial"] - quota.adversarial_questions
            }

            coverage_analysis["by_super_parent"][parent] = analysis

            # Generate warnings for significant deviations
            if abs(analysis["standard_diff"]) > 5:
                self.warnings.append({
                    "type": "coverage_deviation",
                    "super_parent": parent,
                    "message": f"Standard questions: expected {quota.standard_questions}, got {actual['standard']}"
                })

            if abs(analysis["adversarial_diff"]) > 3:
                self.warnings.append({
                    "type": "coverage_deviation",
                    "super_parent": parent,
                    "message": f"Adversarial questions: expected {quota.adversarial_questions}, got {actual['adversarial']}"
                })

        return coverage_analysis

    def analyze_difficulty(self) -> Dict[str, Any]:
        """
        Analyze difficulty distribution.

        Returns:
            Difficulty analysis report
        """
        logger.info("Analyzing difficulty distribution...")

        # Count by difficulty
        difficulty_counts = Counter(q.declared_difficulty for q in self.questions)

        # Expected counts
        total = len(self.questions)
        expected_counts = {
            level: int(total * percentage)
            for level, percentage in self.coverage_config.difficulty_distribution.items()
        }

        # Analysis
        difficulty_analysis = {
            "total_questions": total,
            "distribution": {}
        }

        for level in ["L1", "L2", "L3", "L4", "L5"]:
            actual = difficulty_counts.get(level, 0)
            expected = expected_counts.get(level, 0)
            percentage = actual / total if total > 0 else 0

            difficulty_analysis["distribution"][level] = {
                "expected": expected,
                "actual": actual,
                "percentage": round(percentage * 100, 1),
                "diff": actual - expected
            }

            # Warning for large deviations
            if abs(actual - expected) > 10:
                self.warnings.append({
                    "type": "difficulty_deviation",
                    "level": level,
                    "message": f"Difficulty {level}: expected ~{expected}, got {actual}"
                })

        return difficulty_analysis

    def analyze_format(self) -> Dict[str, Any]:
        """
        Analyze question format distribution.

        Returns:
            Format analysis report
        """
        logger.info("Analyzing format distribution...")

        # Count by format
        format_counts = Counter(q.question_type for q in self.questions)

        # Expected counts
        total = len(self.questions)
        expected_counts = {
            fmt: int(total * percentage)
            for fmt, percentage in self.coverage_config.format_distribution.items()
        }

        # Analysis
        format_analysis = {
            "total_questions": total,
            "distribution": {}
        }

        for fmt, expected in expected_counts.items():
            actual = format_counts.get(fmt, 0)
            percentage = actual / total if total > 0 else 0

            format_analysis["distribution"][fmt] = {
                "expected": expected,
                "actual": actual,
                "percentage": round(percentage * 100, 1),
                "diff": actual - expected
            }

            # Warning for large deviations
            if abs(actual - expected) > 15:
                self.warnings.append({
                    "type": "format_deviation",
                    "format": fmt,
                    "message": f"Format {fmt}: expected ~{expected}, got {actual}"
                })

        return format_analysis

    def analyze_modality(self) -> Dict[str, Any]:
        """
        Analyze modality distribution.

        Returns:
            Modality analysis report
        """
        logger.info("Analyzing modality distribution...")

        # Count by modality
        modality_counts = Counter(q.modality for q in self.questions)
        visual_count = sum(1 for q in self.questions if q.uses_visual)

        modality_analysis = {
            "total_questions": len(self.questions),
            "uses_visual_count": visual_count,
            "visual_percentage": round(visual_count / len(self.questions) * 100, 1) if self.questions else 0,
            "by_modality": dict(modality_counts)
        }

        return modality_analysis

    def check_duplicates(self) -> List[Dict[str, Any]]:
        """
        Check for duplicate question IDs or similar content.

        Returns:
            List of duplicate issues
        """
        logger.info("Checking for duplicates...")

        duplicates = []

        # Check ID duplicates
        id_counts = Counter(q.id for q in self.questions)
        for qid, count in id_counts.items():
            if count > 1:
                duplicates.append({
                    "type": "duplicate_id",
                    "id": qid,
                    "count": count
                })
                self.errors.append({
                    "type": "duplicate_id",
                    "id": qid,
                    "message": f"ID {qid} appears {count} times"
                })

        if not self.question_validity:
            self.question_validity = [True] * len(self.questions)

        # Check duplicate stems, mark later copies invalid
        stem_map: Dict[str, List[int]] = defaultdict(list)
        for idx, question in enumerate(self.questions):
            stem_raw = (question.question_stem or "").strip()
            if not stem_raw:
                continue
            normalized = " ".join(stem_raw.lower().split())
            if not normalized:
                continue
            stem_map[normalized].append(idx)

        for normalized, indices in stem_map.items():
            if len(indices) <= 1:
                continue

            canonical_idx = indices[0]
            canonical_question = self.questions[canonical_idx]
            duplicates.append({
                "type": "duplicate_stem",
                "stem": canonical_question.question_stem[:160],
                "count": len(indices),
                "question_ids": [self.questions[i].id for i in indices]
            })

            for dup_idx in indices[1:]:
                if 0 <= dup_idx < len(self.question_validity):
                    self.question_validity[dup_idx] = False
                self.errors.append({
                    "type": "duplicate_stem",
                    "question_id": self.questions[dup_idx].id,
                    "question_index": dup_idx + 1,
                    "message": (
                        f"Question stem duplicates earlier item "
                        f"(original index {canonical_idx + 1})"
                    )
                })

        return duplicates

    def generate_report(self) -> Dict[str, Any]:
        """
        Generate complete validation report.

        Returns:
            Full validation report
        """
        logger.info("Generating validation report...")

        # Run all analyses
        valid_count, invalid_count = self.validate_all()
        coverage = self.analyze_coverage()
        difficulty = self.analyze_difficulty()
        format_dist = self.analyze_format()
        modality = self.analyze_modality()
        duplicates = self.check_duplicates()

        adjusted_valid = sum(1 for is_valid in self.question_validity if is_valid)
        adjusted_invalid = len(self.questions) - adjusted_valid
        pass_rate = (
            round(adjusted_valid / len(self.questions) * 100, 1)
            if self.questions else 0
        )

        # Compile report
        report = {
            "dataset": self.dataset_name,
            "total_questions": len(self.questions),
            "validation": {
                "valid": adjusted_valid,
                "invalid": adjusted_invalid,
                "pass_rate": pass_rate,
                "initial_valid": valid_count,
                "initial_invalid": invalid_count
            },
            "coverage": coverage,
            "difficulty": difficulty,
            "format": format_dist,
            "modality": modality,
            "duplicates": {
                "count": len(duplicates),
                "issues": duplicates
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "overall_quality": self._calculate_quality_score()
            }
        }

        return report

    def _calculate_quality_score(self) -> str:
        """Calculate overall quality score"""
        if not self.questions:
            return "N/A"

        # Simple scoring based on errors and warnings
        error_penalty = len(self.errors) * 2
        warning_penalty = len(self.warnings)
        total_penalty = error_penalty + warning_penalty

        if total_penalty == 0:
            return "Excellent"
        elif total_penalty < 10:
            return "Good"
        elif total_penalty < 30:
            return "Fair"
        else:
            return "Needs Improvement"

    def save_report(self, output_path: Path):
        """Save validation report to JSON file"""
        report = self.generate_report()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved validation report to {output_path}")

        # Print summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Dataset: {self.dataset_name}")
        print(f"Total Questions: {len(self.questions)}")
        print(f"Valid: {report['validation']['valid']}")
        print(f"Invalid: {report['validation']['invalid']}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Overall Quality: {report['summary']['overall_quality']}")
        print("=" * 80)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate Stage 2 questions")
    parser.add_argument("dataset", help="Dataset name")
    parser.add_argument("questions_file", type=Path, help="Path to questions JSONL file")
    parser.add_argument("--output", type=Path, default=None,
                       help="Output path for validation report")

    args = parser.parse_args()

    # Create validator
    validator = QuestionValidator(args.dataset)

    # Load questions
    try:
        validator.load_questions(args.questions_file)
    except Exception as e:
        logger.error(f"Failed to load questions: {e}")
        return 1

    # Generate report
    output_path = args.output
    if output_path is None:
        output_path = args.questions_file.parent / "validation_report.json"

    validator.save_report(output_path)

    # Exit with error code if validation failed
    if validator.errors:
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
