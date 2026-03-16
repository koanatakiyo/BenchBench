"""
Data Quality Validator

Validates data quality for loaded datasets, checking:
- Required fields present
- Data types correct
- Language consistency
- Format consistency
- Modality labels match content (e.g., images present for multimodal items)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


@dataclass
class ValidationIssue:
    """A single validation issue"""
    item_id: str
    severity: str  # "error", "warning", "info"
    category: str  # "missing_field", "type_error", "consistency", "format"
    message: str
    field: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Validation report for a dataset"""
    dataset_name: str
    total_items: int
    passed_items: int
    failed_items: int
    warnings: int

    issues: List[ValidationIssue] = field(default_factory=list)

    # Field completeness
    fields_completeness: Dict[str, float] = field(default_factory=dict)

    # Language consistency
    language_distribution: Dict[str, int] = field(default_factory=dict)
    language_purity: float = 0.0

    # Format consistency
    format_distribution: Dict[str, int] = field(default_factory=dict)

    # Modality consistency
    modality_distribution: Dict[str, int] = field(default_factory=dict)
    modality_content_match: float = 0.0  # % of multimodal items that actually have images

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert issues list
        data['issues'] = [issue.to_dict() if isinstance(issue, ValidationIssue) else issue for issue in self.issues]
        return data


class QualityValidator:
    """Validates data quality for datasets"""

    # Required fields for all items
    REQUIRED_FIELDS = ['id', 'question', 'answer', 'language', 'format', 'modality']

    # Optional but expected fields
    EXPECTED_FIELDS = ['options', 'context', 'background']

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the quality validator
        
        Args:
            project_root: Root directory of the project. If None, auto-detect from script location
        """
        # Auto-detect project root if not provided
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(project_root)
        self.raw_data_dir = self.project_root / 'data' / 'raw_data'
        
        # Basic language aliases to reduce false mismatches
        self.language_aliases = {
            'en': 'english', 'eng': 'english', 'english': 'english',
            'zh': 'chinese', 'cn': 'chinese', 'chi': 'chinese', 'chinese': 'chinese',
            'fr': 'french', 'french': 'french',
            'de': 'german', 'ger': 'german', 'german': 'german'
        }

    def _normalize_language(self, value: Optional[str]) -> str:
        if not value:
            return 'unknown'
        v = str(value).strip().lower()
        return self.language_aliases.get(v, v)
    
    def _check_images_exist(self, item: Dict[str, Any], dataset_name: Optional[str] = None) -> bool:
        """Check if images referenced in the item actually exist on disk
        
        Args:
            item: Dataset item to check
            dataset_name: Name of the dataset (for determining image base path)
        
        Returns:
            True if images field exists AND at least one image file exists on disk
        """
        images = item.get('images')
        if not images:
            # Also check background.image_path
            bg_image_path = item.get('background', {}).get('image_path')
            if bg_image_path:
                images = [bg_image_path] if isinstance(bg_image_path, str) else bg_image_path
            else:
                return False
        
        # Normalize to list format
        if isinstance(images, str):
            image_paths = [images]
        elif isinstance(images, list):
            image_paths = []
            for img in images:
                if isinstance(img, str):
                    image_paths.append(img)
                elif isinstance(img, dict) and 'image_path' in img:
                    image_paths.append(img['image_path'])
        else:
            return False
        
        if not image_paths:
            return False
        
        # Determine base directory for images based on dataset
        # For wemath: raw_data/wemath/2steps or raw_data/wemath/3steps
        # For medxpertqa_mm: raw_data/medxpertqa/images
        base_dir = None
        if dataset_name and 'wemath' in dataset_name.lower():
            # Check if path contains 2steps or 3steps
            first_path = image_paths[0] if image_paths else ''
            if '2steps' in first_path or first_path.startswith('2steps'):
                base_dir = self.raw_data_dir / 'wemath' / '2steps'
            elif '3steps' in first_path or first_path.startswith('3steps'):
                base_dir = self.raw_data_dir / 'wemath' / '3steps'
            else:
                # Default to checking both
                base_dir = self.raw_data_dir / 'wemath'
        elif dataset_name and 'medxpertqa' in dataset_name.lower():
            base_dir = self.raw_data_dir / 'medxpertqa' / 'images'
        else:
            # Unknown dataset, just check if images field exists
            return bool(images)
        
        # Check if at least one image file exists
        for img_path in image_paths:
            # Handle relative paths
            if '/' in img_path:
                # Path like "2steps/image/1-1.png" -> remove "2steps/" or "3steps/" prefix
                parts = img_path.split('/', 1)
                if len(parts) == 2 and parts[0] in ['2steps', '3steps']:
                    rel_path = parts[1]  # "image/1-1.png"
                else:
                    rel_path = img_path
            else:
                # Just filename like "MM-0-a.jpeg"
                rel_path = img_path
            
            # Try to find the file
            if base_dir and (base_dir / rel_path).exists():
                return True
            # Also try with "image/" subdirectory for medxpertqa
            if base_dir and (base_dir.parent / 'images' / rel_path).exists():
                return True
        
        return False

    def validate_item(
        self,
        item: Dict[str, Any],
        expected_language: Optional[str] = None,
        expected_format: Optional[str] = None,
        expected_modality: Optional[str] = None,
        dataset_name: Optional[str] = None
    ) -> List[ValidationIssue]:
        """Validate a single item

        Args:
            item: Dataset item to validate
            expected_language: Expected language for this dataset
            expected_format: Expected format for this dataset
            expected_modality: Expected modality for this dataset
            dataset_name: Name of the dataset (for image path resolution)

        Returns:
            List of validation issues
        """
        issues = []
        item_id = item.get('id', 'unknown')

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if field not in item or item[field] is None or item[field] == '':
                issues.append(ValidationIssue(
                    item_id=item_id,
                    severity='error',
                    category='missing_field',
                    message=f'Required field "{field}" is missing or empty',
                    field=field
                ))

        # Check language consistency (case-insensitive with aliases)
        if expected_language:
            exp_norm = self._normalize_language(expected_language)
            got_norm = self._normalize_language(item.get('language'))
            if got_norm != exp_norm:
                issues.append(ValidationIssue(
                    item_id=item_id,
                    severity='warning',
                    category='consistency',
                    message=f'Language mismatch: expected "{expected_language}", got "{item.get("language")}"',
                    field='language'
                ))

        # Check format consistency
        if expected_format and item.get('format') != expected_format:
            issues.append(ValidationIssue(
                item_id=item_id,
                severity='warning',
                category='consistency',
                message=f'Format mismatch: expected "{expected_format}", got "{item.get("format")}"',
                field='format'
            ))

        # Check modality consistency
        modality = item.get('modality', 'Text')
        has_images = self._check_images_exist(item, dataset_name=dataset_name or item.get('dataset_name'))

        if modality.lower() in ['multimodal', 'visual', 'mixed'] and not has_images:
            issues.append(ValidationIssue(
                item_id=item_id,
                severity='warning',
                category='consistency',
                message=f'Item marked as "{modality}" but no images found or image files do not exist',
                field='modality'
            ))

        # Check question is not empty
        if item.get('question') and len(str(item['question']).strip()) < 10:
            issues.append(ValidationIssue(
                item_id=item_id,
                severity='warning',
                category='format',
                message='Question is very short (< 10 characters)',
                field='question'
            ))

        # Check answer format
        answer = item.get('answer')
        if answer is not None:
            if isinstance(answer, list) and len(answer) == 0:
                issues.append(ValidationIssue(
                    item_id=item_id,
                    severity='error',
                    category='format',
                    message='Answer is an empty list',
                    field='answer'
                ))
            elif not answer:
                issues.append(ValidationIssue(
                    item_id=item_id,
                    severity='error',
                    category='format',
                    message='Answer is empty',
                    field='answer'
                ))

        # Check options for multiple-choice questions
        if item.get('format') == 'Multiple-choice':
            options = item.get('options', [])
            if not options or len(options) < 2:
                issues.append(ValidationIssue(
                    item_id=item_id,
                    severity='error',
                    category='format',
                    message='Multiple-choice question has fewer than 2 options',
                    field='options'
                ))

        return issues

    def validate_dataset(
        self,
        items: List[Dict[str, Any]],
        dataset_name: str,
        expected_language: Optional[str] = None,
        expected_format: Optional[str] = None,
        expected_modality: Optional[str] = None
    ) -> ValidationReport:
        """Validate an entire dataset

        Args:
            items: List of dataset items
            dataset_name: Name of the dataset
            expected_language: Expected language for this dataset
            expected_format: Expected format for this dataset
            expected_modality: Expected modality for this dataset

        Returns:
            ValidationReport
        """
        total_items = len(items)
        all_issues = []

        # Validate each item
        for item in items:
            item_issues = self.validate_item(
                item,
                expected_language=expected_language,
                expected_format=expected_format,
                expected_modality=expected_modality,
                dataset_name=dataset_name
            )
            all_issues.extend(item_issues)

        # Count passed/failed items
        failed_item_ids = set()
        warning_count = 0

        for issue in all_issues:
            if issue.severity == 'error':
                failed_item_ids.add(issue.item_id)
            elif issue.severity == 'warning':
                warning_count += 1

        failed_items = len(failed_item_ids)
        passed_items = total_items - failed_items

        # Calculate field completeness
        fields_completeness = {}
        for field in self.REQUIRED_FIELDS + self.EXPECTED_FIELDS:
            present_count = sum(1 for item in items if field in item and item[field])
            fields_completeness[field] = present_count / total_items if total_items > 0 else 0.0

        # Calculate language distribution and purity
        from collections import Counter
        language_distribution = dict(Counter(item.get('language', 'Unknown') for item in items))
        primary_language = max(language_distribution.items(), key=lambda x: x[1])[0] if language_distribution else 'Unknown'
        language_purity = language_distribution.get(primary_language, 0) / total_items if total_items > 0 else 0.0

        # Calculate format distribution
        format_distribution = dict(Counter(item.get('format', 'Unknown') for item in items))

        # Calculate modality distribution and content match
        modality_distribution = dict(Counter(item.get('modality', 'Text') for item in items))

        multimodal_items = [
            item for item in items
            if item.get('modality', 'Text').lower() in ['multimodal', 'visual', 'mixed']
        ]
        multimodal_with_images = sum(
            1 for item in multimodal_items
            if self._check_images_exist(item, dataset_name=dataset_name)
        )
        modality_content_match = (
            multimodal_with_images / len(multimodal_items) if multimodal_items else 1.0
        )

        return ValidationReport(
            dataset_name=dataset_name,
            total_items=total_items,
            passed_items=passed_items,
            failed_items=failed_items,
            warnings=warning_count,
            issues=all_issues,
            fields_completeness=fields_completeness,
            language_distribution=language_distribution,
            language_purity=language_purity,
            format_distribution=format_distribution,
            modality_distribution=modality_distribution,
            modality_content_match=modality_content_match
        )

    def save_report(self, report: ValidationReport, output_path: Path) -> None:
        """Save validation report to a JSON file

        Args:
            report: Validation report
            output_path: Path to save the report
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    def print_report_summary(self, report: ValidationReport) -> None:
        """Print a summary of the validation report

        Args:
            report: Validation report to summarize
        """
        print(f"\n=== Validation Report: {report.dataset_name} ===")
        print(f"Total items: {report.total_items}")
        print(f"Passed: {report.passed_items} ({report.passed_items/report.total_items*100:.1f}%)")
        print(f"Failed: {report.failed_items} ({report.failed_items/report.total_items*100:.1f}%)")
        print(f"Warnings: {report.warnings}")

        print(f"\nField completeness:")
        for field, completeness in sorted(report.fields_completeness.items()):
            status = "✓" if completeness >= 0.95 else "✗"
            print(f"  [{status}] {field}: {completeness*100:.1f}%")

        print(f"\nLanguage purity: {report.language_purity*100:.1f}%")
        print(f"  Distribution: {report.language_distribution}")

        print(f"\nFormat distribution: {report.format_distribution}")
        print(f"Modality distribution: {report.modality_distribution}")
        print(f"Modality content match: {report.modality_content_match*100:.1f}%")

        # Show first few errors/warnings
        errors = [issue for issue in report.issues if issue.severity == 'error']
        if errors:
            print(f"\nFirst 5 errors:")
            for issue in errors[:5]:
                print(f"  - {issue.item_id}: {issue.message}")

        warnings = [issue for issue in report.issues if issue.severity == 'warning']
        if warnings and len(errors) < 5:
            print(f"\nFirst 3 warnings:")
            for issue in warnings[:3]:
                print(f"  - {issue.item_id}: {issue.message}")


def main():
    """Test the quality validator"""
    import yaml
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset_loader import DatasetLoader

    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    loader = DatasetLoader()
    validator = QualityValidator()

    # Test with CSBench
    csbench_split = config['datasets']['csbench']['splits'][0]
    print(f"Validating {csbench_split['name']}...")

    items, _ = loader.load_and_analyze_split(csbench_split)

    report = validator.validate_dataset(
        items=items,
        dataset_name=csbench_split['name'],
        expected_language=csbench_split.get('language'),
        expected_modality=csbench_split.get('modality')
    )

    validator.print_report_summary(report)


if __name__ == '__main__':
    main()
