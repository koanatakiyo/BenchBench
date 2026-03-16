"""
Step 1: Load and Validate Complete Datasets

This module handles loading complete datasets and validating their structure.
No sampling is performed - we use entire datasets given their small sizes.

Dataset Sizes:
- CSBench valid splits: 236 each
- CSBench test splits: 2183 each
- ToMBench: 2860 each
- MedXpertQA text: 2450
- MedXpertQA mm: 2000
- WeMath: 1740
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from collections import Counter


@dataclass
class DatasetCharacteristics:
    """Natural characteristics of a complete dataset"""
    dataset_name: str
    split_name: str
    total_size: int
    expected_size: Optional[int]
    size_match: bool

    # Language distribution
    languages: Dict[str, int]
    primary_language: str
    language_purity: float

    # Format distribution
    formats: Dict[str, int]
    primary_format: str

    # Modality distribution
    modalities: Dict[str, int]
    multimodal_ratio: float

    # Domain/subdomain if available (from existing labels)
    domains: Optional[Dict[str, int]] = None
    subdomains: Optional[Dict[str, int]] = None
    subdomain_count: Optional[int] = None

    # Additional metadata
    has_existing_labels: bool = False
    has_images: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetLoader:
    """Handles dataset loading and characteristic analysis"""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the dataset loader

        Args:
            project_root: Root directory of the project. If None, auto-detect from script location
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        self.project_root = Path(project_root)

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load a JSONL dataset file

        Args:
            dataset_path: Relative or absolute path to the dataset file

        Returns:
            List of dataset items (dicts)

        Raises:
            FileNotFoundError: If dataset file doesn't exist
        """
        # Convert to absolute path if relative
        if not Path(dataset_path).is_absolute():
            dataset_path = self.project_root / dataset_path
        else:
            dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        items = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        return items

    def analyze_characteristics(
        self,
        items: List[Dict[str, Any]],
        split_config: Dict[str, Any]
    ) -> DatasetCharacteristics:
        """Analyze natural characteristics of the dataset

        Args:
            items: List of dataset items
            split_config: Configuration dict for the split

        Returns:
            DatasetCharacteristics object
        """
        total_size = len(items)
        expected_size = split_config.get('expected_size')

        # Count languages
        languages = Counter(item.get('language', 'Unknown') for item in items)
        primary_language = languages.most_common(1)[0][0] if languages else 'Unknown'
        language_purity = languages[primary_language] / total_size if total_size > 0 else 0.0

        # Count formats
        formats = Counter(item.get('format', 'Unknown') for item in items)
        primary_format = formats.most_common(1)[0][0] if formats else 'Unknown'

        # Count modalities
        modalities = Counter(item.get('modality', 'Text') for item in items)
        multimodal_count = sum(
            count for mod, count in modalities.items()
            if mod.lower() in ['multimodal', 'visual', 'mixed']
        )
        multimodal_ratio = multimodal_count / total_size if total_size > 0 else 0.0

        # Check for existing domain/subdomain labels
        has_existing_labels = split_config.get('has_existing_labels', False)
        domains = None
        subdomains = None
        subdomain_count = None

        if has_existing_labels:
            # Extract from background field
            bg_field = split_config.get('existing_subdomain_field', 'background.SubDomain')
            if '.' in bg_field:
                # Nested field like "background.SubDomain"
                parts = bg_field.split('.')
                subdomains_list = []
                for item in items:
                    value = item
                    for part in parts:
                        value = value.get(part, {}) if isinstance(value, dict) else None
                        if value is None:
                            break
                    if value:
                        subdomains_list.append(str(value))

                subdomains = dict(Counter(subdomains_list))
                subdomain_count = len(subdomains)

                # Try to extract domain from background.Domain
                domains_list = [
                    str(item.get('background', {}).get('Domain', ''))
                    for item in items
                    if item.get('background', {}).get('Domain')
                ]
                domains = dict(Counter(domains_list))

        # Check if dataset has images
        has_images = any(
            item.get('images') or
            item.get('background', {}).get('image_path')
            for item in items
        )

        return DatasetCharacteristics(
            dataset_name=split_config.get('name', 'unknown'),
            split_name=split_config.get('name', 'unknown'),
            total_size=total_size,
            expected_size=expected_size,
            size_match=(total_size == expected_size) if expected_size else True,
            languages=dict(languages),
            primary_language=primary_language,
            language_purity=language_purity,
            formats=dict(formats),
            primary_format=primary_format,
            modalities=dict(modalities),
            multimodal_ratio=multimodal_ratio,
            domains=domains,
            subdomains=subdomains,
            subdomain_count=subdomain_count,
            has_existing_labels=has_existing_labels,
            has_images=has_images
        )

    def load_and_analyze_split(
        self,
        split_config: Dict[str, Any]
    ) -> tuple[List[Dict[str, Any]], DatasetCharacteristics]:
        """Load a dataset split and analyze its characteristics

        Args:
            split_config: Configuration dict for the split

        Returns:
            Tuple of (items, characteristics)
        """
        # Load the dataset
        dataset_path = split_config['path']
        items = self.load_dataset(dataset_path)

        # Strip image-specific fields if this is a text-only split
        self._strip_image_fields_if_text_only(items, split_config)

        # Analyze characteristics
        characteristics = self.analyze_characteristics(items, split_config)

        return items, characteristics

    def _strip_image_fields_if_text_only(
        self,
        items: List[Dict[str, Any]],
        split_config: Dict[str, Any]
    ) -> None:
        """
        Remove image-related fields for splits that are declared text-only.
        This prevents downstream pipelines from treating the dataset as multimodal.
        """
        modality = (split_config.get('modality') or '').lower()
        text_only_flag = split_config.get('text_only', False)
        if modality != 'text' and not text_only_flag:
            return

        image_keys = {'images', 'image', 'image_caption', 'image_path'}

        for item in items:
            # Drop top-level image metadata
            for key in image_keys:
                item.pop(key, None)

            # Drop image references inside context lists
            context = item.get('context')
            if isinstance(context, list):
                filtered_context = []
                for entry in context:
                    if isinstance(entry, dict):
                        entry_type = str(entry.get('type', '')).lower()
                        entry_category = str(entry.get('category', '')).lower()
                        if entry_type == 'image' or entry_category == 'image':
                            continue
                    filtered_context.append(entry)
                item['context'] = filtered_context

            # Remove image paths from background metadata
            background = item.get('background')
            if isinstance(background, dict):
                for key in image_keys:
                    background.pop(key, None)

    def save_characteristics(
        self,
        characteristics: DatasetCharacteristics,
        output_path: Path
    ) -> None:
        """Save dataset characteristics to a JSON file

        Args:
            characteristics: Dataset characteristics
            output_path: Path to save the characteristics
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(characteristics.to_dict(), f, ensure_ascii=False, indent=2)

    def print_characteristics_summary(self, characteristics: DatasetCharacteristics) -> None:
        """Print a summary of dataset characteristics

        Args:
            characteristics: Dataset characteristics to summarize
        """
        print(f"\n{characteristics.dataset_name}:")
        print(f"  Size: {characteristics.total_size} items", end="")
        if characteristics.expected_size:
            match_str = "✓" if characteristics.size_match else "✗"
            print(f" (expected: {characteristics.expected_size}) [{match_str}]")
        else:
            print()

        print(f"  Language: {characteristics.primary_language} ({characteristics.language_purity:.1%} purity)")
        print(f"  Format: {characteristics.primary_format}")
        print(f"  Modality: {', '.join(f'{k}: {v}' for k, v in characteristics.modalities.items())}")

        if characteristics.multimodal_ratio > 0:
            print(f"  Multimodal ratio: {characteristics.multimodal_ratio:.1%}")

        if characteristics.has_existing_labels and characteristics.subdomain_count:
            print(f"  Subdomains: {characteristics.subdomain_count} unique")
            # Show top 5 subdomains
            if characteristics.subdomains:
                top_subdomains = sorted(
                    characteristics.subdomains.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                for subdomain, count in top_subdomains:
                    pct = (count / characteristics.total_size) * 100
                    print(f"    - {subdomain}: {count} ({pct:.1f}%)")


def main():
    """Test the dataset loader"""
    import yaml

    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize loader
    loader = DatasetLoader()

    # Test with first CSBench split
    csbench_split = config['datasets']['csbench']['splits'][0]

    print(f"Loading and analyzing {csbench_split['name']}...")
    items, characteristics = loader.load_and_analyze_split(csbench_split)

    loader.print_characteristics_summary(characteristics)


if __name__ == '__main__':
    main()
