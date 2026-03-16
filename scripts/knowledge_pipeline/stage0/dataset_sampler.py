"""
Step 1: Initial Random Sampling

This module handles loading datasets and performing initial random sampling
with configurable seed for reproducibility.

Dataset Sizes (from statistics):
- CSBench valid splits: 236 each (use entire dataset)
- CSBench test splits: 2183 each
- ToMBench: 2860 each
- MedXpertQA text: 2450
- MedXpertQA mm: 2000
- WeMath: 1740
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class SampleMetadata:
    """Metadata for a dataset sample"""
    dataset_name: str
    split_name: str
    original_size: int
    sample_size: int
    seed_used: int
    sampled: bool  # True if sampling was performed, False if entire dataset used
    iteration: int  # Which iteration of resampling (0 for first attempt)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetSampler:
    """Handles dataset loading and random sampling"""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the dataset sampler

        Args:
            project_root: Root directory of the project. If None, auto-detect from script location
        """
        if project_root is None:
            # Auto-detect project root (3 levels up from this file)
            project_root = Path(__file__).resolve().parents[3]

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

    def sample_dataset(
        self,
        items: List[Dict[str, Any]],
        sample_size: Optional[int] = None,
        seed: int = 42,
        min_size_for_sampling: Optional[int] = None
    ) -> tuple[List[Dict[str, Any]], bool]:
        """Perform random sampling on dataset items

        Args:
            items: List of dataset items
            sample_size: Number of items to sample. If None or >= len(items), use all items
            seed: Random seed for reproducibility
            min_size_for_sampling: Only sample if dataset size > this value

        Returns:
            Tuple of (sampled_items, was_sampled)
            was_sampled is True if sampling was performed, False if entire dataset used
        """
        dataset_size = len(items)

        # Check if we should skip sampling
        if sample_size is None or sample_size >= dataset_size:
            return items, False

        if min_size_for_sampling is not None and dataset_size < min_size_for_sampling:
            return items, False

        # Perform sampling
        random.seed(seed)
        sampled_items = random.sample(items, sample_size)

        return sampled_items, True

    def sample_split(
        self,
        split_config: Dict[str, Any],
        seed: int = 42,
        iteration: int = 0
    ) -> tuple[List[Dict[str, Any]], SampleMetadata]:
        """Sample a dataset split according to configuration

        Args:
            split_config: Configuration dict for the split
            seed: Random seed for reproducibility
            iteration: Resampling iteration number (0 for first attempt)

        Returns:
            Tuple of (sampled_items, metadata)
        """
        # Load the dataset
        dataset_path = split_config['path']
        items = self.load_dataset(dataset_path)

        # Get sampling parameters
        sample_size = split_config.get('sample_size')
        min_size_for_sampling = split_config.get('min_size_for_sampling')

        # Perform sampling
        sampled_items, was_sampled = self.sample_dataset(
            items=items,
            sample_size=sample_size,
            seed=seed,
            min_size_for_sampling=min_size_for_sampling
        )

        # Create metadata
        metadata = SampleMetadata(
            dataset_name=split_config.get('name', 'unknown'),
            split_name=split_config.get('name', 'unknown'),
            original_size=len(items),
            sample_size=len(sampled_items),
            seed_used=seed,
            sampled=was_sampled,
            iteration=iteration
        )

        return sampled_items, metadata

    def save_sample(
        self,
        items: List[Dict[str, Any]],
        output_path: Path
    ) -> None:
        """Save sampled items to a JSONL file

        Args:
            items: List of sampled items
            output_path: Path to save the sample
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def save_metadata(
        self,
        metadata: SampleMetadata,
        output_path: Path
    ) -> None:
        """Save sample metadata to a JSON file

        Args:
            metadata: Sample metadata
            output_path: Path to save the metadata
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, ensure_ascii=False, indent=2)


def main():
    """Test the dataset sampler"""
    import yaml

    # Load config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize sampler
    sampler = DatasetSampler()

    # Test with first CSBench split
    csbench_split = config['datasets']['csbench']['splits'][0]
    seed = config['sampling']['random_seed']

    print(f"Sampling {csbench_split['name']}...")
    items, metadata = sampler.sample_split(csbench_split, seed=seed)

    print(f"  Original size: {metadata.original_size}")
    print(f"  Sample size: {metadata.sample_size}")
    print(f"  Was sampled: {metadata.sampled}")
    print(f"  Seed used: {metadata.seed_used}")


if __name__ == '__main__':
    main()
