#!/usr/bin/env python3
"""
Example runner for Lean Domain Extraction.

Shows how to use the LeanDomainExtractor with different configurations.
"""

import asyncio
import argparse
from pathlib import Path
from typing import Optional
from lean_domain_extraction import LeanDomainExtractor


async def run_basic_extraction(
    dataset_name: str,
    num_items: int = 100,
    num_runs: int = 1,
    oracle_name: str = "gpt-4o-mini",
    visual_oracle_name: Optional[str] = None
):
    """
    Run basic extraction (no stabilizers).

    This is Stage-1a: Test raw oracle stability without any helpers.
    """
    print("\n" + "="*80)
    print("STAGE-1a: Basic Extraction (No Stabilizers)")
    print("="*80 + "\n")

    extractor = LeanDomainExtractor(
        oracle_name=oracle_name,
        visual_oracle_name=visual_oracle_name,
        embedding_model_name="intfloat/multilingual-e5-small"
    )

    report = await extractor.run_extraction(
        dataset_name=dataset_name,
        num_items=num_items,
        use_micro_enum=False,  # No enum yet
        use_consistency=False,  # No K=2 yet
        num_runs=num_runs
    )

    return report


async def run_with_micro_enum(
    dataset_name: str,
    num_items: int = 100,
    num_runs: int = 1,
    oracle_name: str = "gpt-4o-mini",
    visual_oracle_name: Optional[str] = None
):
    """
    Run extraction with micro-enum canonicalization.

    This is Stage-1b: Apply micro-enum to stabilize wording.
    """
    print("\n" + "="*80)
    print("STAGE-1b: Extraction with Micro-Enum")
    print("="*80 + "\n")

    extractor = LeanDomainExtractor(
        oracle_name=oracle_name,
        visual_oracle_name=visual_oracle_name,
        embedding_model_name="intfloat/multilingual-e5-small"
    )

    report = await extractor.run_extraction(
        dataset_name=dataset_name,
        num_items=num_items,
        use_micro_enum=True,  # Enable micro-enum
        use_consistency=False,  # No K=2 yet
        num_runs=num_runs
    )

    return report


async def run_with_consistency(
    dataset_name: str,
    num_items: int = 100,
    num_runs: int = 1,
    oracle_name: str = "gpt-4o-mini",
    visual_oracle_name: Optional[str] = None
):
    """
    Run extraction with K=2 self-consistency.

    This is Stage-1c: Apply lightweight consensus if micro-enum alone doesn't meet floors.
    """
    print("\n" + "="*80)
    print("STAGE-1c: Extraction with K=2 Self-Consistency")
    print("="*80 + "\n")

    extractor = LeanDomainExtractor(
        oracle_name=oracle_name,
        visual_oracle_name=visual_oracle_name,
        embedding_model_name="intfloat/multilingual-e5-small"
    )

    report = await extractor.run_extraction(
        dataset_name=dataset_name,
        num_items=num_items,
        use_micro_enum=True,
        use_consistency=True,  # Enable K=2 consensus
        num_runs=num_runs
    )

    return report


async def run_full_protocol(
    dataset_name: str,
    num_items: int = 100,
    num_runs: int = 1,
    oracle_name: str = "gpt-4o-mini",
    visual_oracle_name: Optional[str] = None
):
    """
    Run the full protocol with automatic stabilizer escalation.

    Escalation order:
    1. Try basic extraction
    2. If fails, add micro-enum
    3. If still fails, add K=2 consistency
    4. Stop when floors are met (don't chase perfection)
    """
    print("\n" + "="*80)
    print("FULL LEAN DOMAIN CARD PROTOCOL")
    print("="*80 + "\n")

    # Step 1: Basic extraction
    print("\n[Step 1] Trying basic extraction...")
    extractor = LeanDomainExtractor(
        oracle_name=oracle_name,
        visual_oracle_name=visual_oracle_name,
        embedding_model_name="intfloat/multilingual-e5-small"
    )

    report = await extractor.run_extraction(
        dataset_name=dataset_name,
        num_items=num_items,
        use_micro_enum=False,
        use_consistency=False,
        num_runs=num_runs
    )

    if report['overall_pass']:
        print("\n✓ Basic extraction passed all floors!")
        return report

    # Step 2: Add micro-enum
    print("\n[Step 2] Basic extraction failed. Adding micro-enum...")
    report = await extractor.run_extraction(
        dataset_name=dataset_name,
        num_items=num_items,
        use_micro_enum=True,
        use_consistency=False,
        num_runs=num_runs
    )

    if report['overall_pass']:
        print("\n✓ Extraction with micro-enum passed all floors!")
        return report

    # Step 3: Add K=2 consistency
    print("\n[Step 3] Still failing. Adding K=2 self-consistency...")
    report = await extractor.run_extraction(
        dataset_name=dataset_name,
        num_items=num_items,
        use_micro_enum=True,
        use_consistency=True,
        num_runs=num_runs
    )

    if report['overall_pass']:
        print("\n✓ Full protocol with K=2 passed all floors!")
    else:
        print("\n⚠ Even with all stabilizers, some floors not met.")
        print("Consider reviewing prompts or oracle choice.")

    return report


async def main():
    parser = argparse.ArgumentParser(description="Run Lean Domain Extraction")
    parser.add_argument(
        '--dataset',
        type=str,
        default='csbench_en_test',
        choices=[
            'csbench_en_test', 'csbench_cn_test', 'csbench_fr_test', 'csbench_de_test',
            'medxpertqa_text', 'medxpertqa_mm',
            'tombench_cn', 'tombench_en',
            'wemath'
        ],
        help='Dataset to extract from'
    )
    parser.add_argument(
        '--num-items',
        type=int,
        default=100,
        help='Number of items to sample (default: 100)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['basic', 'enum', 'consistency', 'full'],
        help='Extraction mode (default: full)'
    )
    parser.add_argument(
        '--oracle',
        type=str,
        default='gpt-4o-mini',
        help='Oracle model to use for text items (default: gpt-4o-mini). Examples: gpt-4o-mini, gpt-4o, gemini-2.0-flash-exp, claude-3-5-sonnet-20241022'
    )
    parser.add_argument(
        '--visual-oracle',
        type=str,
        default=None,
        help='Oracle model to use for visual/multimodal items (default: same as --oracle). Example: gemini-2.0-flash-exp'
    )
    parser.add_argument(
        '--num-runs',
        type=int,
        default=1,
        help='Number of runs for stability measurement (default: 1)'
    )

    args = parser.parse_args()

    # Run extraction based on mode
    if args.mode == 'basic':
        report = await run_basic_extraction(args.dataset, args.num_items, args.num_runs, args.oracle, args.visual_oracle)
    elif args.mode == 'enum':
        report = await run_with_micro_enum(args.dataset, args.num_items, args.num_runs, args.oracle, args.visual_oracle)
    elif args.mode == 'consistency':
        report = await run_with_consistency(args.dataset, args.num_items, args.num_runs, args.oracle, args.visual_oracle)
    else:  # full
        report = await run_full_protocol(args.dataset, args.num_items, args.num_runs, args.oracle, args.visual_oracle)

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Dataset: {report['dataset']}")
    print(f"Items: {report['num_items']}")
    print(f"Oracle (text): {report['oracle']}")
    if report.get('visual_oracle'):
        print(f"Oracle (visual): {report['visual_oracle']}")
    print(f"Overall Pass: {'✓ YES' if report['overall_pass'] else '✗ NO'}")

    if report['stability_metrics']:
        sm = report['stability_metrics']
        print(f"\nStability Metrics:")
        print(f"  Raw Jaccard (terms): {sm['raw_jaccard_terms']:.3f}")
        soft_jac_str = "N/A" if sm['soft_jaccard_terms'] < 0 else f"{sm['soft_jaccard_terms']:.3f}"
        print(f"  Soft Jaccard (terms): {soft_jac_str}")
        print(f"  NSI: {sm['nsi']:.3f}")
        print(f"  Hierarchical agreement: {sm['hierarchical_agreement']:.3f}")

    qm = report['quality_metrics']
    print(f"\nQuality Metrics:")
    print(f"  Coverage: {qm['coverage']:.3f}")
    print(f"  Anchored%: {qm['anchored_percent']:.3f}")
    print(f"  Near-duplicate%: {qm['deduplication_rate']:.1%} (lower is better)")

    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
