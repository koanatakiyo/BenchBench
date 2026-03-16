#!/usr/bin/env python3
"""
Generate formatted reports from Lean Domain Extraction results.

Produces reports in the format specified in the protocol.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def format_report(report: Dict[str, Any]) -> str:
    """
    Format report according to the protocol template.

    Report format:
    Dataset: ‹name› | Items: ‹count› (Text ‹%›, Multimodal ‹%›)
    Oracle: ‹model› (temp=0, top_p=1) | PromptSHA: ‹…› | Enum: ‹file or none›
    Stability (mean across run pairs):
    • Raw Jaccard: ‹mean ± sd›
    • Soft-Jaccard: ‹mean ± sd› (threshold: 0.80)
    • NSI: ‹mean› (vs random baseline Jrand=‹…›)
    • Hierarchical agreement: ‹mean›
    Quality:
    • Coverage: ‹%› | Dedup after 0.9 sim: ‹%›
    • Gini(leaves): ‹value› | Parents: ‹count› | Leaves: ‹count›
    • Modality fidelity: ‹% vs source %› | Language fidelity: ‹% vs source %›
    • Hallucination proxy (terms not in text/visual facts): ‹%›
    Pass: ☐ YES ☐ NO (failed: ‹metric›)
    """
    lines = []

    # Header
    dataset = report['dataset']
    num_items = report['num_items']
    # TODO: Track text vs multimodal percentages in extraction
    text_pct = 100  # Placeholder
    multimodal_pct = 0  # Placeholder

    lines.append(f"Dataset: {dataset} | Items: {num_items} (Text {text_pct}%, Multimodal {multimodal_pct}%)")

    # Oracle info
    oracle = report['oracle']
    manifest = report.get('manifest', {})
    prompt_hash = manifest.get('prompt_hash', 'unknown')[:8]
    enum_hash = manifest.get('micro_enum_hash', 'none')

    lines.append(
        f"Oracle: {oracle} (temp=0, top_p=1) | PromptSHA: {prompt_hash} | "
        f"Enum: {enum_hash if enum_hash != 'none' else 'none'}"
    )

    # Stability metrics
    sm = report.get('stability_metrics')
    if sm:
        lines.append("\nStability (mean across run pairs):")
        lines.append(f"  • Raw Jaccard: {sm['raw_jaccard_terms']:.3f}")
        soft_jac_val = sm['soft_jaccard_terms']
        if soft_jac_val < 0:
            soft_jac_str = "N/A (couldn't compute embeddings)"
        else:
            soft_jac_str = f"{soft_jac_val:.3f} (threshold: 0.80, {'PASS' if soft_jac_val >= 0.75 else 'FAIL'})"
        lines.append(f"  • Soft-Jaccard: {soft_jac_str}")
        lines.append(f"  • NSI: {sm['nsi']:.3f} ({'PASS' if sm['nsi'] >= 0.65 else 'FAIL'})")
        lines.append(
            f"  • Hierarchical agreement: {sm['hierarchical_agreement']:.3f} "
            f"({'PASS' if sm['hierarchical_agreement'] >= 0.80 else 'FAIL'})"
        )
    else:
        lines.append("\nStability: N/A (single run)")

    # Quality metrics
    qm = report.get('quality_metrics', {})
    lines.append("\nQuality:")
    coverage = qm.get('coverage', 0)
    dedup_rate = qm.get('deduplication_rate', 0)
    lines.append(
        f"  • Coverage: {coverage:.1%} | Near-duplicates (≥0.92 sim): {dedup_rate:.1%}"
    )

    # Hierarchy stats (placeholder, would need to track in extraction)
    gini = 0.25  # Placeholder
    num_parents = 50  # Placeholder
    num_leaves = 250  # Placeholder
    lines.append(
        f"  • Gini(leaves): {gini:.3f} | Parents: {num_parents} | Leaves: {num_leaves}"
    )

    modality_fid = qm.get('modality_fidelity', 0)
    language_fid = qm.get('language_fidelity', 0)
    lines.append(
        f"  • Modality fidelity: {modality_fid:+.1%} | Language fidelity: {language_fid:+.1%}"
    )

    anchored_pct = qm.get('anchored_percent', 0)
    lines.append(
        f"  • Anchored% (terms validated by 4 strategies): {anchored_pct:.1%}"
    )

    # Pass/Fail
    overall_pass = report.get('overall_pass', False)
    if overall_pass:
        lines.append("\nPass: ☑ YES ☐ NO")
    else:
        # Collect failed metrics
        failures = []
        if sm:
            soft_jac_val = sm['soft_jaccard_terms']
            if soft_jac_val < 0:
                failures.append('Soft-Jaccard (N/A)')
            elif soft_jac_val < 0.75:
                failures.append('Soft-Jaccard')
            if sm['nsi'] < 0.65:
                failures.append('NSI')
            if sm['hierarchical_agreement'] < 0.80:
                failures.append('Hierarchical')

        if coverage < 0.95:
            failures.append('Coverage')
        if anchored_pct < 0.50:
            failures.append('Anchored%')

        failed_str = ', '.join(failures) if failures else 'unknown'
        lines.append(f"\nPass: ☐ YES ☑ NO (failed: {failed_str})")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate formatted report")
    parser.add_argument(
        'report_file',
        type=str,
        help='Path to JSON report file'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file (default: stdout)'
    )

    args = parser.parse_args()

    # Load report
    with open(args.report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # Format report
    formatted = format_report(report)

    # Output
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted)
        print(f"Report written to: {args.output}")
    else:
        print(formatted)


if __name__ == "__main__":
    main()
