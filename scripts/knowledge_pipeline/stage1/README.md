# Stage 1: Lean Domain Extraction

**Stage 1** extracts domain knowledge from benchmark datasets using a simplified, deterministic approach with strong stability and quality guarantees.

## Overview

The Lean Domain Card Protocol extracts structured domain metadata from benchmark items:
- **Subdomains** (up to 3): Specialized topic areas
- **Terms** (up to 12): Key concepts and terminology
- **Visual Facts** (up to 3): Objective visual observations (for multimodal datasets)
- **Native Glosses**: Language-specific term translations (for multilingual datasets)

### Key Principles

1. **Single Deterministic Oracle**: Uses temp=0, top_p=1 for consistency
2. **Strict JSON Format**: Enforces structured output with validation
3. **Micro-Enum Canonicalization**: Normalizes terms from corpus data (not external ontologies)
4. **Lightweight Consensus**: K=2 self-consistency only when needed
5. **Quality Over Stability**: Meet quality floors, don't chase perfect metrics

---

## Quick Start

### Run a Single Dataset

```bash
python run_lean_extraction.py \
    --dataset medxpertqa_mm \
    --mode enum \
    --oracle gpt-4o-mini \
    --visual-oracle gemini-2.5-flash
```

### Run All Stage 1 Datasets

```bash
# Text-only datasets
python run_lean_extraction.py --dataset csbench_en_test --mode enum --oracle gpt-4o-mini
python run_lean_extraction.py --dataset csbench_cn_test --mode enum --oracle gpt-4o-mini
python run_lean_extraction.py --dataset csbench_fr_test --mode enum --oracle gpt-4o-mini
python run_lean_extraction.py --dataset csbench_de_test --mode enum --oracle gpt-4o-mini
python run_lean_extraction.py --dataset tombench_en --mode enum --oracle gpt-4o-mini
python run_lean_extraction.py --dataset tombench_cn --mode enum --oracle gpt-4o-mini
python run_lean_extraction.py --dataset medxpertqa_text --mode enum --oracle gpt-4o-mini

# Multimodal datasets (with vision)
python run_lean_extraction.py --dataset medxpertqa_mm --mode enum --oracle gpt-4o-mini --visual-oracle gpt-4o
python run_lean_extraction.py --dataset wemath --mode enum --oracle gpt-4o-mini --visual-oracle gpt-4o
```

---

## Datasets

| Dataset | Items | Language | Modality | Domain |
|---------|-------|----------|----------|--------|
| **csbench_en_test** | 2,183 | English | Text | Computer Science |
| **csbench_cn_test** | 2,183 | Chinese | Text | Computer Science |
| **csbench_fr_test** | 2,183 | French | Text | Computer Science |
| **csbench_de_test** | 2,183 | German | Text | Computer Science |
| **medxpertqa_text** | 2,450 | English | Text | Medical |
| **medxpertqa_mm** | 2,000 | English | Multimodal | Medical |
| **tombench_en** | 2,860 | English | Text | Psychology (ToM) |
| **tombench_cn** | 2,860 | Chinese | Text | Psychology (ToM) |
| **wemath** | 1,740 | English | Multimodal | Mathematics |

**Total**: 20,742 items across 9 datasets

---

## Extraction Modes

### 1. Basic Mode (`--mode basic`)
- Raw oracle output without stabilizers
- No micro-enum canonicalization
- No K=2 consensus
- **Use for**: Testing oracle stability

### 2. Micro-Enum Mode (`--mode enum`) ✅ **Recommended**
- Applies micro-enum canonicalization
- Normalizes terms to corpus vocabulary
- No K=2 consensus (faster)
- **Use for**: Production runs

### 3. Consistency Mode (`--mode consistency`)
- Micro-enum + K=2 self-consistency
- Runs extraction twice, keeps intersection
- Higher stability, slower
- **Use for**: When enum mode fails floors

### 4. Full Protocol Mode (`--mode full`)
- Automatic escalation: basic → enum → consistency
- Stops when quality/stability floors met
- **Use for**: Adaptive extraction

---

## Quality Metrics & Floors

### Quality Metrics
- **Coverage** ≥ 0.95: % items with valid extraction
- **Modality Fidelity** ±0.05: Match between source and extracted modality
- **Language Fidelity** ±0.05: Match between source and extracted language
- **Deduplication Rate** ≤ 0.10: % near-duplicate items
- **Anchored %** ≥ 0.50: % terms validated by 4 anchor strategies

### Stability Metrics (2 runs)
- **Soft Jaccard (terms)** ≥ 0.75: Semantic similarity between runs
- **NSI (Normalized Stability Index)** ≥ 0.65: Combined stability score
- **Hierarchical Agreement** ≥ 0.80: Parent-level subdomain agreement

---

## Vision Support (Multimodal Datasets)

For datasets with images (`medxpertqa_mm`, `wemath`):

1. **Image Detection**: Automatically detects `images`, `image`, or `image_path` fields
2. **Image Loading**: Resolves paths to `data/raw_data/{dataset}/images/`
3. **Vision API**: Uses vision-capable models (gpt-4o, gemini-2.5-flash)
4. **Visual Facts**: Extracts objective visual observations (no diagnosis/speculation)

**Supported Vision Models**:
- `gpt-4o` (recommended - most reliable)
- `gpt-4o-mini` (faster, cheaper)
- `gemini-2.5-flash` (may have safety filter issues with medical images)

---

## Output Files

### Results File (`{dataset}_results_{timestamp}.jsonl`)
One JSON line per item:
```json
{
  "item_id": "MM-123",
  "extraction": {
    "subdomains": ["clinical cardiology", "cardiovascular imaging"],
    "terms": ["echocardiography", "ejection fraction", "myocardial infarction"],
    "visual_facts": ["left ventricle appears dilated", "reduced wall motion"],
    "confidence": 0.95,
    "anchored_count": 3,
    "anchored_fraction": 1.0
  }
}
```

### Report File (`{dataset}_report_{timestamp}.json`)
Summary with:
- Quality metrics (coverage, fidelity, anchored %)
- Stability metrics (Jaccard, NSI, hierarchical agreement)
- Floor pass/fail status
- Manifest (reproducibility record)

---

## Files in Stage 1

### Core Extraction
- **`lean_domain_extraction.py`**: Main extraction engine (LeanDomainExtractor class)
- **`run_lean_extraction.py`**: CLI runner with argument parsing
- **`quality_validator.py`**: Quality metric computation

### Post-Processing
- **`build_domain_card.py`**: Builds final domain card from extraction results
- **`canonicalize_subdomains.py`**: Micro-enum canonicalization utilities
- **`reparse_raw_responses.py`**: Re-parse raw LLM responses (debug/recovery)

### Domain-Specific Tagging
- **`tag_cs_super_parents.py`**: Tag CS benchmark with super-categories
- **`tag_medxpert_super_axes.py`**: Tag medical items with specialty axes
- **`tag_tom_super_axes.py`**: Tag ToM items with theory categories
- **`tag_wemath_super_parents.py`**: Tag math items with topic hierarchy

### Taxonomies
- **`cs_super_parents.py`**: Computer science super-category mappings
- **`med_specialty_taxonomy.py`**: Medical specialty taxonomy
- **`tom_taxonomy.py`**: Theory-of-Mind concept catalog
- **`wemath_super_parents.py`**: Math topic hierarchy

---

## Troubleshooting

### Visual Facts Not Extracted
- Check dataset has `images` field: `jq '.images' dataset.jsonl | head`
- Verify images exist: `ls data/raw_data/{dataset}/images/`
- Use `gpt-4o` instead of Gemini for medical images
- Check spot-check output for visual facts display

### Low Stability (Jaccard < 0.75)
- Try `--mode consistency` for K=2 self-consistency
- Check if oracle is deterministic (temp=0, top_p=1)
- Review spot-check for inconsistencies

### Low Coverage (< 0.95)
- Check for malformed JSON in raw responses
- Review invalid extractions in results file (`is_valid: false`)
- Inspect LLM error logs

### Safety Filter Blocking (Gemini)
- Switch to OpenAI vision: `--visual-oracle gpt-4o`
- Or ensure Gemini safety settings are disabled (already configured)

---

## Advanced Usage

### Custom Oracle Configuration

Edit `llm_manager.py` config or set environment variables:
```bash
export OPENAI_MODEL="gpt-4o"
export GEMINI_MODEL="gemini-2.5-flash"
```

### Sampling for Testing

```bash
# Run on 100 items only
python run_lean_extraction.py \
    --dataset csbench_en_test \
    --num-items 100 \
    --mode enum
```

### Debugging Single Items

```python
from lean_domain_extraction import LeanDomainExtractor

extractor = LeanDomainExtractor(oracle_name="gpt-4o-mini")
extractor.dataset_name = "medxpertqa_mm"

result = await extractor._extract_once(
    item=your_item,
    has_image=True,
    is_multilingual=False
)

print(result.visual_facts)
```

---

## Quality Checklist

Before considering Stage 1 complete for a dataset:

- [ ] **Coverage** ≥ 95%
- [ ] **Modality fidelity** within ±5%
- [ ] **Language fidelity** within ±5%
- [ ] **Deduplication rate** ≤ 10%
- [ ] **Anchored %** ≥ 50%
- [ ] **Soft Jaccard (terms)** ≥ 0.75
- [ ] **NSI** ≥ 0.65
- [ ] **Hierarchical agreement** ≥ 0.80
- [ ] **Visual facts** present for 100% of multimodal items
- [ ] **Spot-check** shows reasonable extractions

---

## Next Steps

After Stage 1 completion:
1. Review domain cards in `outputs/lean_domain_extraction/`
2. Validate hierarchical structure
3. Proceed to **Stage 2**: Question generation and difficulty rating

---

## References

- **Main Documentation**: `README_lean_extraction.md`
- **Stability Metrics**: `STABILITY_METRICS_FIX.md`
- **Behavior Examples**: `BEHAVIOR_EXAMPLES.md`

---

**Last Updated**: 2025-11-28
**Version**: 1.0 with Vision Support
