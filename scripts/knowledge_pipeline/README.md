# Knowledge Pipeline v4 - Phase 0: Dataset Preparation

## Overview

Phase 0 implements a simplified dataset preparation pipeline that:
1. **Loads complete datasets** - No sampling, uses all available data
2. **Validates data quality** - Checks structure, completeness, and consistency
3. **Documents natural distributions** - Analyzes language, format, modality, and subdomain distributions
4. **Sets up infrastructure** - Prepares data and metadata for downstream extraction

## Rationale

**Why use entire datasets instead of sampling?**

All datasets are relatively small (236 to 2860 items), making sampling unnecessary:
- **CSBench** valid splits: 236 each
- **CSBench** test splits: 2183 each
- **ToMBench**: 2860 each (CN & EN)
- **MedXpertQA** text: 2450
- **MedXpertQA** mm: 2000
- **WeMath**: 1740

Using complete datasets ensures:
- ✅ Most representative distributions
- ✅ No sampling bias
- ✅ Simpler workflow
- ✅ Full data available for downstream processing

## Architecture

```
knowledge_pipeline/
├── config.yaml                  # Configuration for all datasets
├── dataset_loader.py           # Load datasets and analyze characteristics
├── quality_validator.py        # Validate data quality
├── phase0_orchestrator.py      # Main orchestrator script
└── README.md                   # This file
```

### Modules

#### 1. Dataset Loader (`dataset_loader.py`)

**Purpose**: Load complete datasets and analyze their natural characteristics.

**Key Features**:
- Loads JSONL datasets from configured paths
- Analyzes language, format, and modality distributions
- Extracts existing subdomain labels (for CSBench)
- Calculates purity metrics (language purity, multimodal ratio)
- Checks dataset size against expected values

**Output**: `DatasetCharacteristics` with:
- Total size and expected size match
- Language distribution and purity
- Format distribution (Multiple-choice, Assertion, Fill-in-the-blank, etc.)
- Modality distribution (Text, Multimodal)
- Subdomain counts (if existing labels available)
- Image presence flags

#### 2. Quality Validator (`quality_validator.py`)

**Purpose**: Validate data quality and consistency.

**Validation Checks**:
- **Required fields**: id, question, answer, language, format, modality
- **Language consistency**: Match expected language for split
- **Format consistency**: Match expected format
- **Modality content match**: Multimodal items actually have images
- **Answer format**: Non-empty, proper format
- **Question quality**: Reasonable length
- **Options completeness**: Multiple-choice questions have adequate options

**Output**: `ValidationReport` with:
- Pass/fail counts
- Field completeness percentages
- Language purity metrics
- Detailed issue log (errors and warnings)

#### 3. Phase 0 Orchestrator (`phase0_orchestrator.py`)

**Purpose**: Coordinate the complete Phase 0 pipeline.

**Workflow**:
1. Load and analyze each dataset split
2. Validate data quality
3. Save characteristics and validation reports
4. Save dataset copy for downstream processing
5. Generate summary report

## Usage

### Run on all datasets

```bash
python phase0_orchestrator.py --datasets all
```

### Run on specific datasets

```bash
# Single dataset
python phase0_orchestrator.py --datasets csbench

# Multiple datasets
python phase0_orchestrator.py --datasets csbench tombench medxpertqa
```

### Run on specific splits

```bash
# Process only valid splits
python phase0_orchestrator.py --datasets csbench --splits valid

# Process specific splits
python phase0_orchestrator.py --datasets all --splits cn_valid en_test
```

## Outputs

All outputs are saved to `outputs/phase0_datasets/`:

```
outputs/phase0_datasets/
├── phase0_summary.json                    # Overall summary
├── csbench_cn_valid/
│   ├── characteristics.json              # Dataset characteristics
│   ├── validation_report.json            # Quality validation report
│   └── dataset.jsonl                     # Dataset copy
├── csbench_en_valid/
│   ├── characteristics.json
│   ├── validation_report.json
│   └── dataset.jsonl
└── ... (other splits)
```

### Output Formats

**characteristics.json**:
```json
{
  "dataset_name": "csbench_cn_valid",
  "total_size": 236,
  "expected_size": 236,
  "size_match": true,
  "languages": {"Chinese": 236},
  "primary_language": "Chinese",
  "language_purity": 1.0,
  "formats": {"Multiple-choice": 145, "Assertion": 49, ...},
  "modalities": {"Text": 236},
  "subdomain_count": 24,
  "has_existing_labels": true
}
```

**validation_report.json**:
```json
{
  "dataset_name": "csbench_cn_valid",
  "total_items": 236,
  "passed_items": 236,
  "failed_items": 0,
  "warnings": 5,
  "fields_completeness": {
    "id": 1.0,
    "question": 1.0,
    "answer": 1.0,
    ...
  },
  "language_purity": 1.0,
  "issues": [...]
}
```

## Configuration

Edit `config.yaml` to add new datasets or modify settings:

```yaml
datasets:
  csbench:
    splits:
      - name: "csbench_cn_valid"
        path: "data/input_data/csbench/csbench_CN_valid_input.jsonl"
        language: "Chinese"
        modality: "Text"
        expected_size: 236
        has_existing_labels: true
        existing_subdomain_field: "background.SubDomain"
```

### Key Configuration Options

- `expected_size`: Expected number of items (for validation)
- `language`: Expected language for consistency checks
- `modality`: Expected modality (Text, Multimodal)
- `has_existing_labels`: Whether dataset has pre-labeled subdomains
- `existing_subdomain_field`: Path to subdomain field in JSON (e.g., "background.SubDomain")

## Dataset Statistics

| Dataset | Split | Size | Language | Modality | Subdomains |
|---------|-------|------|----------|----------|------------|
| CSBench | CN valid | 236 | Chinese | Text | 24 |
| CSBench | CN test | 2183 | Chinese | Text | 24 |
| CSBench | EN valid | 236 | English | Text | 24 |
| CSBench | EN test | 2183 | English | Text | 24 |
| CSBench | FR valid | 236 | French | Text | 24 |
| CSBench | FR test | 2183 | French | Text | 24 |
| CSBench | DE valid | 236 | German | Text | 24 |
| CSBench | DE test | 2183 | German | Text | 24 |
| MedXpertQA | MM | 2000 | English | Multimodal | TBD |
| MedXpertQA | Text | 2450 | English | Text | TBD |
| ToMBench | CN | 2860 | Chinese | Text | TBD |
| ToMBench | EN | 2860 | English | Text | TBD |
| WeMath | - | 1740 | Chinese | Multimodal | TBD |

**Total**: 13 splits, **23,016 items**

## Quality Metrics

Phase 0 tracks the following quality metrics:

1. **Completeness**:
   - All required fields present
   - Field completeness percentages

2. **Consistency**:
   - Language purity (% of items in expected language)
   - Format consistency
   - Modality content match (multimodal items have images)

3. **Correctness**:
   - Valid answer formats
   - Reasonable question lengths
   - Proper option counts for multiple-choice

## Next Steps

After completing Phase 0:

1. **Phase 1**: Knowledge Extraction
   - Use validated datasets from `outputs/phase0_datasets/*/dataset.jsonl`
   - Extract detailed knowledge concepts and hierarchies
   - For CSBench, can optionally use existing subdomain labels as ground truth

2. **Phase 2**: Balance Validation
   - Calculate Gini coefficients on extracted subdomains
   - Analyze domain/subdomain distributions
   - Identify underrepresented areas

3. **Phase 3**: Knowledge Corpus Compilation
   - Compile extracted knowledge into structured corpus
   - Link concepts across datasets
   - Generate knowledge graph

## Testing

Test individual modules:

```bash
# Test dataset loader
python dataset_loader.py

# Test quality validator
python quality_validator.py

# Test orchestrator on small subset
python phase0_orchestrator.py --datasets csbench --splits cn_valid
```

## Troubleshooting

### Dataset not found

**Error**: `FileNotFoundError: Dataset file not found`

**Solution**: Check that paths in `config.yaml` are correct relative to project root.

### Expected size mismatch

**Warning**: `Size: 2183 items (expected: 2000) [✗]`

**Solution**: Update `expected_size` in config.yaml or investigate if dataset was modified.

### Low language purity

**Warning**: `Language purity: 85.0%`

**Solution**: Review dataset for mixed-language items. May need cleaning if this is a monolingual split.

## Development

### Adding a new dataset

1. Add dataset configuration to `config.yaml`:
```yaml
datasets:
  newdataset:
    splits:
      - name: "newdataset_split1"
        path: "data/input_data/newdataset/split1.jsonl"
        language: "English"
        modality: "Text"
        expected_size: 1000
```

2. Run Phase 0:
```bash
python phase0_orchestrator.py --datasets newdataset
```

### Extending validation rules

Edit `quality_validator.py` and add custom checks to the `validate_item()` method.

## Version History

- **v4.0.0** (2025-01-30): Initial Phase 0 implementation
  - Full dataset loading (no sampling)
  - Quality validation
  - Natural distribution analysis
  - Infrastructure setup

## License

Part of the BenchBench project.
