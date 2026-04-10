# Stage 2: Question Generation

Stage 2 generates 300 high-quality benchmark questions per dataset using LLMs, based on the domain cards produced in Stage 1.

## Overview

**Goal**: Design discriminative, well-balanced questions that:
- Cover the domain comprehensively
- Test understanding at multiple difficulty levels
- Include both standard and adversarial questions
- Support multiple formats (MCQ, open-ended, structured)
- Enable multimodal question generation

**Key Innovation**: Explicit control over:
- Domain coverage (quotas per super_parent)
- Difficulty distribution (L1-L5)
- Design type (standard vs adversarial)
- Question format diversity
- Multimodal content

## Architecture

```
stage2/
├── question_schema.py       # Unified question schema
├── coverage_config.py       # Domain quotas & distributions
├── prompt_builder.py        # Structured prompt generation
├── question_generator.py    # LLM-based generation
├── validate_questions.py    # Quality validation
├── run_stage2.py           # Main orchestrator
└── README.md               # This file
```

## Prompting Strategy (New in v4)

- **Single compact system prompt** defines the role, JSON schema, difficulty bands, question types, and adversarial guidance. It is parameterized by dataset name and designer model so every call shares the exact same contract.
- **Per-call user prompts** are assembled dynamically for each super_parent batch (30–60 questions). Each prompt includes:
  - Allowed super_parent(s) and subdomains
  - Standard vs adversarial quotas
  - Difficulty and question-type targets for the batch
  - ID prefix pattern and numbering rules (`gpt5_csbench_en_<super>_bXX_qYYY`)
  - Focused topic hints derived from the domain card (ontology + glossary + sample items)
- The orchestrator splits the 300-question requirement into multiple batches, validates distributions, and issues targeted “top-up” batches if any quotas/difficulties/formats are underfilled (e.g., “need 3 L5 structured adversarial ComputerNetwork questions”).

## Question Design Philosophy

### Standard Questions (225 per dataset)
- **Goal**: Broad, fair coverage of the domain
- **Approach**: Clear, discriminative, non-trick questions
- **Constraints**: Solvable by strong LLMs with some error rate

### Adversarial Questions (75 per dataset)
- **Goal**: Expose LLM blind spots and failure modes
- **Approach**: Longer reasoning chains, subtle edge cases, unusual formats
- **Constraints**: Still unambiguous and verifiable; just harder/trap-like

## Question Schema

Every generated question follows this unified schema:

```python
{
    # Identity & Metadata
    "id": "gpt5_csbench_en_q001",
    "designer_model": "gpt-5-mini",
    "source_dataset": "csbench_en",
    "super_parent": "CS.Core.OperatingSystem",
    "subdomain": "memory_management",

    # Design Classification
    "design_type": "standard",  # or "adversarial"
    "modality": "text",         # or "text+existing_image", "text+imagined_image"
    "language": "en",           # "en", "zh", "fr", "de"
    "question_type": "mcq_single",  # mcq_single, mcq_multi, open_ended, etc.

    # Question Content
    "question_stem": "What happens when a TLB miss occurs?",
    "options": ["...", "...", "...", "..."],  # [] for non-MCQ
    "answer": "B",  # or ["B", "C"] for multi-answer
    "answer_explanation": "On a TLB miss, the MMU...",

    # Difficulty & Design
    "declared_difficulty": "L3",  # L1-L5
    "estimated_time_sec": 45,
    "design_rationale": "Tests knowledge of TLB vs page fault",

    # Visual Information
    "uses_visual": false,
    "visual_instruction": null,

    # Additional Metadata
    "meta_notes": "Based on memory_management subdomain"
}
```

**Key Features**:
- `designer_model`: Enables self/family bias analysis
- `design_type`: Enables adversarial vs standard comparison
- `design_rationale` vs `answer_explanation`: Why designed vs why correct
- `modality` + `visual_instruction`: Supports multimodal generation

## Coverage Configuration

Each dataset has explicit coverage quotas:

### Example: CSBench English

| Super Parent | Standard | Adversarial |
|-------------|----------|-------------|
| CS.Core.DataStructuresAlgorithms | 60 | 20 |
| CS.Core.OperatingSystem | 50 | 15 |
| CS.Core.ComputerNetwork | 40 | 12 |
| CS.Core.ComputerOrganization | 30 | 10 |
| ... | ... | ... |
| **TOTAL** | **225** | **75** |

### Difficulty Distribution

Across all 300 questions:
- L1: 15% (~45 questions) - Very easy, factual recall
- L2: 25% (~75 questions) - Basic understanding
- L3: 30% (~90 questions) - Moderate reasoning
- L4: 20% (~60 questions) - Advanced concepts
- L5: 10% (~30 questions) - Very hard, edge cases

### Format Distribution

Example for CSBench:
- mcq_single: 60% (~180 questions)
- mcq_multi: 10% (~30 questions)
- open_ended: 20% (~60 questions)
- structured: 10% (~30 questions)

Format distributions vary by dataset (e.g., ToMBench includes "judgment" type).

## Usage

### Quick Start

Generate questions for a single dataset:

```bash
cd scripts/knowledge_pipeline/stage2
python run_stage2.py csbench_en
```

### Generate Multiple Datasets

```bash
python run_stage2.py csbench_en tombench_en medxpertqa_text
```

### Generate All Datasets

```bash
python run_stage2.py --all
```

### Advanced Options

```bash
# Use different LLM
python run_stage2.py csbench_en --model gemini

# Use local Ollama with deepseek-r1
python run_stage2.py csbench_en --model ollama

# Use DeepSeek's hosted API (set deepseek.api_key in config/llm.yaml)
python run_stage2.py csbench_en --model deepseek

# Override Ollama model/base URL per run
python run_stage2.py csbench_en --model ollama \
  --ollama-model deepseek-r1:8b \
  --ollama-base-url http://localhost:11435/v1

# Adjust temperature
python run_stage2.py csbench_en --temperature 0.9

# Skip validation
python run_stage2.py csbench_en --no-validate

# Custom output directory
python run_stage2.py csbench_en --output-dir /path/to/output

# Control batch size (30–60 recommended)
python run_stage2.py csbench_en --questions-per-call 50

# Override provider model name (e.g., gpt-4o vs gpt-4o-mini)
python run_stage2.py csbench_en --model openai --llm-model-name gpt-4o
# (designer_model automatically tracks --llm-model-name)

# Run with local Hugging Face model
python run_stage2.py csbench_en --model huggingface \
  --hf-model-id Qwen/Qwen2.5-14B-Instruct \
  --hf-device-map cuda:0 \
  --hf-dtype bfloat16
```

**Note**: For Ollama setup and usage, see [OLLAMA_SETUP.md](OLLAMA_SETUP.md). For DeepSeek hosted inference, populate the `deepseek` section in `config/llm.yaml` (or set `DEEPSEEK_API_KEY`) so Stage 2 can authenticate with `https://api.deepseek.com/v1`.

### Available Models

- `openai`: GPT-4, GPT-4-turbo (default)
- `gemini`: Gemini 2.5 Flash
- `anthropic`: Claude 3.5 Sonnet
- `deepseek`: DeepSeek Chat
- `ollama`: Local models via Ollama (e.g., deepseek-r1) - See [OLLAMA_SETUP.md](OLLAMA_SETUP.md)

## Output Structure

```
outputs/stage2_questions/
├── csbench_en/
│   ├── batches/                    # Per-batch prompts & raw JSONLs (30–60 questions each)
│   │   ├── os_b01_prompt.txt
│   │   ├── os_b01_questions.jsonl
│   │   └── ...
│   ├── questions_openai_gpt-4o-mini_gpt-5-mini.jsonl   # Final merged file (EXACT target count)
│   ├── validation_report_openai_gpt-4o-mini_gpt-5-mini.json
│   └── metadata.json
├── tombench_en/
│   └── ...
└── ...
```

### questions.jsonl

One question per line in JSON format:

```jsonl
{"id": "gpt5_csbench_en_os_b01_q001", "designer_model": "gpt-5-mini", ...}
{"id": "gpt5_csbench_en_os_b01_q002", "designer_model": "gpt-5-mini", ...}
...
```

### validation_report.json

Comprehensive validation report:

```json
{
  "dataset": "csbench_en",
  "total_questions": 300,
  "validation": {
    "valid": 298,
    "invalid": 2,
    "pass_rate": 99.3
  },
  "coverage": {
    "by_super_parent": {...}
  },
  "difficulty": {
    "distribution": {...}
  },
  "format": {
    "distribution": {...}
  },
  "errors": [...],
  "warnings": [...],
  "summary": {
    "overall_quality": "Excellent"
  }
}
```

## Validation

Stage 2 includes comprehensive validation:

### Automatic Checks

1. **Schema Validation**
   - All required fields present
   - Correct data types
   - Valid enum values

2. **Content Validation**
   - Question stem ≥ 10 characters
   - Answer explanation ≥ 20 characters
   - Design rationale ≥ 15 characters
   - MCQ has valid options and answer keys

3. **Coverage Analysis**
   - Compare actual vs expected domain coverage
   - Flag significant deviations (>5 questions)

4. **Difficulty Analysis**
   - Compare actual vs expected difficulty distribution
   - Flag large deviations (>10 questions)

5. **Format Analysis**
   - Compare actual vs expected format distribution

6. **Duplicate Detection**
   - Check for duplicate IDs
   - Warn about very similar question stems

### Manual Validation

Run standalone validation:

```bash
python validate_questions.py csbench_en \
    outputs/stage2_questions/csbench_en/questions.jsonl
```

## Prompt Engineering

Stage 2 uses highly structured prompts:

### System Prompt
- Establishes role as expert question designer
- Emphasizes originality, clarity, discrimination
- Sets expectations for standard vs adversarial

### User Prompt Structure

1. **Task Description**
   - 300 questions: 225 standard + 75 adversarial
   - Language instructions

2. **Domain Card Summary**
   - Ontology (super_parents + subdomains)
   - Key terms glossary
   - Example items from dataset

3. **Coverage Requirements**
   - Table of quotas per super_parent
   - Explicit instructions to follow

4. **Difficulty Distribution**
   - Target percentages for L1-L5
   - Definition of each level

5. **Format Distribution**
   - Target percentages per question_type
   - Instructions for each format

6. **Modality Instructions**
   - Text-only, text+image, or multimodal
   - Visual instruction guidelines

7. **JSON Schema**
   - Complete example question
   - Field descriptions
   - Output format requirements

8. **Final Checklist**
   - Verification items before submission

### Viewing Prompts

Prompts are saved automatically:

```bash
cat outputs/stage2_questions/csbench_en/prompts.txt
```

## Modality Support

### Text-Only Datasets

(CSBench, ToMBench text)

- All questions: `modality: "text"`
- `uses_visual: false`
- `visual_instruction: null`

### Multimodal Datasets

(MedXpertQA-MM, WeMath)

Two approaches:

1. **Existing Images** (`modality: "text+existing_image"`)
   - Reference images from the dataset
   - `visual_instruction`: Brief description of image

2. **Imagined Images** (`modality: "text+imagined_image"`)
   - Describe new images to be created later
   - `visual_instruction`: What the image should show
   - Stage 2.5 or 3 can realize these images

Target: 60-70% visual questions for multimodal datasets.

## Language Support

Stage 2 supports multilingual generation:

- **English** (`en`): CSBench-EN, ToMBench-EN, MedXpertQA
- **Chinese** (`zh`): CSBench-CN, ToMBench-CN, WeMath
- **French** (`fr`): CSBench-FR
- **German** (`de`): CSBench-DE

Language-specific instructions are automatically included in prompts.

## Design Rationale vs Answer Explanation

Two distinct fields capture different aspects:

### design_rationale
**Why this question was designed**
- What concept/skill it tests
- Common misconceptions it exposes
- Why it's discriminative

Example: *"Tests knowledge of TLB vs page fault; common confusion point between hardware and OS responsibilities."*

### answer_explanation
**Why this answer is correct**
- Domain reasoning
- Technical justification
- Why other options are wrong (for MCQ)

Example: *"On a TLB miss, the hardware MMU walks the page table in memory to find the translation; only if the page is not present does a page fault occur, involving the OS."*

## Integration with Stage 3

Stage 2 output is designed for seamless Stage 3 integration:

**Stage 2 produces**:
- Rich question metadata (designer, domain, difficulty)
- Design rationale (why designed this way)
- Visual instructions (for multimodal)

**Stage 3 will use**:
- `designer_model` for bias analysis
- `design_type` for adversarial testing
- `declared_difficulty` for IRT comparison
- `visual_instruction` for image generation
- `design_rationale` for quality filtering

## Best Practices

### Prompt Design

1. **Be Specific**: Explicit quotas and distributions work better than vague guidance
2. **Show Examples**: Include complete example questions in the prompt
3. **Emphasize Originality**: Repeatedly warn against copying examples
4. **Structured Output**: Demand JSON array, no markdown, no comments

### Generation

1. **Temperature**: 0.7-0.9 for creative question design
2. **Model Selection**:
   - GPT-4: Best overall quality
   - Gemini: Fastest, good quality
   - Claude: Excellent for nuanced adversarial questions
3. **Retry Logic**: Auto-retry on parse failures or count mismatches

### Validation

1. **Always Validate**: Don't skip validation in production
2. **Review Warnings**: Some warnings indicate design issues
3. **Spot Check**: Manually review 10-20 questions per dataset
4. **Coverage First**: Ensure good domain coverage before worrying about exact counts

### Iteration

If quality is poor:
1. Check prompts.txt to see what the LLM received
2. Review validation_report.json for specific issues
3. Adjust coverage_config.py quotas if needed
4. Try different temperature or model
5. Regenerate and validate

## Troubleshooting

### "JSON parse error"

**Cause**: LLM output includes markdown fences or comments

**Solution**: Parsing logic auto-strips common patterns, but if it persists:
- Check prompts emphasize "no markdown"
- Try lower temperature (0.7)
- Try different model

### "Expected 300 questions, got X"

**Cause**: LLM didn't follow count instructions

**Solution**: Auto-retries up to 3 times. If persists:
- Check if prompt is too long (may truncate)
- Reduce domain card summary length
- Try different model

### "Coverage deviation warnings"

**Cause**: LLM didn't perfectly follow quotas

**Solution**: Small deviations (±5) are acceptable. If large:
- Emphasize coverage table in prompt
- Reduce number of domains (consolidate small ones)
- Post-process to rebalance

### "Duplicate IDs"

**Cause**: ID generation pattern in prompt not clear

**Solution**: Check that questions have unique sequential IDs

### "Rate limit errors (Gemini)"

**Cause**: Hitting 1000 RPM or 10K RPD limits

**Solution**: Built-in rate limiting, but if persists:
- Reduce concurrent requests
- Add delays between datasets
- Use different model

## Performance

### Generation Time

Typical times per dataset (300 questions):
- **GPT-4**: ~3-5 minutes
- **Gemini 2.5 Flash**: ~1-2 minutes
- **Claude 3.5**: ~3-4 minutes
- **DeepSeek**: ~2-3 minutes

### Cost Estimates

Per dataset (approximate):
- **GPT-4**: $0.50-1.00 (depends on domain card size)
- **Gemini 2.5 Flash**: $0.10-0.20
- **Claude 3.5**: $0.40-0.80
- **DeepSeek**: $0.05-0.15
- **Ollama**: $0.00 (free, runs locally)

For all 9 datasets:
- **GPT-4**: ~$5-9
- **Gemini**: ~$1-2
- **Claude**: ~$4-7
- **DeepSeek**: ~$0.50-1.50
- **Ollama**: $0.00 (free, but requires GPU)

## Dataset-Specific Notes

### CSBench
- Heavy on DataStructures/Algorithms and OS
- Use structured formats (tables, code blocks) sparingly
- Adversarial: Focus on subtle differences between similar concepts

### ToMBench
- Narrative-based questions (stories, scenarios)
- Adversarial: Complex multi-agent belief tracking
- Format: Prefer "judgment" type for moral reasoning

### MedXpertQA
- Clinical case scenarios
- Adversarial: Rare presentations, atypical findings
- MM: Leverage diagnostic images, histopathology slides

### WeMath
- Visual geometry problems
- Adversarial: Multi-step spatial reasoning
- Most questions should reference diagrams

### Visual Ablations (Stage 2)
- New dataset variants to isolate visual priming effects:
  - `wemath_stage2_textonly` vs `wemath_stage2_visualprimed`
  - `medxpertqa_mm_stage2_textonly` vs `medxpertqa_mm_stage2_visualprimed`
- Text-only variants explicitly forbid mentioning images/figures; visual-primed variants allow imagined visuals with concise, high-level descriptions.

## Future Enhancements

Planned improvements:

1. **Stage 2.5: Visual Realization**
   - Generate actual images for "imagined_image" modality
   - Use DALL-E, Stable Diffusion, or diagram tools

2. **Question Clustering**
   - Group questions by semantic similarity
   - Ensure diversity within coverage bins

3. **Difficulty Calibration**
   - Pre-test with multiple LLMs
   - Adjust declared_difficulty based on actual performance

4. **Adversarial Templates**
   - Library of proven adversarial patterns
   - Inject systematically into generation

5. **Cross-Dataset Contamination Check**
   - Detect if generated questions too similar to Stage 0/1

## References

- Stage 1 documentation: [stage1/README.md](../stage1/README.md)
- Stage 3 documentation: [stage3/README.md](../stage3/README.md)
- Question schema: [question_schema.py](question_schema.py)
- Coverage config: [coverage_config.py](coverage_config.py)

## License

Part of the BenchBench project.
