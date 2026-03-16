# BenchBench: Benchmarking Automated Benchmark Generation

A multi-stage pipeline for generating and evaluating high-quality benchmark datasets using LLMs.
The pipeline takes existing academic benchmarks as input and produces structured domain cards,
generated questions, and multi-model answer panels.

## Pipeline Overview

```
Stage 1 → Domain extraction (subdomains, terms, visual facts)
Stage 2 → Question generation (300 questions per dataset)
Stage 3 → Answerer panel evaluation (multiple LLMs score each question)
```

### Supported Datasets

| Dataset | Language | Modality | Items |
|---------|----------|----------|-------|
| CSBench (EN/CN/FR/DE) | Multi | Text | ~2,183 each |
| MedXpertQA Text | EN | Text | 2,450 |
| MedXpertQA MM | EN | Multimodal | 2,000 |
| ToMBench (EN/CN) | Multi | Text | 2,860 each |
| WeMath | EN | Multimodal | 1,740 |

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd benchbench
pip install -r requirements.txt
```

### 2. Configure LLM providers

```bash
cp config/llm.yaml.example config/llm.yaml
# Edit config/llm.yaml and fill in your API keys
```

You only need to configure the providers you plan to use. Supported providers:
OpenAI, Gemini, Anthropic (Claude), Grok, DeepSeek, Qwen (DashScope), Doubao, Llama (OpenRouter), Ollama (local), vLLM (local).

### 3. Configure Google Search (optional, for Stage 1 web search)

```bash
cp config/google_api_keys.yaml.example config/google_api_keys.yaml
# Edit config/google_api_keys.yaml and fill in your credentials
```

### 4. Prepare input data

Place your input benchmark datasets under `data/input_data/<dataset_name>/`.
See `scripts/knowledge_pipeline/stage0/` for dataset loading utilities.

---

## Running the Pipeline

### Stage 1: Domain Extraction

Extracts domain metadata (subdomains, terms, visual facts) from each benchmark item.

```bash
cd scripts/knowledge_pipeline/stage1

python run_lean_extraction.py \
    --dataset medxpertqa_mm \
    --mode enum \
    --llm-provider gemini \
    --output-dir ../../../outputs/stage1_domains/medxpertqa_mm
```



Generates 300 benchmark questions per dataset (225 standard + 75 adversarial).

```bash
cd scripts/knowledge_pipeline/stage2

python run_stage2.py \
    --dataset csbench_en \
    --llm-provider openai \
    --output-dir ../../../outputs/stage2_questions/csbench_en
```


### Stage 3: Answerer Panel Evaluation

Runs multiple eval models over Stage 2 questions and scores correctness.

```bash
# 1. Post-hoc cleanup (run once per designer, before answering)
cd scripts/knowledge_pipeline/stage3

python posthoc_clean_topup.py \
    --dataset csbench_en \
    --provider-dir ../../../outputs/stage2_questions/csbench_en/openai_gpt5mini \
    --designer-model gpt-5-mini \
    --llm-provider openai

# 2. Run answerer panel
python run_stage3.py \
    --config ../../../config/stage3.yaml \
    --dataset csbench_en
```


---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/llm.yaml` | LLM provider API keys and model settings (**gitignored**) |
| `config/llm.yaml.example` | Template for `llm.yaml` |
| `config/google_api_keys.yaml` | Google Search credentials (**gitignored**) |
| `config/google_api_keys.yaml.example` | Template for `google_api_keys.yaml` |
| `config/stage3.yaml` | Stage 3 dataset/designer/output configuration |
| `config/prompts/` | Prompt templates used across stages |

---

## Project Structure

```
benchbench/
├── config/                        # Configuration (API keys gitignored)
├── data/                          # Input datasets and cached data (gitignored)
├── outputs/                       # Pipeline outputs (gitignored)
├── scripts/
│   ├── llm_manager.py             # Unified LLM interface (all providers)
│   └── knowledge_pipeline/
│       ├── stage0/                # Dataset preparation utilities
│       ├── stage1/                # Domain extraction
│       ├── stage2/                # Question generation
│       ├── stage3/                # Answerer panel evaluation
│       └── shared/                # Shared utilities
└── requirements.txt
```
