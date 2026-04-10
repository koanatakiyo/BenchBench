## Stage 3 Toolkit

This directory now contains two complementary workflows:

1. `posthoc_clean_topup.py` – deduplicates Stage-2 questions, trims over-subscribed buckets, and (optionally) launches targeted top-ups to hit coverage goals.
2. `run_stage3.py` – runs the Stage-3 “answerer panel” for any dataset: each Stage-2 designer drop is answered by multiple eval models, scored, and logged.

The steps below assume you already have Stage-2 outputs under `outputs/stage2_questions/<dataset>/<designer>/questions_*_stage3_clean.jsonl`.

---

### 1. Post-hoc cleanup & top-ups

```
python posthoc_clean_topup.py \
  --dataset csbench_en \
  --provider-dir /abs/path/to/outputs/stage2_questions/csbench_en/openai_gpt5mini \
  --designer-model gpt-5-mini \
  --llm-provider openai \
  --llm-model-override gpt-5-mini \
  --temperature 0.6
```

Key outputs:

- `questions_*_stage3_clean.jsonl` – cleaned base for Stage-3 answering.
- `stage3_posthoc_report.json` – audit trail of removals, deficits, and top-ups.

Repeat per Stage-2 designer before moving to the answerer panel.

---

### 2. Stage-3 answerer panel

Configuration lives in `benchbench/config/stage3.yaml`. It enumerates:

- The Stage-2 designer drops to load (absolute paths to the cleaned JSONL files).
- The answerer models, their providers, request batch sizes, and decoding params (temperature/top_p/max tokens).
- Output root (defaults to `outputs/stage3_answers/<dataset>`).

Each (designer_model, answer_model) pair writes a JSONL matrix:

```
outputs/stage3_answers/<dataset>/
  <dataset>_stage3_answers_gpt5mini_vs_qwen3_next_30b.jsonl
  ...
```

Each row includes dataset metadata, question metadata, model output, parsed answer, local correctness (exact/numeric), answer latencies, and tracing info (`call_batch_id`, `call_index_in_batch`, etc.).

#### Prompt templates (hard-coded, token-light)

- `mcq_single`: “Choose exactly one correct option… respond with a single capital letter.”
- `mcq_multi`: “Return correct options as a JSON array of capital letters.”
- `open_ended`: “Give concise, direct answers; one sentence or short expression.”
- `structured`: “Follow the requested structure exactly; no extra commentary.”

#### Running the panel

```
python run_stage3.py \
  --config /home/yandan/clrproj/benchb/benchbench/config/stage3.yaml \
  --designer gpt5mini \
  --answerer qwen3_next_30b \
  --limit 8 \
  --dry-run
```

Or use the bash helper to run every configured designer × answerer pair:

```
./run_all_models.sh                             # default config
./run_all_models.sh "" --dry-run --skip-existing # forward extra args
./run_all_models.sh /custom/stage3.yaml          # custom config
```

Flags:

- `--designer/--answerer` (repeatable): restrict to subsets by key or display name.
- `--limit`: cap number of questions per designer for smoke tests.
- `--dry-run`: build prompts and plan, but skip API calls/writes.
- `--skip-existing`: avoids re-running pairs whose JSONL output already exists.
- `--log-level DEBUG`: inspect per-call traces.

Internally the runner:

1. Loads each designer’s Stage-3-clean file into memory (optionally truncated by `--limit`).
2. Chunks questions by `batch_size` per answerer, spinning concurrent API calls capped by `max_concurrent_requests`.
3. Parses outputs according to format-specific rules, scores them (exact/numeric), and emits JSONL rows.

> **Tip:** keep API keys in `config/llm.yaml`. The runner reuses `LLMManager`, so any provider listed there is usable here as long as it’s in the YAML config.

---

### 3. Verifying a run

1. **Dry-run** with `--limit` to ensure prompts look correct and output paths resolve.
2. **Live run** without `--dry-run`. Watch logs for API errors or malformed responses.
3. Inspect a sample JSONL file (e.g., first 3 rows) to confirm:
   - `model_output_raw` contains only the required minimal answer (single letter / JSON array / short text).
   - `parsed_answer` matches expectations (e.g., `["A","C"]`).
   - `is_correct` toggles correctly for known-easy items.
4. Optionally pipe the JSONL into downstream analytics / parquet conversion.

If you need to re-run a specific pair, delete its output JSONL and re-launch with `--designer ... --answerer ...`.

