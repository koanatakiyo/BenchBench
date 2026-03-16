#!/usr/bin/env python3
"""
Question Generator for Stage 2 - Generate questions using LLM
"""

import json
import asyncio
import sys
import re
from collections.abc import Mapping
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))
from llm_manager import LLMManager

# Local imports
from prompt_builder import build_system_prompt
from question_schema import Question


# Hugging Face models may be large, so keep a simple in-memory cache keyed by
# model + load parameters to avoid reloading per dataset run.
HF_MODEL_CACHE: Dict[Tuple[str, str, str, bool], Tuple[Any, Any]] = {}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generate questions using LLM based on domain cards"""

    def __init__(
        self,
        dataset_name: str,
        designer_model: str = "gpt-5-mini",
        output_dir: Optional[Path] = None,
        ollama_model_override: Optional[str] = None,
        ollama_base_url_override: Optional[str] = None,
        vllm_base_url_override: Optional[str] = None,
        huggingface_options: Optional[Dict[str, Any]] = None,
        llm_model_override: Optional[str] = None,
        vllm_max_tokens_override: Optional[int] = None
    ):
        """
        Initialize question generator.

        Args:
            dataset_name: Name of the dataset (e.g., "csbench_en")
            designer_model: Model identifier for tracking
            output_dir: Directory for outputs (default: outputs/stage2_questions)
        """
        self.dataset_name = dataset_name
        self.designer_model = designer_model
        self.hf_options = huggingface_options or {}
        self.llm_model_override = llm_model_override
        self.vllm_max_tokens_override = (
            int(vllm_max_tokens_override) if vllm_max_tokens_override else None
        )

        # Set output directory
        if output_dir is None:
            project_root = Path(__file__).parents[3]
            output_dir = project_root / "outputs" / "stage2_questions" / dataset_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize LLM manager
        logger.info("Initializing LLM manager...")
        self.llm = LLMManager(models_to_init=['openai', 'gemini', 'anthropic', 'grok', 'deepseek', 'doubao', 'qwen', 'llama', 'ollama', 'vllm'])

        # Apply Ollama overrides when provided so callers can switch models per run
        if ollama_model_override:
            logger.info(f"Ollama model override detected: {ollama_model_override}")
            self.llm.ollama_model_name = ollama_model_override
        if ollama_base_url_override:
            normalized_url = ollama_base_url_override.rstrip("/")
            if not normalized_url.endswith("/v1"):
                normalized_url = f"{normalized_url}/v1"
            logger.info(f"Ollama base URL override detected: {normalized_url}")
            self.llm.ollama_base_url = normalized_url

        if vllm_base_url_override:
            normalized_vllm_url = vllm_base_url_override.rstrip("/")
            if not normalized_vllm_url.endswith("/v1"):
                normalized_vllm_url = f"{normalized_vllm_url}/v1"
            if getattr(self.llm, "vllm_available", False):
                logger.info(f"vLLM base URL override detected: {normalized_vllm_url}")
                self.llm.vllm_base_url = normalized_vllm_url
            else:
                logger.warning(
                    "vLLM base URL override provided but vLLM client is not initialized. "
                    "Ensure --model vllm or config enables vLLM."
                )

        if huggingface_options:
            logger.info("Hugging Face overrides detected; local transformers generation enabled.")

        # Build system prompt
        logger.info(f"Building system prompt for {dataset_name}...")
        self.system_prompt = build_system_prompt(dataset_name, designer_model)

    async def generate_questions_from_prompt(
        self,
        user_prompt: str,
        expected_questions: int,
        model: str = "openai",
        temperature: float = 0.8,
        max_retries: int = 3,
        raw_output_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate questions for a specific batch/user prompt.
        """
        logger.info(
            f"Generating {expected_questions} questions with {model} (temp={temperature})..."
        )

        for attempt in range(max_retries):
            try:
                response = await self._call_llm(model, temperature, user_prompt)
                if raw_output_path:
                    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
                    raw_output_path.write_text(response, encoding="utf-8")
                questions = self._parse_response(response)

                if len(questions) != expected_questions:
                    logger.warning(
                        f"Expected {expected_questions} questions, got {len(questions)}. "
                        "Accepting the partial batch."
                    )
                    return questions

                logger.info(f"Successfully generated {len(questions)} questions")
                return questions

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    if model == "anthropic":
                        retry_delay = 30
                    else:
                        retry_delay = 2 ** attempt
                    logger.info(f"Retrying after {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                else:
                    raise

        raise RuntimeError("Failed to generate questions after all retries")

    async def _call_llm(self, model: str, temperature: float, user_prompt: str) -> str:
        """Call the LLM with the prompts"""

        model_override = self.llm_model_override

        if model == "openai":
            if not self.llm.openai_available:
                raise RuntimeError("OpenAI client not available")

            model_name = model_override or getattr(self.llm, "openai_default_model", "gpt-4o")

            if "gpt-5" in model_name.lower():
                response = await self.llm.openai_client.responses.create(
                    model=model_name,
                    input=[
                    {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_output_tokens=16000
                )

            if hasattr(response, "output_text"):
                return response.output_text
            if hasattr(response, "output") and response.output:
                parts = []
                for item in response.output:
                    content = item.get("content")
                    if isinstance(content, list):
                        for part in content:
                            text = part.get("text")
                            if text:
                                parts.append(text)
                if parts:
                    return "\n".join(parts)
                # Fallback: try to access choices if provided
                if hasattr(response, "choices") and response.choices:
                    return response.choices[0].message.content
                raise RuntimeError("GPT-5 response did not contain text content")
            else:
                params = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": 16000
                }
                response = await self.llm.openai_client.chat.completions.create(**params)
            return response.choices[0].message.content

        elif model == "gemini":
            if not self.llm.gemini_available:
                raise RuntimeError("Gemini client not available")

            # Rate limiting
            await self.llm.rate_limit_gemini()

            prompt = f"{self.system_prompt}\n\n{user_prompt}"
            gemini_model = model_override or self.llm.gemini_model_name
            response = await asyncio.to_thread(
                self.llm.gemini_client.models.generate_content,
                model=gemini_model,
                contents=prompt,
                config=self.llm.genai_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=16000
                )
            )
            if hasattr(response, "text"):
                return response.text
                return response.candidates[0].content.parts[0].text

        elif model == "anthropic":
            if not self.llm.anthropic_available:
                raise RuntimeError("Anthropic client not available")

            anthropic_model = model_override or self.llm.anthropic_model_name
            user_message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt}
                    ],
                }
            ]
            response_text = await self.llm.call_anthropic_messages(
                messages=user_message,
                system_prompt=self.system_prompt,
                temperature=temperature,
                max_tokens=16000,
                model_name=anthropic_model,
            )
            return response_text

        elif model == "grok":
            if not getattr(self.llm, "grok_available", False):
                raise RuntimeError("Grok client not available")

            return await self.llm.call_grok(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=temperature
            )

        elif model == "deepseek":
            if not self.llm.deepseek_available:
                raise RuntimeError("DeepSeek client not available")

            deepseek_model = model_override or self.llm.deepseek_model_name
            response = await self.llm.deepseek_client.chat.completions.create(
                model=deepseek_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=16000,
                extra_body=getattr(self.llm, "deepseek_extra_body", None)
            )
            return response.choices[0].message.content

        elif model == "ollama":
            if not self.llm.ollama_available:
                raise RuntimeError("Ollama client not available")

            # Ollama uses OpenAI-compatible API
            logger.info(f"Calling Ollama with model: {self.llm.ollama_model_name}")

            ollama_model = model_override or self.llm.ollama_model_name
            payload = {
                "model": ollama_model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": temperature,
                "max_tokens": 16000,
                "stream": False
            }

            response = await self.llm.ollama_client.post(
                f"{self.llm.ollama_base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")

            result = response.json()
            return result["choices"][0]["message"]["content"]

        elif model == "doubao":
            if not getattr(self.llm, "doubao_available", False):
                raise RuntimeError("Doubao client not available")

            doubao_model = model_override or self.llm.doubao_model_name
            combined_prompt = f"{self.system_prompt}\n\n{user_prompt}"
            response = await self.llm.doubao_client.responses.create(
                model=doubao_model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": combined_prompt}
                        ]
                    }
                ],
                max_output_tokens=16000
            )
            if hasattr(response, "output_text"):
                return response.output_text
            if getattr(response, "output", None):
                parts: List[str] = []
                for item in response.output:
                    content = item.get("content", [])
                    for block in content:
                        text = block.get("text")
                        if text:
                            parts.append(text)
                if parts:
                    return "".join(parts)
            raise RuntimeError("Doubao response did not contain text content")

        elif model == "qwen":
            if not getattr(self.llm, "qwen_available", False):
                raise RuntimeError("Qwen client not available")

            qwen_model = model_override or self.llm.qwen_model_name
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = await self.llm.qwen_client.chat.completions.create(
                model=qwen_model,
                messages=messages,
                temperature=temperature,
                max_tokens=16000,
                stream=False
            )
            return response.choices[0].message.content

        elif model == "llama":
            if not getattr(self.llm, "llama_available", False):
                raise RuntimeError("Llama client not available")

            llama_model = model_override or self.llm.llama_model_name
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            extra_headers = self.llm.config.get("llama", {}).get("extra_headers", {})
            extra_body = self.llm.config.get("llama", {}).get("extra_body", {})

            response = await self.llm.llama_client.chat.completions.create(
                model=llama_model,
                messages=messages,
                temperature=temperature,
                max_tokens=16000,
                extra_headers=extra_headers,
                extra_body=extra_body
            )
            return response.choices[0].message.content

        elif model == "huggingface":
            return await self._call_huggingface_model(temperature, user_prompt)

        elif model == "vllm":
            if not getattr(self.llm, "vllm_available", False):
                raise RuntimeError("vLLM client not available")

            target_model = model_override or self.llm.vllm_model_name
            max_tokens = self.vllm_max_tokens_override or 16000
            return await self.llm.call_vllm(
                prompt=user_prompt,
                system_prompt=self.system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model_name=target_model
            )

        else:
            raise ValueError(f"Unsupported model: {model}")

    def _parse_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into list of questions"""

        # Clean up response (remove markdown fences if present)
        response = response.strip()
        if response.startswith("<think>"):
            end_idx = response.find("</think>")
            if end_idx != -1:
                response = response[end_idx + len("</think>"):].lstrip()
            else:
                response = response.lstrip("<think>").lstrip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        json_start = response.find("[")
        if json_start > 0:
            response = response[json_start:]

        # Parse JSON
        try:
            questions = json.loads(response)
        except json.JSONDecodeError as e:
            snippet = response[:1000] + ("..." if len(response) > 1000 else "")
            logger.warning(
                "JSON parse error: %s; attempting automatic repair. Raw response snippet:\n%s",
                e,
                snippet
            )
            repaired = self._repair_json_response(response)
            questions = json.loads(repaired)

        # Validate it's a list
        if not isinstance(questions, list):
            raise ValueError(f"Expected list, got {type(questions)}")

        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(questions):
            if isinstance(item, dict):
                normalized.append(item)
                continue

            if isinstance(item, str):
                stripped = item.strip()
                if not stripped:
                    logger.warning("Skipping empty string question at index %d", idx)
                    continue
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict):
                        normalized.append(parsed)
                        continue
                    logger.warning(
                        "Skipping question at index %d: parsed string but got %s",
                        idx,
                        type(parsed).__name__
                    )
                    continue
                except json.JSONDecodeError as err:
                    logger.warning(
                        "Skipping question at index %d: string is not valid JSON (%s)",
                        idx,
                        err
                    )
                    continue

            logger.warning(
                "Skipping question at index %d due to unsupported type %s",
                idx,
                type(item).__name__
            )

        if not normalized:
            raise ValueError("No valid question objects found in model response")

        return normalized

    async def _load_hf_model(self):
        """Load or reuse a Hugging Face model/tokenizer pair."""
        if not self.hf_options:
            logger.info("No Hugging Face options provided; defaulting to deepseek-ai/DeepSeek-R1 causal LM.")

        model_id = self.hf_options.get("model_id") or "deepseek-ai/DeepSeek-R1"
        dtype = (self.hf_options.get("dtype") or "auto").lower()
        device_map = self.hf_options.get("device_map", "auto")
        trust_remote = bool(self.hf_options.get("trust_remote_code", False))

        cache_key = (model_id, dtype, str(device_map), trust_remote)
        if cache_key in HF_MODEL_CACHE:
            return HF_MODEL_CACHE[cache_key]

        def _load():
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig  # type: ignore[import]
                import torch  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError(
                    "Transformers (and torch) must be installed to use --model huggingface. "
                    "Install with `pip install transformers accelerate torch sentencepiece`."
                ) from exc

            # Copy model_id to local scope (we may reassign it for AWQ/GPTQ variants)
            current_model_id = model_id

            logger.info(f"Loading Hugging Face model '{current_model_id}' (device_map={device_map}, dtype={dtype})...")

            # Check for cache directory
            import os
            cache_dir = self.hf_options.get("cache_dir") or os.environ.get("HF_HOME")
            if cache_dir:
                logger.info(f"Using Hugging Face cache directory: {cache_dir}")

            # Load config first to check for FP8 quantization
            config_kwargs = {"trust_remote_code": trust_remote}
            if cache_dir:
                config_kwargs["cache_dir"] = cache_dir
            config = AutoConfig.from_pretrained(current_model_id, **config_kwargs)

            # Check if model has FP8 quantization config and disable it
            if hasattr(config, 'quantization_config') and config.quantization_config is not None:
                quant_method = getattr(config.quantization_config, 'quant_method', None)
                if quant_method == 'fp8':
                    logger.warning(f"Model has FP8 quantization which requires compute capability >= 8.9. Disabling FP8 quantization.")
                    config.quantization_config = None

            model_kwargs: Dict[str, Any] = {
                "trust_remote_code": trust_remote,
                "device_map": device_map,
                "config": config
            }
            if cache_dir:
                model_kwargs["cache_dir"] = cache_dir

            if dtype != "auto":
                torch_dtype = getattr(torch, dtype if dtype in ("float32", "float16", "bfloat16") else dtype, None)
                if torch_dtype is None:
                    raise ValueError(f"Unsupported dtype '{dtype}' for Hugging Face model.")
                model_kwargs["dtype"] = torch_dtype

            max_memory = self.hf_options.get("max_memory")
            max_memory_per_gpu = self.hf_options.get("max_memory_per_gpu")

            if max_memory:
                model_kwargs["max_memory"] = max_memory
            elif max_memory_per_gpu:
                # Build max_memory dict for all GPUs
                num_gpus = torch.cuda.device_count()
                max_memory_dict = {i: max_memory_per_gpu for i in range(num_gpus)}
                max_memory_dict["cpu"] = "0GiB"  # Disable CPU offloading
                model_kwargs["max_memory"] = max_memory_dict
                logger.info(f"Setting max_memory per GPU: {max_memory_per_gpu} across {num_gpus} GPUs (CPU offload disabled)")

            attn_impl = self.hf_options.get("attn_implementation")
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl

            # Handle quantization
            quantization = self.hf_options.get("quantization", "none")
            if quantization == "int8":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Loading model with INT8 quantization (bitsandbytes)")
            elif quantization == "int4_bnb":
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Loading model with INT4 quantization (bitsandbytes NF4)")
            elif quantization == "int4_awq":
                # AWQ models are pre-quantized, just load them normally
                logger.info("Loading AWQ-quantized model (INT4)")
                # Check if AWQ version exists by trying the -AWQ suffix
                if not current_model_id.endswith("-AWQ") and "awq" not in current_model_id.lower():
                    # Try to find AWQ version
                    awq_model_id = f"{current_model_id}-AWQ"
                    logger.info(f"Attempting to load AWQ version: {awq_model_id}")
                    try:
                        # Test if AWQ model exists
                        from transformers import AutoConfig
                        _ = AutoConfig.from_pretrained(awq_model_id, trust_remote_code=trust_remote)
                        current_model_id = awq_model_id
                        logger.info(f"Found AWQ model: {awq_model_id}")
                    except Exception:
                        logger.warning(f"AWQ model not found ({awq_model_id}), falling back to BNB 4-bit quantization")
                        from transformers import BitsAndBytesConfig
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        model_kwargs["quantization_config"] = bnb_config
                        quantization = "int4_bnb"
            elif quantization == "int4_gptq":
                # GPTQ models are also pre-quantized
                logger.info("Loading GPTQ-quantized model (INT4)")
                if not current_model_id.endswith("-GPTQ") and "gptq" not in current_model_id.lower():
                    gptq_model_id = f"{current_model_id}-GPTQ"
                    logger.info(f"Attempting to load GPTQ version: {gptq_model_id}")
                    try:
                        from transformers import AutoConfig
                        _ = AutoConfig.from_pretrained(gptq_model_id, trust_remote_code=trust_remote)
                        current_model_id = gptq_model_id
                        logger.info(f"Found GPTQ model: {gptq_model_id}")
                    except Exception:
                        logger.warning(f"GPTQ model not found ({gptq_model_id}), falling back to BNB 4-bit quantization")
                        from transformers import BitsAndBytesConfig
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
                        model_kwargs["quantization_config"] = bnb_config
                        quantization = "int4_bnb"
            elif quantization == "none":
                logger.info("Loading model in full precision (BF16/FP16)")

            # Backward compatibility
            if self.hf_options.get("load_in_8bit") and "quantization_config" not in model_kwargs:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = bnb_config
            if self.hf_options.get("load_in_4bit") and "quantization_config" not in model_kwargs:
                from transformers import BitsAndBytesConfig
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = bnb_config

            model = AutoModelForCausalLM.from_pretrained(current_model_id, **model_kwargs)

            # Log device placement
            if hasattr(model, 'hf_device_map'):
                logger.info(f"Model distributed across devices: {set(model.hf_device_map.values())}")
                device_counts = {}
                for device in model.hf_device_map.values():
                    device_counts[device] = device_counts.get(device, 0) + 1
                logger.info(f"Layers per device: {device_counts}")

            tokenizer_kwargs = {"trust_remote_code": trust_remote}
            if cache_dir:
                tokenizer_kwargs["cache_dir"] = cache_dir
            tokenizer = AutoTokenizer.from_pretrained(current_model_id, **tokenizer_kwargs)

            # Store quantization info for metadata
            if not hasattr(self, '_quantization_method'):
                self._quantization_method = quantization

            HF_MODEL_CACHE[cache_key] = (tokenizer, model)
            return tokenizer, model

        return await asyncio.to_thread(_load)

    def _hf_generate(self, tokenizer, model, temperature: float, user_prompt: str) -> str:
        """Run synchronous Hugging Face generation (executed in a thread)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )
        else:
            prompt = f"{self.system_prompt}\n\n{user_prompt}\n"
            inputs = tokenizer(prompt, return_tensors="pt")

        if hasattr(inputs, "to"):
            inputs = inputs.to(model.device)

        if isinstance(inputs, Mapping):
            input_kwargs = dict(inputs)
        elif hasattr(inputs, "input_ids"):
            input_kwargs = {"input_ids": inputs.input_ids}
        else:
            input_kwargs = {"input_ids": inputs}

        normalized_kwargs = {}
        for key, value in input_kwargs.items():
            if hasattr(value, "to"):
                normalized_kwargs[key] = value.to(model.device)
            else:
                normalized_kwargs[key] = value
        input_kwargs = normalized_kwargs

        generation_kwargs = {
            "max_new_tokens": self.hf_options.get("max_new_tokens", 2048),
            "temperature": temperature,
            "do_sample": temperature > 0,
            "top_p": self.hf_options.get("top_p", 0.95),
            "top_k": self.hf_options.get("top_k"),
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id
        }

        # Remove None values
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

        outputs = model.generate(**input_kwargs, **generation_kwargs)
        start_idx = input_kwargs["input_ids"].shape[-1]
        text = tokenizer.decode(outputs[0][start_idx:], skip_special_tokens=True)
        return text.strip()

    async def _call_huggingface_model(self, temperature: float, user_prompt: str) -> str:
        """Asynchronously call a local Hugging Face model."""
        tokenizer, model = await self._load_hf_model()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._hf_generate, tokenizer, model, temperature, user_prompt)

    def save_questions(
        self,
        questions: List[Dict[str, Any]],
        filename: str = "questions.jsonl"
    ) -> Path:
        """
        Save questions to JSONL file.

        Args:
            questions: List of question dictionaries
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(questions)} questions to {output_path}")
        return output_path

    @staticmethod
    def _repair_json_response(response: str) -> str:
        """Best-effort repair for slightly malformed JSON arrays."""
        escaped = QuestionGenerator._escape_unescaped_newlines(response)
        trimmed = QuestionGenerator._trim_after_json_array(escaped)
        balanced = QuestionGenerator._balance_pairs(trimmed, "[", "]")
        balanced = QuestionGenerator._balance_pairs(balanced, "{", "}")
        sanitized = QuestionGenerator._escape_invalid_backslashes(balanced)
        return sanitized

    @staticmethod
    def _trim_after_json_array(text: str) -> str:
        last_bracket = text.rfind("]")
        if last_bracket == -1:
            return text
        return text[:last_bracket + 1]

    @staticmethod
    def _balance_pairs(text: str, open_char: str, close_char: str) -> str:
        balance = 0
        for ch in text:
            if ch == open_char:
                balance += 1
            elif ch == close_char and balance > 0:
                balance -= 1
        if balance > 0:
            text += close_char * balance
        return text

    @staticmethod
    def _escape_unescaped_newlines(text: str) -> str:
        result = []
        in_string = False
        escaped = False

        for ch in text:
            if in_string:
                if escaped:
                    result.append(ch)
                    escaped = False
                    continue
                if ch == "\\":
                    escaped = True
                    result.append(ch)
                    continue
                if ch == '"':
                    in_string = False
                    result.append(ch)
                    continue
                if ch == "\n":
                    result.append("\\n")
                    continue
                if ch == "\r":
                    result.append("\\r")
                    continue
                if ord(ch) < 0x20:
                    result.append(f"\\u{ord(ch):04x}")
                    continue
                result.append(ch)
            else:
                if ch == '"':
                    in_string = True
                result.append(ch)

        return "".join(result)

    @staticmethod
    def _escape_invalid_backslashes(text: str) -> str:
        """
        Replace stray backslashes (e.g., '\_' from Markdown) with escaped versions
        so that json.loads doesn't raise Invalid \\escape.
        """
        # Backslashes that are already part of valid escapes (\", \\, \/, \b, \f, \n, \r, \t, \uXXXX) should be left alone.
        pattern = r'\\(?!["\\/bfnrtu])'
        return re.sub(pattern, r'\\\\', text)

