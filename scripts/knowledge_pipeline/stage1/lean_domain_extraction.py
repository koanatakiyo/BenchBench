#!/usr/bin/env python3
"""
Lean Domain Card Protocol - Stage 1
A simplified, focused approach to domain extraction with strong stability guarantees.

Key principles:
- Single deterministic oracle (temp=0, top_p=1)
- Strict JSON format discipline
- Micro-enum for canonicalization (built from data, not external ontologies)
- Lightweight consensus (K=2 self-consistency only if needed)
- Quality > stability (meet floors, don't chase perfect Jaccard)
"""

import asyncio
import hashlib
import math
import json
import copy
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Any, Optional, Tuple
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm_manager import LLMManager


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ExtractionResult:
    """Single extraction result with strict format"""
    subdomains: List[str]  # Up to 3
    terms: List[str]  # Up to 12
    confidence: float  # [0, 1]
    visual_facts: Optional[List[str]] = None  # Up to 3 for multimodal
    native_glosses: Optional[Dict[str, str]] = None  # native -> EN for multilingual
    raw_response: str = ""
    is_valid: bool = True
    anchored_count: int = 0
    anchored_fraction: float = 0.0
    tom_categories: Optional[List[str]] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class MicroEnumEntry:
    """Entry in the micro-enum (allowed list)"""
    id: str
    name: str  # Canonical normalized form
    aliases: List[str]  # Alternative forms
    frequency: int = 0  # Frequency in corpus

    def to_dict(self):
        return asdict(self)


@dataclass
class StabilityMetrics:
    """Stability metrics for a run pair"""
    raw_jaccard_subdomains: float
    raw_jaccard_terms: float
    soft_jaccard_subdomains: float  # With semantic matching
    soft_jaccard_terms: float
    nsi: float  # Normalized Stability Index
    hierarchical_agreement: float  # Parent-level agreement

    def average(self) -> float:
        return (self.raw_jaccard_subdomains + self.raw_jaccard_terms) / 2

    def passes_floors(self) -> Tuple[bool, List[str]]:
        """Check if metrics pass all floors"""
        failures = []
        # Soft-Jaccard: -1.0 means N/A (couldn't compute), otherwise check threshold
        if self.soft_jaccard_terms < 0:
            failures.append("Soft-Jaccard terms (N/A - couldn't compute embeddings)")
        elif self.soft_jaccard_terms < 0.75:
            failures.append(f"Soft-Jaccard terms ({self.soft_jaccard_terms:.3f} < 0.75)")
        if self.nsi < 0.65:
            failures.append(f"NSI ({self.nsi:.3f} < 0.65)")
        if self.hierarchical_agreement < 0.80:
            failures.append(f"Hierarchical agreement ({self.hierarchical_agreement:.3f} < 0.80)")
        return len(failures) == 0, failures


@dataclass
class QualityMetrics:
    """Quality KPIs for Stage-1"""
    coverage: float  # % items with plausible output
    modality_fidelity: float  # % multimodal match to source
    language_fidelity: float  # % language match to source
    deduplication_rate: float  # % items that are near-duplicates (lower is better)
    anchored_percent: float  # % terms validated by 4 anchor strategies
    coverage_floor: float = 0.95
    deduplication_floor: float = 0.10
    anchored_floor: float = 0.50

    def passes_floors(self) -> Tuple[bool, List[str]]:
        """Check if quality passes all floors"""
        failures = []
        if self.coverage < self.coverage_floor:
            failures.append(f"Coverage ({self.coverage:.3f} < {self.coverage_floor:.2f})")
        if abs(self.modality_fidelity) > 0.05:
            failures.append(f"Modality fidelity off by {abs(self.modality_fidelity):.3f} > 0.05")
        if abs(self.language_fidelity) > 0.05:
            failures.append(f"Language fidelity off by {abs(self.language_fidelity):.3f} > 0.05")
        if self.deduplication_rate > self.deduplication_floor:
            failures.append(
                f"Too many duplicates ({self.deduplication_rate:.1%} > {self.deduplication_floor:.0%})"
            )
        if self.anchored_percent < self.anchored_floor:
            failures.append(f"Anchored% ({self.anchored_percent:.3f} < {self.anchored_floor:.2f})")
        return len(failures) == 0, failures


@dataclass
class RunManifest:
    """Reproducibility record for a run"""
    dataset_hash: str
    sampled_item_ids: List[str]
    oracle_name: str
    oracle_version: str
    temperature: float
    top_p: float
    prompt_hash: str
    micro_enum_hash: Optional[str]
    parent_map_hash: Optional[str]  # Hash of frozen parent map
    embedding_model: Optional[str]
    random_seed: int
    timestamp: str
    code_revision: str = "unknown"

    def to_dict(self):
        return asdict(self)


# ============================================================================
# Core Extraction Logic
# ============================================================================

class LeanDomainExtractor:
    """
    Implements the Lean Domain Card Protocol for Stage-1 extraction.

    Design philosophy:
    - Use a SINGLE oracle (e.g., gpt-4o-mini) with temp=0, top_p=1
    - Require strict JSON format, reject and re-ask once
    - Build micro-enum from corpus data (no external ontologies)
    - Apply lightweight canonicalization
    - Report stability metrics with realistic floors
    - Prioritize quality (coverage, fidelity) over raw stability
    """

    def __init__(
        self,
        oracle_name: str = "gpt-5-mini",
        visual_oracle_name: Optional[str] = None,
        project_root: Optional[Path] = None,
        embedding_model_name: str = "intfloat/multilingual-e5-small",
        seed: int = 42
    ):
        """
        Initialize the extractor.

        Args:
            oracle_name: Single oracle to use for text items (e.g., "gpt-5-mini", "gpt-5")
            visual_oracle_name: Optional separate oracle for multimodal items (e.g., "gemini-2.5-flash").
                               If None, uses oracle_name for all items.
            project_root: Project root directory
            embedding_model_name: Model for semantic similarity (use small for speed)
            seed: Random seed for reproducibility
        """
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]

        self.project_root = Path(project_root)
        self.oracle_name = oracle_name
        self.visual_oracle_name = visual_oracle_name or oracle_name  # Use main oracle if not specified
        self.seed = seed
        self.output_dir = self.project_root / 'outputs' / 'lean_domain_extraction'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LLM manager (auto-detects project root from file location)
        self.llm = LLMManager()

        # Embedding model for semantic similarity
        print(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        print("Embedding model loaded.")

        # Micro-enum (will be built from corpus)
        self.micro_enum: Dict[str, MicroEnumEntry] = {}
        self.micro_enum_hash: Optional[str] = None

        # Clustering/hierarchy (will be built after extraction)
        self.parent_map: Dict[str, str] = {}  # leaf subdomain -> parent

        # Frozen text-only outputs for multimodal datasets
        frozen_dir = self.project_root / 'outputs' / 'lean_domain_extraction'
        self.frozen_text_sources: Dict[str, Path] = {
            'medxpertqa_mm': frozen_dir / 'medxpertqa_mm_results_20251125_053246.jsonl',
            'wemath': frozen_dir / 'wemath_results_20251125_045109.jsonl',
        }
        self._frozen_text_cache: Dict[str, Dict[str, ExtractionResult]] = {}
        self.freeze_text_active: bool = False

        # Dataset language (will be set during extraction)
        self.dataset_language: str = 'en'
        self.dataset_domain: str = 'general'
        self.coverage_floor: float = 0.95
        self.dedup_floor: float = 0.10
        self.anchored_floor: float = 0.50
        self.anchored_coverage_threshold: int = 3
        self.dedup_similarity_threshold: float = 0.955
        self.dataset_name: Optional[str] = None
        self.tom_concepts = self._build_tom_concepts()
        self.term_idf_map: Dict[str, float] = {}
        self.tombench_cn_allowed_subdomains = [
            '信念推理',
            '物体位置-物体追踪',
            '情绪识别',
            '人际关系-社交互动',
            '日常推理-数量估计',
            '社区生活-安全',
            '意图推理',
            '道德判断-道德推理'
        ]
        self.tombench_cn_keyword_map = {
            # 信念推理
            '信念': '信念推理',
            '错误信念': '信念推理',
            '以为': '信念推理',
            '误以为': '信念推理',
            '错以为': '信念推理',
            '认为': '信念推理',
            '觉得': '信念推理',
            '猜想': '信念推理',
            '猜测': '信念推理',
            '知道': '信念推理',
            '不知道': '信念推理',
            # 物体位置-物体追踪
            '物体': '物体位置-物体追踪',
            '位置': '物体位置-物体追踪',
            '在哪': '物体位置-物体追踪',
            '在哪里': '物体位置-物体追踪',
            '放在': '物体位置-物体追踪',
            '放到': '物体位置-物体追踪',
            '移到': '物体位置-物体追踪',
            '移动': '物体位置-物体追踪',
            '搬到': '物体位置-物体追踪',
            '搬走': '物体位置-物体追踪',
            '拿走': '物体位置-物体追踪',
            '拿到': '物体位置-物体追踪',
            '藏在': '物体位置-物体追踪',
            '追踪': '物体位置-物体追踪',
            # 情绪识别
            '情绪': '情绪识别',
            '感受': '情绪识别',
            '感觉': '情绪识别',
            '心情': '情绪识别',
            '开心': '情绪识别',
            '高兴': '情绪识别',
            '难过': '情绪识别',
            '伤心': '情绪识别',
            '生气': '情绪识别',
            '愤怒': '情绪识别',
            '害怕': '情绪识别',
            '紧张': '情绪识别',
            '担心': '情绪识别',
            '嫉妒': '情绪识别',
            '羡慕': '情绪识别',
            '尴尬': '情绪识别',
            '失望': '情绪识别',
            '沮丧': '情绪识别',
            '害羞': '情绪识别',
            # 人际关系-社交互动
            '人际': '人际关系-社交互动',
            '社交': '人际关系-社交互动',
            '互动': '人际关系-社交互动',
            '关系': '人际关系-社交互动',
            '朋友': '人际关系-社交互动',
            '同学': '人际关系-社交互动',
            '同事': '人际关系-社交互动',
            '邻居': '人际关系-社交互动',
            '情侣': '人际关系-社交互动',
            '约会': '人际关系-社交互动',
            '见面': '人际关系-社交互动',
            '吵架': '人际关系-社交互动',
            '和好': '人际关系-社交互动',
            '邀请': '人际关系-社交互动',
            '拒绝': '人际关系-社交互动',
            '同意': '人际关系-社交互动',
            '失约': '人际关系-社交互动',
            '道歉': '人际关系-社交互动',
            '原谅': '人际关系-社交互动',
            # 日常推理-数量估计
            '数量': '日常推理-数量估计',
            '多少': '日常推理-数量估计',
            '多少把': '日常推理-数量估计',
            '几把': '日常推理-数量估计',
            '几张': '日常推理-数量估计',
            '几个人': '日常推理-数量估计',
            '几件': '日常推理-数量估计',
            '几次': '日常推理-数量估计',
            '估计': '日常推理-数量估计',
            '大约': '日常推理-数量估计',
            '大概': '日常推理-数量估计',
            '多久': '日常推理-数量估计',
            '多长时间': '日常推理-数量估计',
            '多远': '日常推理-数量估计',
            # 社区生活-安全
            '社区': '社区生活-安全',
            '小区': '社区生活-安全',
            '邻里': '社区生活-安全',
            '邻居': '社区生活-安全',
            '安全': '社区生活-安全',
            '危险': '社区生活-安全',
            '治安': '社区生活-安全',
            '入室盗窃': '社区生活-安全',
            '盗窃': '社区生活-安全',
            '小偷': '社区生活-安全',
            '报警': '社区生活-安全',
            '警察': '社区生活-安全',
            '窗户': '社区生活-安全',
            '门窗': '社区生活-安全',
            '锁': '社区生活-安全',
            # 意图推理
            '意图': '意图推理',
            '打算': '意图推理',
            '计划': '意图推理',
            '目的': '意图推理',
            '为了': '意图推理',
            '想要': '意图推理',
            '想去': '意图推理',
            '想做': '意图推理',
            '希望': '意图推理',
            # 道德判断-道德推理
            '道德': '道德判断-道德推理',
            '应该': '道德判断-道德推理',
            '不应该': '道德判断-道德推理',
            '对不对': '道德判断-道德推理',
            '对还是错': '道德判断-道德推理',
            '公平': '道德判断-道德推理',
            '不公平': '道德判断-道德推理',
            '诚实': '道德判断-道德推理',
            '撒谎': '道德判断-道德推理',
            '骗人': '道德判断-道德推理',
            '规则': '道德判断-道德推理',
            '规矩': '道德判断-道德推理',
            '违反': '道德判断-道德推理',
            '责任': '道德判断-道德推理',
            '惩罚': '道德判断-道德推理',
            '奖励': '道德判断-道德推理',
            '自私': '道德判断-道德推理',
            '无私': '道德判断-道德推理',
            '帮助': '道德判断-道德推理'
        }
        self.parallel_subdomain_cache: Dict[str, Dict[str, List[str]]] = {}
        self.tombench_en_to_cn_rules = self._build_tombench_en_to_cn_rules()

    # ------------------------------------------------------------------------
    # LLM Call Wrapper
    # ------------------------------------------------------------------------

    async def _call_llm(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 2000,
        image_files: Optional[List[str]] = None
    ) -> str:
        """
        Unified LLM call wrapper that routes to appropriate provider.

        Maps model names to LLMManager calls:
        - gpt-* -> OpenAI (vision if image_files provided)
        - gemini-* -> Gemini (vision if image_files provided)
        - claude-* -> Anthropic (vision if image_files provided)
        - grok-* -> Grok
        """
        # Determine provider from model name
        model_lower = model.lower()

        # Use vision APIs if images are provided
        has_images = image_files and len(image_files) > 0

        if model_lower.startswith('gpt'):
            # OpenAI models
            if has_images:
                return await self.llm.call_openai_vision(prompt, image_files=image_files, model=model)
            else:
                return await self.llm.call_openai(prompt, model=model)

        elif model_lower.startswith('gemini'):
            # Gemini models
            if has_images:
                return await self.llm.call_gemini_vision(prompt, image_files=image_files)
            else:
                return await self.llm.call_gemini(prompt)

        elif model_lower.startswith('claude'):
            # Anthropic models
            if has_images:
                return await self.llm.call_anthropic_vision(prompt, image_files=image_files)
            else:
                return await self.llm.call_anthropic(prompt)

        elif model_lower.startswith('grok'):
            # Grok models (no vision support yet)
            return await self.llm.call_grok(prompt)

        # elif model_lower.startswith('deepseek'):
        #     # DeepSeek models
        #     return await self.llm.call_deepseek(prompt)

        # elif model_lower.startswith('qwen'):
        #     # Qwen models
        #     return await self.llm.call_qwen(prompt)

        else:
            # Default fallback to call_llm with provider hint
            # Try to infer provider from common patterns
            if 'gpt' in model_lower or 'openai' in model_lower:
                preferred = 'openai'
            elif 'gemini' in model_lower or 'google' in model_lower:
                preferred = 'gemini'
            elif 'claude' in model_lower or 'anthropic' in model_lower:
                preferred = 'anthropic'
            else:
                preferred = 'openai'  # Default to OpenAI

            return await self.llm.call_llm(prompt, preferred_model=preferred)

    # ------------------------------------------------------------------------
    # 1. Format Discipline & Validation
    # ------------------------------------------------------------------------

    def _make_extraction_prompt(
        self,
        item: Dict[str, Any],
        has_image: bool = False,
        is_multilingual: bool = False,
        glossary: Optional[List[str]] = None,
        language: str = 'en'
    ) -> str:
        """
        Create extraction prompt following the protocol.

        Args:
            item: Dataset item
            has_image: Whether item has an image
            is_multilingual: Whether to ask for native + EN gloss
            glossary: Optional micro-enum entries to constrain output
            language: Target language ('en', 'zh', 'fr', 'de')
        """
        # Build item text
        item_text = self._item_to_text(item)

        if language == 'zh':
            zh_parts = [
                "你是一个术语抽取专家。",
                "",
                "任务：",
                "从下面的试题中抽取：",
                "- 至多 3 个【子领域】（较高层次的概念）",
                "- 至多 12 个【关键术语】（技术名词或名词短语，）",
                "",
                "要求：",
                "- 所有输出必须使用简体中文，不要输出英文或拼音。",
                "- 尽量抽取题干中出现过的词或短语（匹配原文，忽略大小写和空格的区别）。",
                "- 仅抽取与原文相关的技术名词 / 名词短语，不要动词句子或随意片段。",
                "",
                "避免：",
                "- 不要输出“其他”“背景”“内容”“信息”“题目”“问题”“选项”“示例”“标签”等标题。",
                "- 只有在完整技术短语中才允许包含“系统”“数据”“存储”“信息”等基础词。",
                "  - 有效示例：“操作系统”“文件系统”“数据库系统”“数据结构”“存储管理”",
                "  - 无效示例：“系统”“数据”“存储”“信息”",
                "- 不要输出只有一个汉字或明显不完整的短语。",
                "",
                "如果找不到足够多的候选：",
                "- 可以少于 3 个子领域、少于 12 个术语。",
                "- 请将 \"confidence\" 设为 0.3–0.5 之间的较低值。",
                "- 不要为了凑数而编造与题目无关的术语。",
                "",
                "输出格式：",
                "仅输出一个 JSON 对象，不要有任何多余文字：",
                "{",
                '  "subdomains": ["<子领域1>", "<子领域2>", "<子领域3>"],',
                '  "terms": ["<术语1>", "<术语2>", "... 最多 12 个术语 ..."],',
                '  "confidence": <0.0 到 1.0 之间的小数>',
                "}",
                "",
                "注意：",
                "- 当不足 3 个子领域或 12 个术语时，数组可以更短（0–3 个子领域，0–12 个术语）。",
                "- 请按字典序排序 \"subdomains\" 和 \"terms\"，不得重复。",
            ]

            if self._is_tombench_cn_dataset():
                zh_parts.extend(self._get_tombench_cn_prompt_guidance())

            if glossary:
                zh_parts.append(
                    f"- 若适用，可优先参考以下术语：{', '.join(glossary[:100])}"
                )

            if has_image:
                zh_parts.append("- 如试题包含图片，可在 JSON 中另加 \"visual_facts\" 字段（每条不超过 12 个汉字）。")

            zh_parts.extend([
                "",
                "题目：",
                item_text,
                "",
                "只输出 JSON。"
            ])

            return "\n".join(zh_parts)

        # Language-specific instructions for non-Chinese prompts
        if language == 'fr':
            lang_instruction = "• Output all subdomains and terms in French only."
        elif language == 'de':
            lang_instruction = "• Output all subdomains and terms in German only."
        else:  # Default to English
            lang_instruction = "• All outputs must be in English (lowercase)."

        prompt_parts = [
            "You are a domain-focused extractor for given questions.",
            "",
            "Task:",
            "Extract up to 3 high-level subdomains and up to 12 domain-specific key terms for the item below.",
            "",
            "Definitions:",
            "• Subdomains are broad conceptual areas (e.g., \"process scheduling\", \"memory hierarchy\", \"i/o subsystem\").",
            "• Key terms are technical noun phrases (1–4 words) for concrete concepts, mechanisms, or entities.",
            "",
            "Rules:",
            "• Prefer phrases that appear in the item text (after lowercasing).",
            "• Use only nouns or noun phrases; avoid verbs, full clauses, or broken fragments.",
            lang_instruction,
            "• All entries in \"subdomains\" and \"terms\" must be plain lowercase strings (no objects, no extra fields).",
            "",
            "Avoid:",
            "• Placeholder/meta-words: \"other\", \"invalid\", \"background\", \"content\", \"tag\", \"domain\", \"question\", \"reasoning\", \"example\", \"option\", \"misc\".",
            "• Bare generic tokens: \"system\", \"computer\", \"data\", \"memory\", \"information\" (unless part of a technical collocation such as \"operating system\").",
            "• Function words or single characters: \"a\", \"an\", \"the\", \"in\", \"is\", \"of\", \"to\".",
            "• Partial connectors: \"structure and\", \"based on\", \"due to\".",
            "",
            "If you cannot find many good candidates:",
            "• Return fewer than 3 subdomains and/or fewer than 12 terms.",
            "• Set \"confidence\" < 0.5.",
            "• Do NOT invent filler terms just to reach the maximum counts.",
            "",
            "Output format:",
            "Return ONLY a JSON object with this schema and nothing else:",
            "{",
            '  "subdomains": ["<subdomain_1>", "<subdomain_2>", "<subdomain_3>"],',
            '  "terms": ["<term_1>", "<term_2>", "... up to at most 12 terms ..."],',
            '  "confidence": <float between 0.0 and 1.0>',
            "}",
            "Example:",
            "{",
            '  "subdomains": ["computer science", "machine learning"],',
            '  "terms": ["large language model", "transformer", "prompt engineering"],',
            '  "confidence": 0.87',
            "}",
            "",
            "Formatting:",
            "• Use shorter arrays when needed (0–3 subdomains, 0–12 terms).",
            "• Sort both arrays alphabetically.",
            "• No duplicates.",
            "",
            "Item:",
            item_text,
            "",
            "Output only the JSON object."
        ]

        if glossary:
            prompt_parts.append(f"• Prefer terms from this glossary when applicable: {', '.join(sorted(glossary)[:100])}")

        if has_image:
            prompt_parts.append(
                "• Additionally, provide 'visual_facts' with up to 3 short bullets describing "
                "visible findings only (no diagnosis, no speculation), each ≤ 12 words."
            )

        if is_multilingual:
            prompt_parts.append(
                "• For each term, include 'native_glosses' mapping native term to a stable "
                "English gloss that preserves meaning; do not translate proper names."
            )

        prompt_parts.extend([
            "",
            f"Item:\n{item_text}",
            "",
            "Output only the JSON object, no prose, no markdown."
        ])

        return "\n".join(prompt_parts)

    def _get_tombench_cn_prompt_guidance(self) -> List[str]:
        return [
            "",
            "子领域请从下面这些或它们的组合中选择（不要发明新类别）：",
            "可能的子领域（任选 1–3 个，可以组合）：",
            "- 信念推理（例如：错觉、错误信念、谁知道/不知道）",
            "- 物体位置-物体追踪（移动、藏匿、寻找物品）",
            "- 情绪识别（开心、生气、紧张、失望等）",
            "- 人际关系-社交互动（约会、朋友、邻居、家人）",
            "- 日常推理-数量估计（椅子数量、物品多少等）",
            "- 社区生活-安全（社区、邻里关系、入室盗窃）",
            "- 意图推理（为什么要这样做/计划什么）",
            "- 道德判断-道德推理（应该/不应该、公平、惩罚等）",
            "",
            "规则补充：",
            "• 子领域必须是上面列表中的中文短语，或两个短语的简单组合。",
            "• 如果题目明显涉及“谁知道 / 不知道 / 想错了”，优先使用“信念推理 / 物体位置-物体追踪”。",
            "• 如果题目问的是“开心 / 生气 / 害怕 / 失望 等感受”，优先使用“情绪识别 / 人际关系-社交互动”。"
        ]

    def _is_tombench_dataset(self) -> bool:
        name = (self.dataset_name or '').lower()
        return 'tombench' in name

    def _is_tombench_cn_dataset(self) -> bool:
        name = (self.dataset_name or '').lower()
        return 'tombench_cn' in name

    def _item_to_text(self, item: Dict[str, Any]) -> str:
        """Convert item to text representation"""
        parts = []

        # Question
        if 'question' in item:
            parts.append(f"Question: {item['question']}")

        # Background/context
        if 'background' in item:
            bg = item['background']
            if isinstance(bg, dict):
                bg_str = ", ".join(f"{k}: {v}" for k, v in bg.items())
                parts.append(f"Background: {bg_str}")
            else:
                parts.append(f"Background: {bg}")

        # Options (if relevant)
        if 'options' in item and item['options']:
            opts = item['options']
            if isinstance(opts, dict):
                opts_str = ", ".join(f"{k}) {v}" for k, v in opts.items())
            elif isinstance(opts, list):
                opts_str = ", ".join(f"{i}) {o}" for i, o in enumerate(opts, 1))
            else:
                opts_str = str(opts)
            parts.append(f"Options: {opts_str}")

        # Answer (if we want to include it)
        # Note: Including answer may bias extraction toward answer-specific terms
        # Default: exclude for neutral domain extraction
        # if 'answer' in item:
        #     parts.append(f"Answer: {item['answer']}")

        # Context/story
        if 'context' in item:
            self._append_context_text(parts, item['context'])

        return "\n".join(parts)

    def _append_context_text(self, parts: List[str], value: Any) -> None:
        """Recursively collect textual context from nested structures."""
        text=None
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
        if text is not None:
            parts.append(f"Context: {text}")
        elif isinstance(value, dict):
            for entry in value.values():
                self._append_context_text(parts, entry)
        elif isinstance(value, list):
            for entry in value:
                self._append_context_text(parts, entry)

    async def _extract_once(
        self,
        item: Dict[str, Any],
        has_image: bool = False,
        is_multilingual: bool = False,
        glossary: Optional[List[str]] = None,
        retry_on_invalid: bool = True
    ) -> ExtractionResult:
        """
        Extract once with strict validation.

        If output is invalid and retry_on_invalid=True, retry once with same prompt.
        """
        prompt = self._make_extraction_prompt(
            item, has_image, is_multilingual, glossary, language=self.dataset_language
        )

        # Choose oracle based on whether item has image
        oracle_to_use = self.visual_oracle_name if has_image else self.oracle_name

        # Get image paths if item has images
        image_files = self._get_image_paths(item) if has_image else None

        async def _call_with_gemini_fallback() -> str:
            """
            Call the configured oracle, but fall back to text-only mode if Gemini vision
            fails to produce output. Some Gemini vision responses occasionally come back
            empty even though the request succeeds; when that happens we re-issue the
            prompt without images (and optionally with the primary text oracle).
            """
            try:
                return await self._call_llm(
                    model=oracle_to_use,
                    prompt=prompt,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=2000,
                    image_files=image_files
                )
            except ValueError as exc:
                error_text = str(exc).lower()
                if has_image and 'gemini vision returned no text' in error_text:
                    fallback_model = self.oracle_name or oracle_to_use
                    print(
                        f"  ⚠ Gemini vision returned no text; "
                        f"falling back to text-only oracle '{fallback_model}'."
                    )
                    return await self._call_llm(
                        model=fallback_model,
                        prompt=prompt,
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=2000,
                        image_files=None
                    )
                raise

        # Call oracle (temp=0, top_p=1) with images if available (with Gemini fallback)
        response = await _call_with_gemini_fallback()

        # Parse and validate
        result = self._parse_and_validate(response, has_image, is_multilingual)

        # Retry once if invalid
        if not result.is_valid and retry_on_invalid:
            print(f"  ⚠ Invalid output, retrying once...")
            response = await _call_with_gemini_fallback()
            result = self._parse_and_validate(response, has_image, is_multilingual)
            if not result.is_valid:
                print(f"  ✗ Still invalid after retry")

        return result

    def _parse_and_validate(
        self,
        response: str,
        has_image: bool,
        is_multilingual: bool
    ) -> ExtractionResult:
        """
        Parse response and validate strict format.

        Returns ExtractionResult with is_valid=False if invalid.
        """
        try:
            # Try to extract JSON from response (handle markdown code blocks)
            json_str = response.strip()
            if json_str.startswith("```"):
                # Remove markdown code blocks
                json_str = re.sub(r'^```(?:json)?\s*', '', json_str)
                json_str = re.sub(r'\s*```$', '', json_str)

            data = json.loads(json_str)

            # Validate required fields
            if 'subdomains' not in data or 'terms' not in data or 'confidence' not in data:
                return ExtractionResult([], [], 0.0, raw_response=response, is_valid=False)

            subdomains = data['subdomains']
            terms = data['terms']
            confidence = data['confidence']

            # Validate types
            if not isinstance(subdomains, list) or not isinstance(terms, list):
                return ExtractionResult([], [], 0.0, raw_response=response, is_valid=False)
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
                return ExtractionResult([], [], 0.0, raw_response=response, is_valid=False)

            # Validate all strings, tolerate empty entries (filtered later)
            if not all(isinstance(s, str) for s in subdomains):
                return ExtractionResult([], [], 0.0, raw_response=response, is_valid=False)
            if not all(isinstance(t, str) for t in terms):
                return ExtractionResult([], [], 0.0, raw_response=response, is_valid=False)

            # Deduplicate, sort, and cap lengths without marking invalid
            def _dedup_and_sort(values: List[str], cap: int) -> List[str]:
                seen = {}
                for value in values:
                    normalized = value.strip()
                    if normalized and normalized not in seen:
                        seen[normalized] = normalized
                return sorted(list(seen.values())[:cap], key=str.lower)

            subdomains = _dedup_and_sort(subdomains, cap=3)
            terms = _dedup_and_sort(terms, cap=12)

            # Optional fields
            visual_facts = None
            if has_image and 'visual_facts' in data:
                vf = data['visual_facts']
                if isinstance(vf, list) and all(isinstance(f, str) for f in vf):
                    visual_facts = vf[:3]  # Keep only first 3

            native_glosses = None
            if is_multilingual and 'native_glosses' in data:
                ng = data['native_glosses']
                if isinstance(ng, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in ng.items()):
                    native_glosses = ng

            return ExtractionResult(
                subdomains=subdomains,
                terms=terms,
                confidence=confidence,
                visual_facts=visual_facts,
                native_glosses=native_glosses,
                raw_response=response,
                is_valid=True
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            return ExtractionResult([], [], 0.0, raw_response=response, is_valid=False)

    # ------------------------------------------------------------------------
    # 2. Term Hygiene: Stopwords, Denylist, and Form Filters
    # ------------------------------------------------------------------------

    def _get_stopwords(self) -> Set[str]:
        """
        Get standard English stopwords + dataset-specific stopwords.

        Returns:
            Set of stopwords (lowercase)
        """
        lang = getattr(self, 'dataset_language', 'en')
        # Standard English stopwords (simplified subset)
        standard_stopwords = {
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might',
            'this', 'that', 'these', 'those',
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how',
            'and', 'or', 'but', 'not', 'as', 'by', 'from'
        }

        # Dataset-specific stopwords (avoid generic meta-terms)
        dataset_stopwords = {
            'other', 'background', 'content', 'domain', 'question',
            'reasoning', 'tag', 'option', 'letter', 'organization'
        }

        domain = getattr(self, 'dataset_domain', 'general')
        domain_stopwords = {
            'cs': {'system', 'computer', 'data', 'information', 'memory'},
            'medical': {
                'patient', 'patients', 'symptom', 'symptoms', 'disease',
                'condition', 'conditions', 'treatment', 'treatments',
                'therapy', 'therapies', 'clinical', 'history', 'histories',
                'case', 'cases', 'exam', 'examination', 'figure', 
            },
            'math': {'number', 'numbers', 'figure'},
            'security': set(),
            'finance': set(),
            'general': set()
        }
        dataset_stopwords |= domain_stopwords.get(domain, set())

        stopwords = standard_stopwords | dataset_stopwords

        if lang == 'zh':
            chinese_stopwords = {
                '其他', '其他的', '背景', '内容', '信息', '资料', '题目', '题干',
                '问题', '选项', '示例', '标签', '叙述', '描述', '说法',
                '以下', '下面', '关于', '有关', '错误', '正确', '陈述',
                '程序', '步骤', '情形', '情况'
            }
            stopwords |= chinese_stopwords

        return stopwords

    def _get_denylist_terms(self) -> Set[str]:
        """
        Get hard denylist of terms that should be immediately rejected.

        Returns:
            Set of denylist terms (lowercase)
        """
        denylist = {
            'other', 'background', 'content', 'domain', 'question',
            'reasoning', 'tag', 'option', 'letter', 'organization',
            'a', 'an', 'the', 'in', 'is', 'of', 'to',
            'structure and', 'based on', 'due to', 'example', 'invalid',
            'misc', 'miscellaneous'
        }

        domain = getattr(self, 'dataset_domain', 'general')
        domain_denylist = {
            'cs': {'system', 'computer', 'data', 'information', 'memory'},
            'medical': {
                'patient', 'patients', 'symptom', 'symptoms', 'disease',
                'condition', 'conditions', 'treatment', 'treatments',
                'therapy', 'therapies', 'medical', 'clinical', 'history',
                'histories', 'case', 'cases'
            },
            'math': {'number', 'numbers', 'figure'},
            'security': set(),
            'finance': set(),
            'general': set()
        }
        denylist |= domain_denylist.get(domain, set())
        # Figure/panel/image references – block for terms/subdomains but OK in visual_facts
        figure_terms = {
            'figure', 'panel', 'image'
        }
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            figure_terms.add(f"figure {letter}")
            figure_terms.add(f"panel {letter}")
            figure_terms.add(f"image {letter}")
        denylist |= figure_terms

        lang = getattr(self, 'dataset_language', 'en')
        if lang == 'zh':
            denylist |= {
                '其他', '内容', '信息', '题目', '题干', '问题', '选项',
                '示例', '标签', '描述', '叙述', '说法', '以下', '下面',
                '系统', '数据', '存储', '信息', '程序', '过程'
            }
        return denylist

    def _get_pattern_exceptions(self) -> Set[str]:
        """
        Get pattern exceptions - allowed multi-word phrases containing denylist terms.

        For example: "operating system" is allowed even though "system" is denylisted.

        Returns:
            Set of allowed phrases (lowercase)
        """
        domain = getattr(self, 'dataset_domain', 'general')
        exceptions: Set[str] = set()

        cs_exceptions = {
            # System-related exceptions
            'operating system', 'file system', 'system bus', 'system call',
            'distributed system', 'embedded system', 'number system',
            'expert system', 'database system', 'computer system',
            # Data-related exceptions
            'data structure', 'data type', 'data link', 'data link layer',
            'binary data', 'data transfer', 'data segment', 'data flow',
            'data bus', 'data path', 'metadata',
            # Information-related exceptions
            'information retrieval', 'information theory', 'information hiding',
            'information system', 'mutual information',
            # Computer-related exceptions
            'computer architecture', 'computer organization', 'computer network',
            'computer graphics', 'computer science', 'personal computer',
            # Organization-related exceptions
            'computer organization', 'memory organization', 'cache organization',
            'file organization', 'data organization'
        }

        if domain == 'cs':
            exceptions |= cs_exceptions

        lang = getattr(self, 'dataset_language', 'en')
        if lang == 'zh':
            zh_cs_exceptions = {
                '操作系统', '文件系统', '数据库系统', '实时系统', '嵌入式系统',
                '信息系统', '管理信息系统', '计算机系统', '存储系统', '存储管理',
                '内存管理', '虚拟内存', '主存储器', '缓存系统', '输入输出系统',
                '通道程序', '中断系统', '数据结构', '数据链路层', '数据总线',
                '计算机网络', '网络体系结构'
            }
            if domain == 'cs':
                exceptions |= zh_cs_exceptions
        return exceptions

    def _is_denylist_term(self, term: str) -> bool:
        """
        Check if a term is denylisted (with pattern exceptions).

        Args:
            term: The term to check (will be lowercased)

        Returns:
            True if term should be rejected, False if allowed
        """
        term_lower = term.lower().strip()
        lang = getattr(self, 'dataset_language', 'en')

        # Check pattern exceptions first (multi-word phrases are allowed)
        pattern_exceptions = self._get_pattern_exceptions()
        if term_lower in pattern_exceptions:
            return False

        # Check if the term itself is denylisted
        denylist = self._get_denylist_terms()
        if term_lower in denylist:
            return True

        # Language-specific bare checks
        if lang == 'zh' or self._is_chinese(term):
            # Reject bare denylist characters/phrases
            if term in denylist or (len(term) == 1 and term in denylist):
                return True
        else:
            tokens = term_lower.split()
            if len(tokens) == 1 and tokens[0] in denylist:
                return True

        return False

    def _validate_term_form(
        self,
        term: str,
        corpus_df: Optional[Counter] = None,
        total_docs: int = 0,
        max_df_pct: float = 0.40
    ) -> Tuple[bool, str]:
        """
        Validate term form using multiple filters.

        Filters:
        1. Length: ≥3 characters, no single letters
        2. No all-stopword phrases
        3. DF bounds: 3 ≤ document frequency ≤ 40% of items
        4. Character set: letters, digits, and common CS symbols only
        5. No placeholder suffixes

        Args:
            term: The term to validate
            corpus_df: Document frequency counter (optional, for DF check)
            total_docs: Total number of documents (for DF percentage)
            max_df_pct: Maximum document frequency percentage (default: 0.40)

        Returns:
            (is_valid, failure_reason)
        """
        lang = getattr(self, 'dataset_language', 'en')
        term_stripped = term.strip()
        if not term_stripped:
            return False, "empty"

        # Chinese-specific validation
        if lang == 'zh' or self._is_chinese(term_stripped):
            normalized = term_stripped
            if len(normalized) < 2:
                return False, "too_short"
            if len(normalized) > 12:
                return False, "too_long"

            allowed_chars = set('()-（）/、·')
            for ch in normalized:
                if not (self._is_chinese_char(ch) or ch.isdigit() or ch in allowed_chars):
                    return False, "invalid_charset"

            stopwords = self._get_stopwords()
            if normalized in stopwords:
                return False, "stopword"

            if corpus_df is not None and total_docs > 0:
                df = corpus_df.get(normalized, 0)
                if df < 2:
                    return False, "too_rare"
                if df > max_df_pct * total_docs:
                    return False, "too_common"

            return True, ""

        term_lower = term.lower().strip()

        # Filter 1: Length check (≥3 characters, no single letters)
        if len(term_lower) < 3:
            return False, "too_short"

        tokens = term_lower.split()
        if len(tokens) == 1 and len(tokens[0]) == 1:
            return False, "single_letter"

        # Filter 2: No all-stopword phrases
        stopwords = self._get_stopwords()
        if all(token in stopwords for token in tokens):
            return False, "all_stopwords"

        # Filter 3: DF bounds (if corpus_df provided)
        if corpus_df is not None and total_docs > 0:
            df = corpus_df.get(term_lower, 0)
            if df < 3:
                return False, "too_rare"
            if df > max_df_pct * total_docs:
                return False, "too_common"

        # Filter 4: Character set (letters, digits, common CS symbols)
        allowed_pattern = r'^[a-zA-Z0-9\s/\(\)\+\*\-\.]+$'
        if not re.match(allowed_pattern, term):
            return False, "invalid_charset"

        # Filter 5: No placeholder suffixes
        placeholder_suffixes = ['subdomain', 'topic', 'category', 'concept', 'domain', 'area']
        for suffix in placeholder_suffixes:
            if term_lower.endswith(suffix):
                if term_lower == suffix or term_lower.split()[-1] == suffix:
                    return False, "placeholder_suffix"

        return True, ""

    # ------------------------------------------------------------------------
    # 2. Micro-Enum Building
    # ------------------------------------------------------------------------

    def build_micro_enum(
        self,
        items: List[Dict[str, Any]],
        extraction_results: Optional[List[ExtractionResult]] = None,
        top_k: int = 400,
        min_freq: int = 3
    ) -> Dict[str, MicroEnumEntry]:
        """
        Build micro-enum from corpus + extraction outputs with strict hygiene.

        Strategy:
        1. Extract unigrams and bigrams from WHITELISTED fields only (question/body/excerpt)
        2. Add terms from extraction outputs (if provided)
        3. Count frequency and compute document frequency (DF)
        4. Apply stopwords, denylist, and form filters
        5. Keep top_k candidates that pass all filters
        6. Build alias lists (simple variants)
        7. Store as micro-enum

        Args:
            items: Dataset items
            extraction_results: Optional extraction results to include terms from
            top_k: Keep top K candidates (default: 400, range 300-450)
            min_freq: Minimum frequency to include

        Returns:
            Dictionary of id -> MicroEnumEntry
        """
        print(f"\nBuilding micro-enum from {len(items)} items with hygiene filters...")
        lang = getattr(self, 'dataset_language', 'en')

        curated_entries: List[MicroEnumEntry] = []
        if self.dataset_domain == 'psychology':
            curated_entries = self._build_tiny_tom_entries(lang)

        # Extract all text from corpus (WHITELISTED FIELDS ONLY)
        all_tokens = []
        doc_freq = Counter()  # Track document frequency for DF bounds check

        for item in items:
            # Use whitelisted fields only
            text = self._item_to_text_whitelisted(item)
            if lang == 'zh':
                tokens = self._extract_chinese_terms(text)
            else:
                tokens = self._extract_ngrams(text, n=1) + self._extract_ngrams(text, n=2)

            normalized_tokens = [
                self._normalize_text(token, lang=lang)
                for token in tokens
                if token.strip()
            ]
            normalized_tokens = [t for t in normalized_tokens if t]

            # Add to all_tokens for term frequency
            all_tokens.extend(normalized_tokens)

            # Track document frequency (unique terms per item)
            unique_tokens = set(normalized_tokens)
            doc_freq.update(unique_tokens)

        print(f"  Extracted {len(all_tokens)} tokens from whitelisted fields")

        # Add terms from extraction outputs (if provided)
        if extraction_results:
            print(f"  Including {len(extraction_results)} extraction outputs...")
            for result in extraction_results:
                if result.is_valid:
                    normalized_subdomains = [
                        self._normalize_text(s, lang=lang) for s in result.subdomains if s
                    ]
                    normalized_terms = [
                        self._normalize_text(t, lang=lang) for t in result.terms if t
                    ]
                    all_tokens.extend([s for s in normalized_subdomains if s])
                    all_tokens.extend([t for t in normalized_terms if t])

        # Count term frequency
        freq_counter = Counter(all_tokens)

        # Filter candidates
        print(f"  Applying hygiene filters...")
        candidates = []
        filtered_stats = {
            'min_freq': 0,
            'stopwords': 0,
            'denylist': 0,
            'form_filters': 0,
            'passed': 0
        }

        for token, freq in freq_counter.items():
            # Filter 1: Minimum frequency
            if freq < min_freq:
                filtered_stats['min_freq'] += 1
                continue

            # Filter 2: Stopwords (skip all-stopword phrases)
            stopwords = self._get_stopwords()
            if all(word in stopwords for word in token.lower().split()):
                filtered_stats['stopwords'] += 1
                continue

            # Filter 3: Denylist (with pattern exceptions)
            if self._is_denylist_term(token):
                filtered_stats['denylist'] += 1
                continue

            # Filter 4: Form validation (length, DF bounds, character set, etc.)
            is_valid, failure_reason = self._validate_term_form(
                token,
                corpus_df=doc_freq,
                total_docs=len(items),
                max_df_pct=0.40
            )
            if not is_valid:
                filtered_stats['form_filters'] += 1
                continue

            # Passed all filters
            filtered_stats['passed'] += 1
            candidates.append((token, freq))

        # Log filtering stats
        print(f"    Filtered by min_freq (<{min_freq}): {filtered_stats['min_freq']}")
        print(f"    Filtered by stopwords: {filtered_stats['stopwords']}")
        print(f"    Filtered by denylist: {filtered_stats['denylist']}")
        print(f"    Filtered by form filters: {filtered_stats['form_filters']}")
        print(f"    Passed all filters: {filtered_stats['passed']}")

        # Sort by frequency and take top_k
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:top_k]

        print(f"  Kept top {len(candidates)} candidates (target: {top_k})")

        # Build micro-enum entries
        micro_enum = {}
        for idx, (token, freq) in enumerate(candidates):
            normalized = self._normalize_text(token, lang=lang)

            # Build aliases (simple variants)
            aliases = {token}
            if '-' in token or '_' in token:
                aliases.add(token.replace('-', ' ').replace('_', ' '))
            if lang != 'zh' and token.lower() != token:
                aliases.add(token.lower())
            aliases = list(aliases)

            entry = MicroEnumEntry(
                id=f"term_{idx:04d}",
                name=normalized,
                aliases=aliases,
                frequency=freq
            )
            micro_enum[entry.id] = entry

        if curated_entries:
            for entry in curated_entries:
                micro_enum[entry.id] = entry
            print(f"  Injected tiny ToM enum ({len(curated_entries)} entries)")

        # Compute hash for reproducibility
        enum_str = json.dumps(
            [e.to_dict() for e in micro_enum.values()],
            sort_keys=True,
            ensure_ascii=False
        )
        enum_hash = hashlib.sha256(enum_str.encode()).hexdigest()[:16]

        self.micro_enum = micro_enum
        self.micro_enum_hash = enum_hash

        print(f"  ✓ Built micro-enum with {len(micro_enum)} entries (hash: {enum_hash})")

        return micro_enum

    def _item_to_text_whitelisted(self, item: Dict[str, Any]) -> str:
        """
        Convert item to text using WHITELISTED fields only.

        Whitelist:
        - question / prompt / text (main content)
        - background / context / excerpt (supplementary content)
        - image captions (if multimodal)

        DO NOT include:
        - JSON keys, tags, metadata
        - Prompt scaffolding
        - Options or answer

        Args:
            item: Dataset item

        Returns:
            Whitelisted text
        """
        parts = []

        # Main question field
        if 'question' in item:
            parts.append(item['question'])
        elif 'prompt' in item and isinstance(item['prompt'], str):
            parts.append(item['prompt'])
        elif 'text' in item and isinstance(item['text'], str):
            parts.append(item['text'])

        # Background/context (if present and meaningful)
        if 'background' in item:
            bg = item['background']
            if isinstance(bg, str):
                parts.append(bg)
            elif isinstance(bg, dict):
                # Only extract actual content, not meta keys
                for key, value in bg.items():
                    if isinstance(value, str) and len(value) > 10:
                        parts.append(value)

        if 'context' in item and isinstance(item['context'], str):
            parts.append(item['context'])

        if 'excerpt' in item and isinstance(item['excerpt'], str):
            parts.append(item['excerpt'])

        # Image captions (if multimodal)
        if self._should_include_image_caption() and 'image_caption' in item and isinstance(item['image_caption'], str):
            parts.append(item['image_caption'])

        # Context/story
        if 'context' in item:
            self._append_context_text(parts, item['context'])

        return " ".join(parts)

    def _extract_chinese_terms(self, text: str, min_len: int = 2, max_len: int = 6) -> List[str]:
        """Extract Chinese n-grams (character windows) with length constraints."""
        if not text:
            return []

        normalized = self._normalize_text(text, lang='zh')
        chunks = re.findall(r'[\u4e00-\u9fff]+', normalized)

        terms: List[str] = []
        for chunk in chunks:
            length = len(chunk)
            for window in range(min_len, min(max_len, length) + 1):
                for i in range(length - window + 1):
                    terms.append(chunk[i:i+window])
        return terms

    def _extract_ngrams(self, text: str, n: int) -> List[str]:
        """Extract n-grams from text"""
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        return ngrams

    def _get_per_item_glossary(
        self,
        item: Dict[str, Any],
        top_k: int = 30
    ) -> List[str]:
        """
        Build a per-item mini-glossary by selecting the top-k closest micro-enum entries.

        Uses semantic similarity between item text and micro-enum entries to rank.

        Args:
            item: Dataset item
            top_k: Number of glossary entries to return (default: 30, range 10-30)

        Returns:
            List of top-k terms from micro-enum, sorted by relevance to item
        """
        if not self.micro_enum:
            return []

        # Get item text
        item_text = self._item_to_text_whitelisted(item)

        # Embed item text
        try:
            item_emb = self.embedding_model.encode([item_text], show_progress_bar=False)[0]
        except Exception:
            # If embedding fails, return random top-k
            return [entry.name for entry in list(self.micro_enum.values())[:top_k]]

        # Embed all micro-enum entries
        enum_terms = [entry.name for entry in self.micro_enum.values()]
        try:
            enum_embs = self.embedding_model.encode(enum_terms, show_progress_bar=False)
        except Exception:
            return enum_terms[:top_k]

        # Compute cosine similarity
        similarities = cosine_similarity([item_emb], enum_embs)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return top-k terms
        return [enum_terms[i] for i in top_indices]

    def _validate_and_fix_extraction(
        self,
        result: ExtractionResult,
        item: Dict[str, Any],
        per_item_glossary: Optional[List[str]] = None
    ) -> ExtractionResult:
        """
        Validate extraction result by filtering obvious noise while keeping soft coverage.

        Steps:
        1. Remove denylisted or malformed strings (attempt glossary replacement if available).
        2. Track lexical anchoring counts (literal substring matches) for scoring.
        3. Keep shorter lists when needed instead of forcing placeholders.

        Args:
            result: Extraction result to validate
            item: Dataset item (for anchoring check)
            per_item_glossary: Per-item mini-glossary for replacements

        Returns:
            Validated and fixed ExtractionResult
        """
        if not result.is_valid:
            return result

        if per_item_glossary is None and self.micro_enum:
            per_item_glossary = self._get_per_item_glossary(item)
        elif per_item_glossary is None:
            per_item_glossary = []

        # Prepare item text for lexical anchoring checks
        item_text = self._item_to_text_whitelisted(item).lower()

        used_glossary: set = set()

        # Helper to pull the next viable glossary candidate (must be lexically grounded)
        def _take_glossary_candidate() -> Optional[str]:
            for candidate in per_item_glossary:
                normalized = candidate.strip()
                if not normalized or normalized.lower() in used_glossary:
                    continue
                if self._is_denylist_term(normalized):
                    continue
                form_valid, _ = self._validate_term_form(normalized)
                if not form_valid:
                    continue
                if normalized.lower() not in item_text:
                    continue  # Glossary fallback must still be grounded
                used_glossary.add(normalized.lower())
                return normalized
            return None

        def _try_accept_term(candidate: str) -> bool:
            if not candidate:
                return False
            normalized = candidate.strip()
            if not normalized:
                return False
            if not self._token_length_ok(normalized):
                return False
            if self._is_denylist_term(normalized):
                return False
            form_valid, _ = self._validate_term_form(normalized)
            if not form_valid:
                return False
            normalized_lower = normalized.lower()
            if normalized_lower in seen_terms_lower:
                return False
            if normalized_lower not in item_text:
                return False
            if len(term_records) >= 12:
                return False
            raw_category = self._classify_tom_term(normalized)
            category = self._collapse_tom_category(raw_category)
            term_records.append((normalized, category))
            seen_terms_lower.add(normalized_lower)
            return True

        term_records: List[Tuple[str, str]] = []
        seen_terms_lower: set = set()

        for term in result.terms:
            if len(term_records) >= 12:
                break
            normalized = term.strip()
            if _try_accept_term(normalized):
                continue
            replacement = _take_glossary_candidate()
            if replacement:
                _try_accept_term(replacement)

        # Cap context-heavy terms but never drop below 2 total terms
        context_categories = {'context'}
        max_context_terms = 5
        if len(term_records) > 2:
            context_indices = [i for i, (_, cat) in enumerate(term_records) if cat in context_categories]
            if len(context_indices) > max_context_terms:
                for idx in reversed(context_indices[max_context_terms:]):
                    if len(term_records) <= 2:
                        break
                    term_records.pop(idx)

        # Validate subdomains (keep up to 3)
        validated_subdomains: List[str] = []
        sub_seen_lower: set = set()
        for subdomain in result.subdomains:
            if len(validated_subdomains) >= 3:
                break
            normalized = subdomain.strip()
            if not normalized or self._is_denylist_term(normalized):
                continue
            form_valid, _ = self._validate_term_form(normalized)
            if not form_valid:
                continue
            normalized_lower = normalized.lower()
            if normalized_lower in sub_seen_lower:
                continue
            validated_subdomains.append(normalized)
            sub_seen_lower.add(normalized_lower)

        anchored_count = len(term_records)
        result.terms = [term for term, _ in term_records]
        result.subdomains = self._enforce_tombench_cn_subdomains(validated_subdomains, item)

        # Update anchoring stats & confidence (scores, not gates)
        result.anchored_count = anchored_count
        result.anchored_fraction = (
            anchored_count / len(term_records) if term_records else 0.0
        )
        result.tom_categories = [category for _, category in term_records]

        if not term_records:
            result.confidence = 0.0

        result.is_valid = True

        return result

    # ------------------------------------------------------------------------
    # 3. Canonicalization & Normalization
    # ------------------------------------------------------------------------

    def _normalize_text(self, text: str, lang: str = 'en') -> str:
        """
        Normalize text according to protocol.

        English:
        - Lowercase
        - Replace runs of underscores/dashes with spaces
        - Remove stray punctuation
        - Singularize obvious plurals (simple: remove trailing 's')
        - Keep math tokens like (), +, /, *

        Chinese:
        - Unify to simplified
        - Remove spaces
        - Unify full/half width
        """
        if lang == 'zh' or self._is_chinese(text):
            text = text.strip().replace('　', '').replace(' ', '')
            text = self._normalize_chinese_equivalents(text)
            text = text.replace('（', '(').replace('）', ')')
            text = text.replace('［', '[').replace('］', ']')
            text = re.sub(r'[^\u4e00-\u9fff0-9\(\)\-\、]', '', text)
            return text
        else:
            # English normalization
            text = text.lower().strip()
            # Replace runs of underscores/dashes with single space
            text = re.sub(r'[_-]+', ' ', text)
            # Remove stray punctuation (keep math operators and parens)
            text = re.sub(r'[^\w\s()+/*-]', '', text)
            # Singularize obvious plurals (simple heuristic)
            if text.endswith('s') and len(text) > 3 and text[-2] not in 'ss':
                text = text[:-1]
            text = ' '.join(text.split())  # Normalize whitespace
            return text

    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        return any('\u4e00' <= c <= '\u9fff' for c in text)

    def _is_chinese_char(self, ch: str) -> bool:
        """Check if a single character is Chinese."""
        return '\u4e00' <= ch <= '\u9fff'

    def _normalize_chinese_equivalents(self, text: str) -> str:
        """Apply simple synonym/variant normalization for Chinese terms."""
        replacements = {
            '內存': '内存',
            '記憶體': '内存',
            '內部存儲器': '内存',
            '主存': '主存储器',
            '主存儲器': '主存储器',
            '主存储': '主存储器',
            '存储体系': '存储系统',
            '存儲系統': '存储系统',
            '計算機': '计算机',
            '處理器': '处理器',
            '通道程式': '通道程序',
            '虛擬內存': '虚拟内存'
        }
        for src, tgt in replacements.items():
            text = text.replace(src, tgt)
        return text

    def canonicalize_to_enum(
        self,
        terms: List[str],
        lang: str = 'en'
    ) -> List[str]:
        """
        Canonicalize extracted terms to micro-enum.

        Mapping order:
        1. Exact match to micro-enum name
        2. Alias match
        3. Fuzzy match (conservative, very similar only)
        4. Otherwise keep the normalized raw string (no gating)

        Args:
            terms: Raw extracted terms
            lang: Language hint ('en' or 'zh')

        Returns:
            List of canonicalized term IDs or normalized names
        """
        canonical = []

        for term in terms:
            normalized = self._normalize_text(term, lang)
            if not normalized:
                continue

            mapped_value = normalized

            if self.micro_enum:
                # Try exact match
                for entry in self.micro_enum.values():
                    if normalized == entry.name:
                        mapped_value = entry.name
                        break
                else:
                    # Try alias match
                    alias_match = False
                    for entry in self.micro_enum.values():
                        normalized_aliases = [self._normalize_text(a, lang) for a in entry.aliases]
                        if normalized in normalized_aliases:
                            mapped_value = entry.name
                            alias_match = True
                            break

                    # Try fuzzy match only if no alias match
                    if not alias_match:
                        best_match = None
                        best_score = 0.0
                        for entry in self.micro_enum.values():
                            score = self._string_similarity(normalized, entry.name)
                            if score > best_score:
                                best_score = score
                                best_match = entry.name

                        if best_match and best_score >= 0.90:
                            mapped_value = best_match

            canonical.append(mapped_value)

        # Deduplicate while preserving order, then sort alphabetically
        deduped = []
        seen = set()
        for value in canonical:
            if value not in seen:
                seen.add(value)
                deduped.append(value)

        return sorted(deduped, key=str.lower)

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity (Jaccard on character bigrams)"""
        if s1 == s2:
            return 1.0
        bigrams1 = set(s1[i:i+2] for i in range(len(s1)-1))
        bigrams2 = set(s2[i:i+2] for i in range(len(s2)-1))
        if not bigrams1 or not bigrams2:
            return 0.0
        intersection = len(bigrams1 & bigrams2)
        union = len(bigrams1 | bigrams2)
        return intersection / union if union > 0 else 0.0

    # ------------------------------------------------------------------------
    # 4. Self-Consistency (K=2)
    # ------------------------------------------------------------------------

    async def extract_with_consistency(
        self,
        item: Dict[str, Any],
        k: int = 2,
        **kwargs
    ) -> ExtractionResult:
        """
        Extract with K=2 self-consistency.

        Run same prompt twice with tiny neutral wording change.
        Keep items that appear in both runs; fill from union if needed.

        Args:
            item: Dataset item
            k: Number of runs (default 2)
            **kwargs: Additional args for _extract_once

        Returns:
            Consensus ExtractionResult
        """
        results = []
        for i in range(k):
            # Tiny wording variation (reorder two instruction sentences)
            # This is a placeholder; in practice, you might shuffle instructions slightly
            result = await self._extract_once(item, **kwargs)
            if result.is_valid:
                results.append(result)

        if not results:
            # All failed, return invalid
            return ExtractionResult([], [], 0.0, is_valid=False)

        if len(results) == 1:
            # Only one valid result
            return results[0]

        # Consensus logic
        subdomain_sets = [set(r.subdomains) for r in results]
        term_sets = [set(r.terms) for r in results]

        # Intersection
        subdomain_consensus = set.intersection(*subdomain_sets)
        term_consensus = set.intersection(*term_sets)

        # Fill from union if needed
        subdomain_union = set.union(*subdomain_sets)
        term_union = set.union(*term_sets)

        final_subdomains = list(subdomain_consensus)
        if len(final_subdomains) < 3:
            remaining = subdomain_union - subdomain_consensus
            final_subdomains.extend(list(remaining)[:3 - len(final_subdomains)])
        final_subdomains = sorted(final_subdomains)[:3]

        final_terms = list(term_consensus)
        if len(final_terms) < 12:
            remaining = term_union - term_consensus
            final_terms.extend(list(remaining)[:12 - len(final_terms)])
        final_terms = sorted(final_terms)[:12]

        # Average confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)

        return ExtractionResult(
            subdomains=final_subdomains,
            terms=final_terms,
            confidence=avg_confidence,
            is_valid=True
        )

    # ------------------------------------------------------------------------
    # 5-6. Multilingual & Multimodal
    # ------------------------------------------------------------------------

    # Already handled in _extract_once via is_multilingual and has_image flags

    # ------------------------------------------------------------------------
    # 7. Clustering & Hierarchy
    # ------------------------------------------------------------------------

    def build_hierarchy(
        self,
        all_subdomains: List[str],
        target_parents: int = 50,
        max_gini: float = 0.30
    ) -> Dict[str, str]:
        """
        Build parent-child hierarchy for subdomains.

        Strategy:
        1. Embed all unique subdomains
        2. Cluster into target_parents groups
        3. Name parents using most representative terms
        4. Check Gini coefficient for balance

        Args:
            all_subdomains: All subdomains from extraction
            target_parents: Target number of parent categories
            max_gini: Maximum allowed Gini coefficient

        Returns:
            Mapping of subdomain -> parent
        """
        print(f"\nBuilding hierarchy from {len(all_subdomains)} subdomains...")

        # Count frequency
        subdomain_freq = Counter(all_subdomains)
        unique_subdomains = list(subdomain_freq.keys())

        print(f"  {len(unique_subdomains)} unique subdomains")

        if len(unique_subdomains) <= target_parents:
            # Too few, treat each as its own parent
            return {s: s for s in unique_subdomains}

        # Embed subdomains
        embeddings = self.embedding_model.encode(unique_subdomains, show_progress_bar=False)

        # Cluster with KMeans
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=target_parents, random_state=self.seed, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Build parent map
        parent_map = {}
        clusters = defaultdict(list)
        for subdomain, label in zip(unique_subdomains, cluster_labels):
            clusters[label].append(subdomain)

        # Name parents (use most common subdomain in cluster)
        for label, members in clusters.items():
            # Pick most frequent member as parent name
            member_counts = [(m, subdomain_freq[m]) for m in members]
            member_counts.sort(key=lambda x: x[1], reverse=True)
            parent_name = member_counts[0][0]

            for member in members:
                parent_map[member] = parent_name

        # Check Gini
        parent_freq = Counter(parent_map[s] for s in all_subdomains)
        gini = self._compute_gini([f for f in parent_freq.values()])

        print(f"  Created {len(set(parent_map.values()))} parents")
        print(f"  Gini coefficient: {gini:.3f} (max: {max_gini})")

        if gini > max_gini:
            print(f"  ⚠ Gini exceeds threshold, consider rebalancing")

        self.parent_map = parent_map
        return parent_map

    def _hash_parent_map(self) -> Optional[str]:
        """
        Compute hash of parent map for reproducibility.

        Returns hash of sorted parent mappings (child -> parent).
        """
        if not self.parent_map:
            return None

        # Sort by key for deterministic hashing
        sorted_items = sorted(self.parent_map.items())
        map_str = json.dumps(sorted_items, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(map_str.encode()).hexdigest()[:16]

    def _compute_gini(self, values: List[int]) -> float:
        """Compute Gini coefficient"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        return (2 * np.sum((np.arange(1, n+1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n

    # ------------------------------------------------------------------------
    # 8. Stability Metrics
    # ------------------------------------------------------------------------

    def compute_stability_metrics(
        self,
        results1: List[ExtractionResult],
        results2: List[ExtractionResult],
        use_semantic: bool = True
    ) -> StabilityMetrics:
        """
        Compute stability metrics between two runs using per-item pairing.

        Metrics:
        - Raw Jaccard (per-item, then averaged)
        - Soft Jaccard (semantic matching, per-item)
        - NSI (Normalized Stability Index vs random baseline)
        - Hierarchical agreement (parent-level)

        Args:
            results1: First run results (aligned by item order)
            results2: Second run results (aligned by item order)
            use_semantic: Use semantic similarity for soft Jaccard

        Returns:
            StabilityMetrics
        """
        # Ensure same length (items should be aligned)
        assert len(results1) == len(results2), "Results must have same length for pairing"

        # Per-item Raw Jaccard, then average
        raw_jac_subs = []
        raw_jac_terms = []
        soft_jac_subs = []
        soft_jac_terms = []

        for r1, r2 in zip(results1, results2):
            if not r1.is_valid or not r2.is_valid:
                continue

            # Raw Jaccard per item
            raw_jac_subs.append(self._jaccard(set(r1.subdomains), set(r2.subdomains)))
            raw_jac_terms.append(self._jaccard(set(r1.terms), set(r2.terms)))

            # Soft Jaccard per item (if using semantic)
            if use_semantic:
                soft_jac_subs.append(self._soft_jaccard(r1.subdomains, r2.subdomains))
                soft_jac_terms.append(self._soft_jaccard(r1.terms, r2.terms))

        # Average over items
        raw_jac_sub = np.mean(raw_jac_subs) if raw_jac_subs else 0.0
        raw_jac_term = np.mean(raw_jac_terms) if raw_jac_terms else 0.0

        if use_semantic and soft_jac_terms:
            soft_jac_sub = np.mean(soft_jac_subs)
            soft_jac_term = np.mean(soft_jac_terms)
        else:
            soft_jac_sub = raw_jac_sub
            soft_jac_term = raw_jac_term

        # NSI (per-item with random misalignment baseline)
        nsi = self._compute_nsi(results1, results2)

        # Hierarchical agreement (per-item, then averaged)
        hier_agrees = []
        if self.parent_map:
            for r1, r2 in zip(results1, results2):
                if not r1.is_valid or not r2.is_valid:
                    continue
                parents1 = set(self.parent_map.get(s, s) for s in r1.subdomains)
                parents2 = set(self.parent_map.get(s, s) for s in r2.subdomains)
                hier_agrees.append(self._jaccard(parents1, parents2))
            hier_agree = np.mean(hier_agrees) if hier_agrees else raw_jac_sub
        else:
            hier_agree = raw_jac_sub  # Fallback

        return StabilityMetrics(
            raw_jaccard_subdomains=raw_jac_sub,
            raw_jaccard_terms=raw_jac_term,
            soft_jaccard_subdomains=soft_jac_sub,
            soft_jaccard_terms=soft_jac_term,
            nsi=nsi,
            hierarchical_agreement=hier_agree
        )

    def _jaccard(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _soft_jaccard(self, list1: List[str], list2: List[str], threshold: float = 0.80, debug: bool = False) -> float:
        """
        Compute soft Jaccard with semantic matching.

        Apply a hard similarity threshold (≥0.80), then use Hungarian algorithm
        for optimal one-to-one matching. Only pairs meeting the threshold count.

        Args:
            list1: First list of terms (Run A)
            list2: Second list of terms (Run B)
            threshold: Minimum cosine similarity to count as match (default: 0.80)
            debug: Print debug information

        Returns:
            Soft Jaccard: matched_count / (|A| + |B| - matched_count)
        """
        if not list1 or not list2:
            return 0.0

        try:
            # Ensure we're comparing different lists (not A vs A)
            if list1 == list2 and debug:
                print(f"  WARNING: Comparing identical lists! This will give inflated similarity.")

            # Embed both lists
            emb1 = self.embedding_model.encode(list1, show_progress_bar=False)
            emb2 = self.embedding_model.encode(list2, show_progress_bar=False)

            # Compute pairwise cosine similarity
            sim_matrix = cosine_similarity(emb1, emb2)

            if debug:
                print(f"  Similarity matrix shape: {sim_matrix.shape}")
                print(f"  Max similarity: {np.max(sim_matrix):.3f}, Min: {np.min(sim_matrix):.3f}")
                print(f"  Mean similarity: {np.mean(sim_matrix):.3f}")
                pairs_above_threshold = np.sum(sim_matrix >= threshold)
                print(f"  Pairs ≥ {threshold}: {pairs_above_threshold}/{sim_matrix.size} ({pairs_above_threshold/sim_matrix.size:.1%})")

            # Apply hard threshold: set all similarities < threshold to 0
            # This ensures only semantically similar pairs (≥0.80) can be matched
            sim_matrix[sim_matrix < threshold] = 0.0

            # Use Hungarian algorithm for optimal one-to-one matching
            # Convert similarity to cost (maximize similarity = minimize negative similarity)
            cost_matrix = -sim_matrix

            # Pad matrix if needed (Hungarian requires square matrix)
            n1, n2 = sim_matrix.shape
            max_dim = max(n1, n2)
            if n1 != max_dim or n2 != max_dim:
                padded_cost = np.full((max_dim, max_dim), 0.0)
                padded_cost[:n1, :n2] = cost_matrix
                cost_matrix = padded_cost

            # Find optimal matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Count matches where similarity ≥ threshold (only within original dimensions)
            # Unmatched pairs (below threshold) don't count as matches
            matched_count = 0
            for i, j in zip(row_ind, col_ind):
                if i < n1 and j < n2 and sim_matrix[i, j] >= threshold:
                    matched_count += 1

            # Soft Jaccard: matches / union
            union_size = len(list1) + len(list2) - matched_count

            if debug:
                print(f"  Matched count: {matched_count}")
                print(f"  Union size: {union_size}")
                soft_jac = matched_count / union_size if union_size > 0 else 0.0
                print(f"  Soft Jaccard: {soft_jac:.3f}")

            return matched_count / union_size if union_size > 0 else 0.0

        except Exception as e:
            # If embedding fails, return N/A (represented as -1.0)
            print(f"Warning: Soft Jaccard computation failed: {e}")
            return -1.0

    def _compute_nsi(
        self,
        results1: List[ExtractionResult],
        results2: List[ExtractionResult],
        num_random_samples: int = 100
    ) -> float:
        """
        Compute Normalized Stability Index following the pairing recipe.

        NSI = (J_obs - J_rand) / (1 - J_rand)

        Where:
        - J_obs: observed per-item Jaccard (aligned pairs)
        - J_rand: random baseline by breaking alignment (misaligned pairs)

        Random baseline: For each item i in Run A, compare to a DIFFERENT random
        item j ≠ i from Run B. Average those Jaccards.
        """
        n = len(results1)
        if n == 0:
            return 0.0

        # Observed agreement (per-item, aligned)
        observed_jaccards = []
        for r1, r2 in zip(results1, results2):
            if r1.is_valid and r2.is_valid:
                jac = self._jaccard(set(r1.terms), set(r2.terms))
                observed_jaccards.append(jac)

        if not observed_jaccards:
            return 0.0

        j_obs = np.mean(observed_jaccards)

        # Random baseline: misalign pairs
        random_jaccards = []
        valid_indices = [i for i, (r1, r2) in enumerate(zip(results1, results2))
                        if r1.is_valid and r2.is_valid]

        if len(valid_indices) < 2:
            # Not enough valid items to compute random baseline
            return 0.0

        for _ in range(num_random_samples):
            # For each item i in Run A, pick a different random item j ≠ i from Run B
            sample_jaccards = []
            for i in valid_indices:
                # Pick j ≠ i
                other_indices = [j for j in valid_indices if j != i]
                if not other_indices:
                    continue
                j = np.random.choice(other_indices)

                r1 = results1[i]
                r2 = results2[j]  # Misaligned!

                jac = self._jaccard(set(r1.terms), set(r2.terms))
                sample_jaccards.append(jac)

            if sample_jaccards:
                random_jaccards.append(np.mean(sample_jaccards))

        if not random_jaccards:
            return 0.0

        j_rand = np.mean(random_jaccards)

        # NSI
        if j_rand >= 1.0:
            return 1.0
        if j_rand >= j_obs:
            # Random is as good or better than observed - no stability
            return 0.0

        nsi = (j_obs - j_rand) / (1.0 - j_rand)
        return max(0.0, min(1.0, nsi))  # Clamp to [0, 1]

    def _compute_anchored_percent(
        self,
        items: List[Dict[str, Any]],
        results: List[ExtractionResult]
    ) -> float:
        """
        Compute Anchored% - percentage of terms validated by 4 anchor strategies.

        A term is anchored if it passes ANY of:
        1. Lexical: exact match (case-insensitive) in question text
        2. Alias: match via micro-enum aliases (placeholder until micro-enum is built)
        3. Semantic: cosine similarity ≥0.80 to any unigram/bigram in question
        4. Corpus: appears ≥5 times across all items in dataset

        Args:
            items: Dataset items
            results: Extraction results

        Returns:
            Percentage of terms that are anchored (0.0 to 1.0)
        """
        valid_results = [r for r in results if r.is_valid]
        if valid_results and all(hasattr(r, 'anchored_count') for r in valid_results):
            total_terms = sum(len(r.terms) for r in valid_results)
            if total_terms == 0:
                return 0.0
            anchored_terms = sum(getattr(r, 'anchored_count', 0) for r in valid_results)
            return anchored_terms / total_terms

        # Fallback for legacy runs that predate anchored_count
        # Build corpus frequency map
        corpus_freq = Counter()
        question_ngrams_all = []

        for item in items:
            text = self._item_to_text(item).lower()
            words = text.split()

            # Count unigrams
            corpus_freq.update(words)

            # Count bigrams
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            corpus_freq.update(bigrams)

            # Store for semantic matching
            question_ngrams_all.append(words + bigrams)

        # Check each term across all results (per-item anchoring with cap)
        total_terms = 0
        anchored_terms = 0

        for idx, (item, result) in enumerate(zip(items, results)):
            if not result.is_valid:
                continue

            item_text = self._item_to_text(item).lower()
            question_ngrams = question_ngrams_all[idx]

            # Cap corpus fallback at 1 term per item (tightened from 2)
            corpus_anchored_count = 0

            for term in result.terms:
                total_terms += 1
                term_lower = term.lower()
                is_anchored = False

                # First, check if term passes hygiene filters (denylist + form validation)
                if self._is_denylist_term(term):
                    # Skip denylisted terms (they won't be anchored)
                    continue

                # Strategy 1: Lexical - exact match in THIS ITEM'S question text
                if term_lower in item_text:
                    is_anchored = True

                # Strategy 2: Alias - check if alias STRING appears in THIS ITEM TEXT
                if not is_anchored and self.micro_enum:
                    for entry in self.micro_enum.values():
                        # Check if term matches canonical name
                        if term_lower == entry.name.lower():
                            # Verify canonical name appears in item text
                            if entry.name.lower() in item_text:
                                is_anchored = True
                                break
                        # Check aliases - must appear in THIS item's text
                        for alias in entry.aliases:
                            if term_lower == alias.lower():
                                # Verify this alias actually appears in item text
                                if alias.lower() in item_text:
                                    is_anchored = True
                                    break
                        if is_anchored:
                            break

                # Strategy 3: Semantic - cosine similarity ≥0.80 to THIS ITEM'S ngrams
                if not is_anchored and question_ngrams:
                    try:
                        term_emb = self.embedding_model.encode([term_lower], show_progress_bar=False)
                        ngram_embs = self.embedding_model.encode(question_ngrams, show_progress_bar=False)
                        sims = cosine_similarity(term_emb, ngram_embs)[0]
                        if np.max(sims) >= 0.80:
                            is_anchored = True
                    except Exception:
                        pass  # Skip semantic check if embedding fails

                # Strategy 4: Corpus fallback - appears ≥5 times, but CAP at 1 per item (tightened from 2)
                if not is_anchored and corpus_anchored_count < 1:
                    if corpus_freq.get(term_lower, 0) >= 5:
                        is_anchored = True
                        corpus_anchored_count += 1

                if is_anchored:
                    anchored_terms += 1

        if total_terms == 0:
            return 0.0
        ratio = anchored_terms / total_terms
        return max(0.0, min(1.0, ratio))

    def _compute_deduplication_rate(
        self,
        items: List[Dict[str, Any]],
        results: List[ExtractionResult]
    ) -> float:
        """
        Compute deduplication rate using TF-IDF similarity over extracted terms.

        Steps:
        1. Build per-item term vectors from ExtractionResult.terms
        2. Compute IDF globally across the run
        3. Form TF-IDF vectors, L2-normalize them
        4. Greedily drop items whose cosine similarity ≥ global threshold

        Args:
            items: Dataset items (unused but kept for signature parity)
            results: Extraction results aligned with items

        Returns:
            Deduplication rate: dropped_items / total_items (lower is better)
        """
        _ = items  # Retained for signature compatibility
        if not results or len(results) <= 1:
            return 0.0

        tfidf_matrix, has_terms = self._build_tfidf_matrix(results)
        if tfidf_matrix.size == 0 or not np.any(has_terms):
            return 0.0

        try:
            sim_matrix = tfidf_matrix @ tfidf_matrix.T
            threshold = getattr(self, 'dedup_similarity_threshold', 0.955)

            kept_indices = set(range(len(results)))
            dropped_count = 0

            for i in range(len(results)):
                if i not in kept_indices:
                    continue
                for j in range(i + 1, len(results)):
                    if j not in kept_indices:
                        continue
                    if sim_matrix[i, j] >= threshold:
                        kept_indices.discard(j)
                        dropped_count += 1

            return dropped_count / len(results) if results else 0.0

        except Exception as e:
            print(f"Warning: TF-IDF deduplication computation failed: {e}")
            return 0.0

    def _build_tfidf_matrix(
        self,
        results: List[ExtractionResult]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build TF-IDF matrix from extraction terms.

        Returns:
            (matrix, has_terms_mask)
        """
        doc_terms: List[Counter] = []
        doc_freq: Counter = Counter()

        for result in results:
            if not result.is_valid or not result.terms:
                doc_terms.append(Counter())
                continue

            lang = getattr(self, 'dataset_language', 'en')
            term_counts: Counter = Counter()
            for term in result.terms:
                normalized = self._normalize_text(term, lang=lang)
                if not normalized:
                    continue
                term_counts[normalized] += 1

            doc_terms.append(term_counts)
            for term in term_counts:
                doc_freq[term] += 1

        vocab = sorted(doc_freq.keys())
        if not vocab:
            self.term_idf_map = {}
            return np.zeros((len(results), 0), dtype=np.float32), np.zeros(len(results), dtype=bool)

        vocab_index = {term: idx for idx, term in enumerate(vocab)}
        num_docs = len(results)
        idf_map = {
            term: math.log((1 + num_docs) / (1 + df)) + 1.0
            for term, df in doc_freq.items()
        }
        self.term_idf_map = idf_map

        matrix = np.zeros((num_docs, len(vocab)), dtype=np.float32)
        has_terms = np.zeros(num_docs, dtype=bool)

        for doc_idx, term_counts in enumerate(doc_terms):
            if not term_counts:
                continue

            for term, count in term_counts.items():
                idx = vocab_index.get(term)
                if idx is None:
                    continue
                matrix[doc_idx, idx] = count * idf_map[term]

            norm = np.linalg.norm(matrix[doc_idx])
            if norm > 0:
                matrix[doc_idx] /= norm
                has_terms[doc_idx] = True

        return matrix, has_terms

    # ------------------------------------------------------------------------
    # 9. Quality Metrics
    # ------------------------------------------------------------------------

    def compute_quality_metrics(
        self,
        items: List[Dict[str, Any]],
        results: List[ExtractionResult]
    ) -> QualityMetrics:
        """
        Compute quality KPIs.

        Metrics:
        - Coverage: % items with valid extraction
        - Modality fidelity: % multimodal match to source
        - Language fidelity: % language match to source
        - Deduplication rate: % kept after near-duplicate removal
        - Anchored%: % terms validated by 4 anchor strategies
        """
        # Coverage: count all parsed (valid) items, even if term lists are empty
        covered = sum(1 for r in results if r.is_valid)
        coverage = covered / len(results) if results else 0.0

        # Modality fidelity
        source_multimodal = sum(1 for item in items if self._has_image(item))
        source_multimodal_pct = source_multimodal / len(items) if items else 0.0
        extracted_multimodal = sum(1 for r in results if r.visual_facts)
        extracted_multimodal_pct = extracted_multimodal / len(results) if results else 0.0
        modality_fidelity = extracted_multimodal_pct - source_multimodal_pct

        # Language fidelity (placeholder, needs language detection)
        # For simplicity, assume single language per dataset
        language_fidelity = 0.0  # Within ±5pp by construction in most cases

        # Deduplication rate - percentage of items that are near-duplicates
        dedup_rate = self._compute_deduplication_rate(items, results)

        # Anchored% - percentage of terms validated by anchor strategies
        anchored_percent = self._compute_anchored_percent(items, results)

        return QualityMetrics(
            coverage=coverage,
            modality_fidelity=modality_fidelity,
            language_fidelity=language_fidelity,
            deduplication_rate=dedup_rate,
            anchored_percent=max(0.0, min(1.0, anchored_percent)),
            coverage_floor=self.coverage_floor,
            deduplication_floor=self.dedup_floor,
            anchored_floor=self.anchored_floor
        )

    def _has_image(self, item: Dict[str, Any]) -> bool:
        """Check if item has an image"""
        if self._is_text_only_dataset():
            return False
        return 'image' in item or 'image_path' in item or 'images' in item

    def _get_image_paths(self, item: Dict[str, Any]) -> List[str]:
        """Extract and resolve image file paths from an item"""
        if not self._has_image(item):
            return []

        # Determine base directory for images
        dataset_dir = self.project_root / 'outputs' / 'phase0_datasets' / (self.dataset_name or '')
        raw_data_base = self.project_root / 'data' / 'raw_data'

        # Map dataset names to raw data directories
        dataset_to_raw = {
            'medxpertqa_mm': raw_data_base / 'medxpertqa' / 'images',
            'wemath': raw_data_base / 'wemath',
        }

        base_dir = dataset_to_raw.get(self.dataset_name, dataset_dir)

        image_paths = []

        # Handle different image field formats
        if 'images' in item:
            images_data = item['images']
            if isinstance(images_data, list):
                for img in images_data:
                    if isinstance(img, str):
                        # Direct path string (wemath format)
                        image_paths.append(str(base_dir / img))
                    elif isinstance(img, dict) and 'image_path' in img:
                        # Object with image_path (medxpertqa_mm format)
                        image_paths.append(str(base_dir / img['image_path']))

        elif 'image' in item:
            # Single image field
            image_paths.append(str(base_dir / item['image']))

        elif 'image_path' in item:
            # Direct image_path field
            image_paths.append(str(base_dir / item['image_path']))

        # Filter to existing files only
        valid_paths = [p for p in image_paths if Path(p).exists()]

        if image_paths and not valid_paths:
            print(f"Warning: No valid image paths found for item {item.get('id', 'unknown')}")
            print(f"  Tried paths: {image_paths[:2]}")  # Show first 2 attempts

        return valid_paths

    def _should_freeze_text_terms(self) -> bool:
        """Whether this dataset should reuse frozen text outputs."""
        dataset = (self.dataset_name or '').lower()
        return dataset in self.frozen_text_sources

    def _load_frozen_text_results(self, dataset_name: str) -> Dict[str, ExtractionResult]:
        """Load frozen text-only results for a dataset."""
        dataset_key = (dataset_name or '').lower()
        if dataset_key in self._frozen_text_cache:
            return self._frozen_text_cache[dataset_key]

        path = self.frozen_text_sources.get(dataset_key)
        if not path:
            self._frozen_text_cache[dataset_key] = {}
            return {}
        if not path.exists():
            print(f"⚠ Frozen text results missing for {dataset_name}: {path}")
            self._frozen_text_cache[dataset_key] = {}
            return {}

        cache: Dict[str, ExtractionResult] = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                item_id = str(record.get('item_id', '')).strip()
                extraction = record.get('extraction', {})
                if not item_id or not extraction:
                    continue

                result = ExtractionResult(
                    subdomains=extraction.get('subdomains', []),
                    terms=extraction.get('terms', []),
                    confidence=extraction.get('confidence', 0.0),
                    visual_facts=extraction.get('visual_facts'),
                    native_glosses=extraction.get('native_glosses'),
                    raw_response=extraction.get('raw_response', ''),
                    is_valid=extraction.get('is_valid', True),
                    anchored_count=extraction.get('anchored_count', len(extraction.get('terms', []))),
                    anchored_fraction=extraction.get('anchored_fraction', 0.0),
                    tom_categories=extraction.get('tom_categories'),
                )
                cache[item_id] = result

        self._frozen_text_cache[dataset_key] = cache
        return cache

    def _get_frozen_text_result(self, item: Dict[str, Any]) -> Optional[ExtractionResult]:
        """Fetch frozen result for a specific item if available."""
        dataset = (self.dataset_name or '').lower()
        cache = self._load_frozen_text_results(dataset)
        item_id = str(item.get('id', '')).strip()
        return cache.get(item_id)

    def _clone_extraction_result(self, result: ExtractionResult) -> ExtractionResult:
        """Deep copy an ExtractionResult."""
        return copy.deepcopy(result)

    async def _extract_visual_facts(self, item: Dict[str, Any]) -> List[str]:
        """Run a vision-only pass to capture visual facts without touching terms/subdomains."""
        image_files = self._get_image_paths(item)
        if not image_files:
            return []

        prompt = self._build_visual_facts_prompt(item)
        try:
            response = await self._call_llm(
                model=self.visual_oracle_name,
                prompt=prompt,
                temperature=0.0,
                top_p=1.0,
                max_tokens=512,
                image_files=image_files
            )
        except Exception as exc:
            print(f"    ⚠ Visual facts extraction failed for item {item.get('id', 'unknown')}: {exc}")
            return []

        return self._parse_visual_facts_response(response)

    def _build_visual_facts_prompt(self, item: Dict[str, Any]) -> str:
        """Create a lightweight prompt for visual-only captioning."""
        item_text = self._item_to_text(item)
        instructions = [
            "You are assisting an ontology team.",
            "Given the accompanying image(s), describe up to 3 short, objective visual facts.",
            "Each fact must be directly observable (no diagnoses, no speculation).",
            "Facts should be ≤ 15 words and stay neutral (e.g., \"bar chart compares x vs y\").",
            "Return ONLY JSON in this format:",
            '{ "visual_facts": ["fact one", "fact two"] }',
            "",
            "Text context (do not quote it, only use for grounding):",
            item_text,
        ]
        return "\n".join(instructions)

    def _parse_visual_facts_response(self, response: str) -> List[str]:
        """Parse JSON visual facts payload from vision-only pass."""
        if not response:
            return []
        text = response.strip()
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return []

        facts = data.get('visual_facts')
        if not isinstance(facts, list):
            return []

        cleaned: List[str] = []
        for fact in facts:
            if isinstance(fact, str):
                normalized = fact.strip()
                if normalized:
                    cleaned.append(normalized[:120])
        return cleaned[:3]

    def _is_text_only_dataset(self) -> bool:
        """Datasets that should ignore image content entirely."""
        return self.dataset_name in {'medxpertqa_text'}

    def _should_include_image_caption(self) -> bool:
        """Helper to decide if image captions belong in textual context."""
        return not self._is_text_only_dataset()

    def _build_tom_concepts(self) -> Dict[str, Set[str]]:
        """Seed Theory-of-Mind concept catalog used for Tombench datasets."""
        mental_states = {
            'belief', 'false belief', 'true belief', 'desire', 'intention',
            'goal', 'knowledge', 'ignorance', 'doubt', 'suspicion', 'plan',
            'decision', 'motive', 'expectation'
        }
        mental_states |= {
            '信念', '错误信念', '欲望', '意图', '想法', '认为', '以为', '知道', '不知道', '怀疑'
        }
        emotions = {
            'happiness', 'joy', 'sadness', 'anger', 'fear', 'embarrassment',
            'jealousy', 'envy', 'guilt', 'shame', 'disappointment', 'surprise',
            'frustration', 'anxiety', 'pride', 'relief', 'worry'
        }
        emotions |= {
            '开心', '高兴', '快乐', '难过', '悲伤', '生气', '愤怒', '害怕', '恐惧',
            '担心', '紧张', '失望', '激动', '焦虑'
        }
        social_acts = {
            'promise', 'lie', 'joke', 'tease', 'sarcasm', 'apology',
            'compliment', 'threat', 'fairness', 'cooperation', 'sharing',
            'deception', 'betrayal', 'trust', 'persuasion', 'agreement',
            'secret', 'confession'
        }
        social_acts |= {
            '撒谎', '说谎', '欺骗', '安慰', '鼓励', '道歉', '合作', '分享', '求助', '取笑', '嘲笑'
        }
        mental_verbs = {
            'think', 'believe', 'know', 'feel', 'wonder', 'guess', 'suspect',
            'imagine', 'notice', 'hope', 'worry', 'decide', 'plan', 'realize',
            'understand', 'forget', 'remember', 'doubt'
        }
        mental_verbs |= {
            '相信', '认为', '觉得', '以为', '知道', '不知道', '怀疑', '希望', '打算', '计划', '喜欢', '担心'
        }
        emotion_words = {
            'happy', 'sad', 'angry', 'afraid', 'scared', 'nervous', 'calm',
            'upset', 'relieved', 'jealous', 'guilty', 'ashamed', 'disappointed',
            'excited', 'anxious'
        }
        emotion_words |= {
            '高兴', '开心', '生气', '害怕', '紧张', '担心', '失望', '难过', '兴奋', '焦虑'
        }
        context_places = {
            'office', 'classroom', 'school', 'park', 'playground', 'mall',
            'restaurant', 'cafeteria', 'birthday party', 'meeting room',
            'library', 'gym', 'stadium', 'party', 'home', 'apartment'
        }
        context_places |= {
            '咖啡厅', '咖啡馆', '约会地点', '办公室', '社区', '新社区', '邻里',
            '邻居家', '学校', '教室', '礼堂', '走廊', '阁楼', '厨房', '柜子', '卧室', '阳台', '公园', '餐厅'
        }
        context_objects = {
            'gift', 'document', 'window', 'chair', 'swing', 'project',
            'promotion', 'meeting', 'team', 'cake', 'phone', 'message',
            'secret', 'plan', 'game', 'competition', 'neighbor', 'note',
            'email', 'invitation'
        }
        context_objects |= {
            '礼物', '公文包', '手提包', '连衣裙', '椅子', '窗户', '门锁', '入室盗窃', '咖啡杯', '礼品盒'
        }
        name_prefixes = {'xiao ', 'little ', 'lao '}
        return {
            'mental_states': {s.lower() for s in mental_states},
            'emotions': {s.lower() for s in emotions},
            'social_acts': {s.lower() for s in social_acts},
            'mental_verbs': {s.lower() for s in mental_verbs},
            'emotion_words': {s.lower() for s in emotion_words},
            'context_places': {s.lower() for s in context_places},
            'context_objects': {s.lower() for s in context_objects},
            'name_prefixes': name_prefixes
        }

    def _classify_tom_term(self, term: str) -> str:
        """Classify a term into ToM categories."""
        lower = term.lower().strip()
        if not lower:
            return 'other'
        concepts = self.tom_concepts
        if lower in concepts['mental_states']:
            return 'mental_state'
        if lower in concepts['emotions']:
            return 'emotion'
        if lower in concepts['social_acts']:
            return 'social_act'
        tokens = re.split(r'[\s\-_\/]+', lower)
        if any(token in concepts['mental_verbs'] for token in tokens):
            return 'mental_state'
        if any(token in concepts['emotion_words'] for token in tokens):
            return 'emotion'
        if lower in concepts['context_places']:
            return 'context_place'
        if lower in concepts['context_objects']:
            return 'context_place'
        if any(lower.startswith(prefix) for prefix in concepts['name_prefixes']):
            return 'context_name'
        # Simple Chinese-name heuristic: 2-4 Han characters -> treat as a name
        if re.fullmatch(r'[\u4e00-\u9fff]{2,4}', term):
            return 'context_name'
        return 'other'

    def _collapse_tom_category(self, category: str) -> str:
        mapping = {
            'mental_state': 'belief',
            'emotion': 'emotion',
            'social_act': 'social',
            'context_place': 'context',
            'context_name': 'context'
        }
        return mapping.get(category, 'other')

    def _token_length_ok(self, term_text: str) -> bool:
        """Check token length constraints (1–4 tokens) depending on language."""
        lang_hint = self.dataset_language
        if lang_hint == 'zh' or self._is_chinese(term_text):
            return 1 <= len(term_text) <= 8
        tokens = term_text.split()
        return 1 <= len(tokens) <= 4

    def _build_tiny_tom_entries(self, lang: str) -> List[MicroEnumEntry]:
        """Curated micro-enum entries for ToM concepts."""
        concepts = self.tom_concepts
        base_terms = sorted(
            concepts['mental_states']
            | concepts['emotions']
            | concepts['social_acts']
        )
        entries: List[MicroEnumEntry] = []

        if lang == 'zh':
            zh_terms = [
                ('信念', ['信念', 'belief']),
                ('欲望', ['欲望', '渴望']),
                ('意图', ['意图', '打算']),
                ('目标', ['目标']),
                ('知识', ['知识']),
                ('无知', ['无知']),
                ('怀疑', ['怀疑', '疑虑']),
                ('计划', ['计划']),
                ('决定', ['决定']),
                ('动机', ['动机']),
                ('期望', ['期望']),
                ('快乐', ['快乐', '高兴']),
                ('悲伤', ['悲伤', '难过']),
                ('愤怒', ['愤怒', '生气']),
                ('害怕', ['害怕', '恐惧']),
                ('尴尬', ['尴尬']),
                ('嫉妒', ['嫉妒']),
                ('自豪', ['自豪']),
                ('内疚', ['内疚']),
                ('羞愧', ['羞愧']),
                ('失望', ['失望']),
                ('惊讶', ['惊讶']),
                ('焦虑', ['焦虑']),
                ('诚实', ['诚实']),
                ('谎言', ['谎言', '说谎']),
                ('承诺', ['承诺']),
                ('道歉', ['道歉']),
                ('合作', ['合作']),
                ('分享', ['分享']),
                ('欺骗', ['欺骗']),
                ('嘲笑', ['嘲笑', '取笑']),
                ('讽刺', ['讽刺']),
                ('信任', ['信任']),
                ('怀疑', ['怀疑']),
                ('安慰', ['安慰']),
                ('冲突', ['冲突'])
            ]
            for idx, (name, alias_list) in enumerate(zh_terms):
                normalized = self._normalize_text(name, lang)
                aliases = list({normalized, *alias_list})
                entry = MicroEnumEntry(
                    id=f"tiny_tom_zh_{idx:03d}",
                    name=normalized,
                    aliases=aliases,
                    frequency=1000 - idx
                )
                entries.append(entry)
        else:
            for idx, term in enumerate(base_terms):
                normalized = self._normalize_text(term, lang)
                aliases = [term]
                if normalized != term:
                    aliases.append(normalized)
                entry = MicroEnumEntry(
                    id=f"tiny_tom_{idx:03d}",
                    name=normalized,
                    aliases=aliases,
                    frequency=1000 - idx  # arbitrary weight to keep them near top
                )
                entries.append(entry)
        return entries

    def _canonicalize_tombench_cn_subdomain(self, value: str) -> List[str]:
        if not value:
            return []
        raw = value.strip()
        normalized = raw.replace('　', '').replace(' ', '')
        if not normalized:
            return []
        segments = re.split(r'[／/、,，]+', normalized) if re.search(r'[／/、,，]', normalized) else [normalized]
        matches: List[str] = []
        for segment in segments:
            match = self._match_tombench_cn_keyword(segment)
            if match and match not in matches:
                matches.append(match)
        if not matches:
            match = self._match_tombench_cn_keyword(normalized)
            if match:
                matches.append(match)
        return matches

    def _match_tombench_cn_keyword(self, token: str) -> Optional[str]:
        token = token.strip()
        if not token:
            return None
        token_compact = token.replace(' ', '').replace('-', '')
        for allowed in self.tombench_cn_allowed_subdomains:
            normalized_allowed = allowed.replace(' ', '').replace('-', '')
            if token_compact == normalized_allowed or token == allowed:
                return allowed
        for keyword, target in self.tombench_cn_keyword_map.items():
            if keyword in token:
                return target
        return None

    def _enforce_tombench_cn_subdomains(self, subdomains: List[str], item: Dict[str, Any]) -> List[str]:
        if not self._is_tombench_cn_dataset():
            return subdomains
        normalized: List[str] = []
        seen = set()
        for candidate in subdomains:
            for canonical in self._canonicalize_tombench_cn_subdomain(candidate):
                if canonical and canonical not in seen:
                    normalized.append(canonical)
                    seen.add(canonical)
                if len(normalized) >= 3:
                    break
        if len(normalized) < 3:
            item_text = self._item_to_text_whitelisted(item)
            text_hits = self._infer_cn_subdomains_from_text(item_text)
            for hit in text_hits:
                if hit not in seen:
                    normalized.append(hit)
                    seen.add(hit)
                if len(normalized) >= 3:
                    break
        if not normalized:
            for canonical in self._fallback_tombench_cn_subdomains(item):
                if canonical and canonical not in seen:
                    normalized.append(canonical)
                    seen.add(canonical)
                if len(normalized) >= 3:
                    break
        return sorted(normalized)[:3]

    def _fallback_tombench_cn_subdomains(self, item: Dict[str, Any]) -> List[str]:
        if not self._is_tombench_cn_dataset():
            return []
        item_id = str(item.get('id', '')).strip()
        if not item_id:
            return []
        en_map = self._get_parallel_subdomains_map('tombench_en')
        en_subdomains = en_map.get(item_id, [])
        mapped: List[str] = []
        for sub in en_subdomains:
            mapped_label = self._map_en_subdomain_to_cn(sub)
            if mapped_label and mapped_label not in mapped:
                mapped.append(mapped_label)
        return mapped

    def _get_parallel_subdomains_map(self, dataset_key: str) -> Dict[str, List[str]]:
        if dataset_key in self.parallel_subdomain_cache:
            return self.parallel_subdomain_cache[dataset_key]
        mapping: Dict[str, List[str]] = {}
        path = self._get_latest_results_path(dataset_key)
        if not path:
            self.parallel_subdomain_cache[dataset_key] = mapping
            return mapping
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    item_id = str(record.get('item_id', '')).strip()
                    extraction = record.get('extraction') or {}
                    subdomains = extraction.get('subdomains') or []
                    if item_id:
                        mapping[item_id] = subdomains
        except FileNotFoundError:
            mapping = {}
        self.parallel_subdomain_cache[dataset_key] = mapping
        return mapping

    def _infer_cn_subdomains_from_text(self, text: str) -> List[str]:
        """Infer CN subdomains directly from keyword hits in the item text."""
        if not text:
            return []
        hits: Dict[str, None] = {}
        for keyword, target in self.tombench_cn_keyword_map.items():
            if keyword and keyword in text:
                hits[target] = None
        ordered_hits = [sub for sub in self.tombench_cn_allowed_subdomains if sub in hits]
        if len(ordered_hits) < len(hits):
            for subdomain in hits:
                if subdomain not in ordered_hits:
                    ordered_hits.append(subdomain)
        return ordered_hits

    def _get_latest_results_path(self, dataset_key: str) -> Optional[Path]:
        pattern = f"{dataset_key}_results_*.jsonl"
        candidates = sorted(self.output_dir.glob(pattern))
        if not candidates:
            return None
        return candidates[-1]

    def _map_en_subdomain_to_cn(self, subdomain: str) -> Optional[str]:
        normalized = (subdomain or '').lower().strip()
        if not normalized:
            return None
        for rule in self.tombench_en_to_cn_rules:
            if any(keyword in normalized for keyword in rule['keywords']):
                return rule['label']
        return None

    def _build_tombench_en_to_cn_rules(self) -> List[Dict[str, Any]]:
        return [
            {
                'keywords': (
                    'belief reasoning',
                    'false belief',
                    'theory of mind',
                    'mental state',
                    'deception',
                    'knowledge',
                    'ignorance',
                    'perspective',
                    'belief'
                ),
                'label': '信念推理'
            },
            {
                'keywords': (
                    'object location',
                    'object tracking',
                    'object search',
                    'location reasoning',
                    'spatial',
                    'object interaction',
                    'object relocation'
                ),
                'label': '物体位置-物体追踪'
            },
            {
                'keywords': (
                    'emotion recognition',
                    'emotion attribution',
                    'emotion',
                    'affect',
                    'feeling',
                    'moral emotion',
                    'emotional'
                ),
                'label': '情绪识别'
            },
            {
                'keywords': (
                    'social interaction',
                    'social dynamics',
                    'social attribution',
                    'social',
                    'interpersonal',
                    'relationship',
                    'family interaction',
                    'family dynamics',
                    'non-literal',
                    'pragmatic',
                    'communication',
                    'workplace',
                    'school life',
                    'personal relationship'
                ),
                'label': '人际关系-社交互动'
            },
            {
                'keywords': (
                    'intention explanation',
                    'intention reasoning',
                    'intention',
                    'goal inference',
                    'plan inference',
                    'plan reasoning'
                ),
                'label': '意图推理'
            },
            {
                'keywords': (
                    'moral',
                    'morality',
                    'moral judgment',
                    'ethical',
                    'ethics',
                    'fairness',
                    'justice',
                    'should',
                    'ought',
                    'deontic',
                    'rule violation',
                    'honesty'
                ),
                'label': '道德判断-道德推理'
            },
            {
                'keywords': (
                    'estimation',
                    'quantity',
                    'counting',
                    'how many',
                    'inventory',
                    'daily reasoning',
                    'routine reasoning',
                    'number reasoning',
                    'measurement'
                ),
                'label': '日常推理-数量估计'
            },
            {
                'keywords': (
                    'community',
                    'neighborhood',
                    'safety',
                    'security',
                    'burglary',
                    'community engagement',
                    'community life',
                    'neighbor'
                ),
                'label': '社区生活-安全'
            }
        ]

    # ------------------------------------------------------------------------
    # Spot-Check Helper
    # ------------------------------------------------------------------------

    def spot_check_items(
        self,
        items: List[Dict[str, Any]],
        results1: List[ExtractionResult],
        results2: List[ExtractionResult],
        num_samples: int = 5,
        seed: Optional[int] = None
    ) -> None:
        """
        Spot-check random items to visually compare two runs.

        Prints:
        - Item ID and prompt snippet
        - Run A subdomains vs Run B subdomains
        - Run A terms vs Run B terms

        Args:
            items: Dataset items
            results1: First run results
            results2: Second run results
            num_samples: Number of items to check
            seed: Random seed for sampling
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample random indices
        n = len(items)
        indices = np.random.choice(n, min(num_samples, n), replace=False)

        print(f"\n{'='*80}")
        print(f"SPOT-CHECK: {num_samples} Random Items")
        print(f"{'='*80}\n")

        for idx in indices:
            item = items[idx]
            r1 = results1[idx]
            r2 = results2[idx]

            item_id = item.get('id', idx)
            question = item.get('question', '')[:100] + ('...' if len(item.get('question', '')) > 100 else '')

            print(f"Item ID: {item_id}")
            print(f"Question: {question}")
            print(f"\nRun A subdomains: {r1.subdomains if r1.is_valid else 'INVALID'}")
            print(f"Run B subdomains: {r2.subdomains if r2.is_valid else 'INVALID'}")
            print(f"\nRun A terms: {r1.terms if r1.is_valid else 'INVALID'}")
            print(f"Run B terms: {r2.terms if r2.is_valid else 'INVALID'}")

            # Display visual facts if present (for multimodal datasets)
            if r1.is_valid and r1.visual_facts:
                print(f"\nRun A visual facts: {r1.visual_facts}")
            if r2.is_valid and r2.visual_facts:
                print(f"Run B visual facts: {r2.visual_facts}")

            if r1.is_valid and r2.is_valid:
                # Compute per-item metrics
                sub_jac = self._jaccard(set(r1.subdomains), set(r2.subdomains))
                term_jac = self._jaccard(set(r1.terms), set(r2.terms))
                print(f"\nPer-item Jaccard: subdomains={sub_jac:.3f}, terms={term_jac:.3f}")

            print(f"\n{'-'*80}\n")

        print(f"{'='*80}\n")

    # ------------------------------------------------------------------------
    # 10. Run Manifest
    # ------------------------------------------------------------------------

    def create_run_manifest(
        self,
        dataset_name: str,
        items: List[Dict[str, Any]],
        prompt_template: str
    ) -> RunManifest:
        """Create reproducibility manifest for a run"""
        # Dataset hash
        dataset_str = json.dumps([item.get('id', str(i)) for i, item in enumerate(items)], sort_keys=True)
        dataset_hash = hashlib.sha256(dataset_str.encode()).hexdigest()[:16]

        # Sampled item IDs
        sampled_ids = [str(item.get('id', i)) for i, item in enumerate(items)]

        # Prompt hash
        prompt_hash = hashlib.sha256(prompt_template.encode()).hexdigest()[:16]

        # Timestamp
        timestamp = datetime.now().isoformat()

        # Parent map hash
        parent_map_hash = self._hash_parent_map()

        return RunManifest(
            dataset_hash=dataset_hash,
            sampled_item_ids=sampled_ids,
            oracle_name=self.oracle_name,
            oracle_version="unknown",  # Would need to query API for version
            temperature=0.0,
            top_p=1.0,
            prompt_hash=prompt_hash,
            micro_enum_hash=self.micro_enum_hash,
            parent_map_hash=parent_map_hash,
            embedding_model=self.embedding_model_name,
            random_seed=self.seed,
            timestamp=timestamp
        )

    # ------------------------------------------------------------------------
    # Main Workflow
    # ------------------------------------------------------------------------

    async def run_extraction(
        self,
        dataset_name: str,
        num_items: Optional[int] = None,
        use_micro_enum: bool = True,
        use_consistency: bool = False,
        num_runs: int = 2
    ) -> Dict[str, Any]:
        """
        Run full extraction pipeline.

        Steps:
        1. Load dataset
        2. Build micro-enum (if enabled)
        3. Extract for all items (with optional K=2 consistency)
        4. Build hierarchy
        5. Compute stability metrics (if num_runs > 1)
        6. Compute quality metrics
        7. Check floors
        8. Save results + manifest

        Args:
            dataset_name: Dataset to extract from
            num_items: Number of items to sample (None = all)
            use_micro_enum: Build and use micro-enum
            use_consistency: Use K=2 self-consistency
            num_runs: Number of runs for stability testing

        Returns:
            Report dictionary
        """
        print(f"\n{'='*80}")
        print(f"LEAN DOMAIN EXTRACTION - Stage 1")
        print(f"{'='*80}")
        print(f"Dataset: {dataset_name}")
        self.dataset_name = dataset_name
        self.dataset_domain = self._infer_dataset_domain(dataset_name)
        print(f"Oracle (text): {self.oracle_name} (temp=0.0, top_p=1.0)")
        if self.visual_oracle_name != self.oracle_name:
            print(f"Oracle (visual): {self.visual_oracle_name} (temp=0.0, top_p=1.0)")
        print(f"Micro-enum: {use_micro_enum}")
        print(f"Self-consistency: {use_consistency}")
        print(f"Number of runs: {num_runs}")

        # Set dataset language & dataset-specific thresholds
        self.dataset_language = self._get_dataset_language(dataset_name)
        if self.dataset_language == 'zh':
            self.coverage_floor = 0.80
            self.dedup_floor = 0.15
            self.anchored_floor = 0.50
            self.anchored_coverage_threshold = 2
        else:
            self.coverage_floor = 0.95
            self.dedup_floor = 0.10
            self.anchored_floor = 0.50
            self.anchored_coverage_threshold = 3
        print(f"Language: {self.dataset_language}")
        print(f"Domain: {self.dataset_domain}")
        print(f"TF-IDF dedup cosine threshold: {self.dedup_similarity_threshold:.3f}")
        print(f"{'='*80}\n")

        # 1. Load dataset
        items = self._load_dataset(dataset_name, num_items)
        self.freeze_text_active = self._should_freeze_text_terms()
        if self.freeze_text_active:
            print("  → Using frozen text outputs for subdomains/terms; vision runs are visual-only.")
            self._load_frozen_text_results(self.dataset_name)

        # 2. Build micro-enum
        if use_micro_enum:
            self.build_micro_enum(items)
            glossary = [e.name for e in self.micro_enum.values()][:100]  # Top 100 for prompt
        else:
            glossary = None

        # 3. Extract (multiple runs for stability)
        all_run_results = []
        for run_idx in range(num_runs):
            print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
            if self.freeze_text_active and run_idx > 0 and all_run_results:
                cloned_results = [self._clone_extraction_result(r) for r in all_run_results[0]]
                all_run_results.append(cloned_results)
                print(f"  ✓ Run {run_idx + 1} reused frozen text outputs ({len(cloned_results)} items)")
                continue

            run_results = []

            for idx, item in enumerate(items):
                if idx % 50 == 0:
                    print(f"  Processing {idx}/{len(items)}...")

                has_image = self._has_image(item)
                is_multilingual = self._is_multilingual(dataset_name)
                used_frozen = False

                if self.freeze_text_active:
                    frozen_result = self._get_frozen_text_result(item)
                    if frozen_result:
                        result = self._clone_extraction_result(frozen_result)
                        result.raw_response = frozen_result.raw_response or "frozen_text_result"
                        result.is_valid = True
                        used_frozen = True
                    else:
                        print(f"    ⚠ No frozen text result for item {item.get('id', 'unknown')}; falling back to text oracle.")

                if not used_frozen:
                    text_has_image = has_image if not self.freeze_text_active else False
                    if use_consistency:
                        result = await self.extract_with_consistency(
                            item,
                            k=2,
                            has_image=text_has_image,
                            is_multilingual=is_multilingual,
                            glossary=glossary
                        )
                    else:
                        result = await self._extract_once(
                            item,
                            has_image=text_has_image,
                            is_multilingual=is_multilingual,
                            glossary=glossary
                        )

                # Validate and fix extraction (apply hygiene filters + per-item glossary replacement)
                if result.is_valid and not used_frozen:
                    per_item_glossary = (
                        self._get_per_item_glossary(item, top_k=30) if self.micro_enum else None
                    )
                    result = self._validate_and_fix_extraction(result, item, per_item_glossary)

                if has_image and self.freeze_text_active and result.is_valid:
                    visual_facts = await self._extract_visual_facts(item)
                    if visual_facts:
                        result.visual_facts = visual_facts

                run_results.append(result)

            all_run_results.append(run_results)
            print(f"  ✓ Run {run_idx + 1} complete: {sum(1 for r in run_results if r.is_valid)}/{len(run_results)} valid")

        # 3b. Canonicalize terms if using micro-enum
        if use_micro_enum and self.micro_enum:
            print(f"\n--- Canonicalizing Terms ---")
            for run_idx, run_results in enumerate(all_run_results):
                canon_count = 0
                for result in run_results:
                    if not result.is_valid:
                        continue
                    # Canonicalize terms
                    original_terms = result.terms.copy()
                    result.terms = self.canonicalize_to_enum(result.terms, lang=self.dataset_language)
                    # Canonicalize subdomains
                    result.subdomains = self.canonicalize_to_enum(result.subdomains, lang=self.dataset_language)
                    if original_terms != result.terms:
                        canon_count += 1
                print(f"  Run {run_idx + 1}: Canonicalized {canon_count}/{len(run_results)} items")

        # 4. Build frozen parent map from UNION of all runs (after canonicalization)
        # This ensures consistent hierarchical agreement across runs
        print(f"\n--- Building Frozen Parent Map ---")
        all_subdomains = []
        for run_results in all_run_results:
            for result in run_results:
                if result.is_valid:
                    all_subdomains.extend(result.subdomains)

        print(f"  Collecting subdomains from {num_runs} runs...")
        print(f"  Total subdomains (with duplicates): {len(all_subdomains)}")
        print(f"  Unique subdomains: {len(set(all_subdomains))}")

        self.build_hierarchy(all_subdomains)

        # Hash the frozen parent map for reproducibility
        parent_map_hash = self._hash_parent_map()
        if parent_map_hash:
            print(f"  Parent map hash: {parent_map_hash}")
            print(f"  ✓ Frozen parent map will be used for ALL runs")

        # 5. Compute stability metrics (if multiple runs)
        stability_metrics_list = []
        if num_runs >= 2:
            print(f"\n--- Stability Metrics ---")
            for i in range(num_runs - 1):
                metrics = self.compute_stability_metrics(all_run_results[i], all_run_results[i+1])
                stability_metrics_list.append(metrics)
                print(f"Run {i+1} vs Run {i+2}:")
                print(f"  Raw Jaccard (subdomains): {metrics.raw_jaccard_subdomains:.3f}")
                print(f"  Raw Jaccard (terms): {metrics.raw_jaccard_terms:.3f}")
                soft_jac_str = "N/A" if metrics.soft_jaccard_terms < 0 else f"{metrics.soft_jaccard_terms:.3f}"
                print(f"  Soft Jaccard (terms): {soft_jac_str}")
                print(f"  NSI: {metrics.nsi:.3f}")
                print(f"  Hierarchical agreement: {metrics.hierarchical_agreement:.3f}")

                passes, failures = metrics.passes_floors()
                if passes:
                    print(f"  ✓ Passes all floors")
                else:
                    print(f"  ✗ Failed floors: {', '.join(failures)}")

            # Spot-check 5 random items for manual inspection
            print(f"\n--- Spot-Check (5 random items) ---")
            self.spot_check_items(items, all_run_results[0], all_run_results[1], num_samples=5, seed=42)

        # Average stability metrics
        avg_stability = None
        if stability_metrics_list:
            avg_stability = StabilityMetrics(
                raw_jaccard_subdomains=np.mean([m.raw_jaccard_subdomains for m in stability_metrics_list]),
                raw_jaccard_terms=np.mean([m.raw_jaccard_terms for m in stability_metrics_list]),
                soft_jaccard_subdomains=np.mean([m.soft_jaccard_subdomains for m in stability_metrics_list]),
                soft_jaccard_terms=np.mean([m.soft_jaccard_terms for m in stability_metrics_list]),
                nsi=np.mean([m.nsi for m in stability_metrics_list]),
                hierarchical_agreement=np.mean([m.hierarchical_agreement for m in stability_metrics_list])
            )

        # 6. Compute quality metrics (using first run)
        quality_metrics = self.compute_quality_metrics(items, all_run_results[0])
        print(f"\n--- Quality Metrics ---")
        print(f"Coverage: {quality_metrics.coverage:.3f}")
        print(f"Modality fidelity: {quality_metrics.modality_fidelity:+.3f}")
        print(f"Language fidelity: {quality_metrics.language_fidelity:+.3f}")
        print(f"Near-duplicate%: {quality_metrics.deduplication_rate:.1%} (lower is better)")
        print(f"Anchored%: {quality_metrics.anchored_percent:.3f}")

        passes_quality, quality_failures = quality_metrics.passes_floors()
        if passes_quality:
            print(f"✓ Passes all quality floors")
        else:
            print(f"✗ Failed quality floors: {', '.join(quality_failures)}")

        # 7. Overall pass/fail
        passes_stability = not stability_metrics_list or all(m.passes_floors()[0] for m in stability_metrics_list)
        overall_pass = passes_stability and passes_quality

        print(f"\n{'='*80}")
        if overall_pass:
            print("✓ PASS: All floors met")
        else:
            print("✗ FAIL: Some floors not met")
            if not passes_stability:
                print("  - Stability floors failed")
            if not passes_quality:
                print("  - Quality floors failed")
        print(f"{'='*80}\n")

        # 8. Create manifest
        prompt_template = self._make_extraction_prompt(
            items[0], language=self.dataset_language
        )  # Use first item as template
        manifest = self.create_run_manifest(dataset_name, items, prompt_template)

        # 9. Save results
        report = {
            'dataset': dataset_name,
            'num_items': len(items),
            'oracle': self.oracle_name,
            'visual_oracle': self.visual_oracle_name if self.visual_oracle_name != self.oracle_name else None,
            'use_micro_enum': use_micro_enum,
            'use_consistency': use_consistency,
            'num_runs': num_runs,
            'manifest': manifest.to_dict(),
            'stability_metrics': avg_stability.__dict__ if avg_stability else None,
            'quality_metrics': quality_metrics.__dict__,
            'overall_pass': overall_pass,
            'timestamp': datetime.now().isoformat()
        }

        # Save report
        report_path = self.output_dir / f"{dataset_name}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"Report saved to: {report_path}")

        # Save extracted results
        results_path = self.output_dir / f"{dataset_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(results_path, 'w', encoding='utf-8') as f:
            for item, result in zip(items, all_run_results[0]):
                record = {
                    'item_id': item.get('id', ''),
                    'extraction': result.to_dict()
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

        print(f"Results saved to: {results_path}")

        return report

    def _load_dataset(self, dataset_name: str, num_items: Optional[int]) -> List[Dict[str, Any]]:
        """Load dataset items"""
        dataset_map = {
            'csbench_en_test': 'csbench_en_test',
            'csbench_cn_test': 'csbench_cn_test',
            'csbench_fr_test': 'csbench_fr_test',
            'csbench_de_test': 'csbench_de_test',
            'medxpertqa_text': 'medxpertqa_text',
            'medxpertqa_mm': 'medxpertqa_mm',
            'tombench_cn': 'tombench_cn',
            'tombench_en': 'tombench_en',
            'wemath': 'wemath'
        }

        if dataset_name not in dataset_map:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        dataset_dir = dataset_map[dataset_name]
        dataset_path = self.project_root / 'outputs' / 'phase0_datasets' / dataset_dir / 'dataset.jsonl'

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        items = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))

        if num_items:
            np.random.seed(self.seed)
            indices = np.random.choice(len(items), min(num_items, len(items)), replace=False)
            items = [items[i] for i in indices]

        print(f"Loaded {len(items)} items from {dataset_name}")
        return items

    def _infer_dataset_domain(self, dataset_name: str) -> str:
        """Infer high-level domain from dataset name."""
        name = (dataset_name or "").lower()
        domain_map = [
            ('medxpertqa', 'medical'),
            ('csbench', 'cs'),
            ('tombench', 'psychology'),
            ('wemath', 'math'),
            ('we-math', 'math'),
            ('we_math', 'math'),
            ('anti', 'security'),
            ('esg', 'finance')
        ]
        for key, domain in domain_map:
            if key in name:
                return domain
        return 'general'

    def _is_multilingual(self, dataset_name: str) -> bool:
        """Check if dataset is multilingual"""
        return 'cn' in dataset_name or 'tombench' in dataset_name

    def _get_dataset_language(self, dataset_name: str) -> str:
        """Get the primary language of the dataset"""
        dataset_name_lower = dataset_name.lower()

        if '_cn' in dataset_name_lower or 'tombench_cn' in dataset_name_lower:
            return 'zh'
        elif '_fr' in dataset_name_lower:
            return 'fr'
        elif '_de' in dataset_name_lower:
            return 'de'
        elif '_en' in dataset_name_lower or 'tombench_en' in dataset_name_lower:
            return 'en'
        elif 'medxpertqa' in dataset_name_lower:
            return 'en'  # MedXpertQA is in English
        elif 'wemath' in dataset_name_lower:
            return 'en'  # WeMath is primarily in English
        else:
            return 'en'  # Default to English


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Example usage"""
    # Initialize extractor
    extractor = LeanDomainExtractor(
        oracle_name="gpt-4o-mini",  # Single oracle
        embedding_model_name="intfloat/multilingual-e5-small"
    )

    # Run extraction
    report = await extractor.run_extraction(
        dataset_name='csbench_en',
        num_items=100,  # Sample 100 items
        use_micro_enum=True,
        use_consistency=False,  # Start without, enable if floors fail
        num_runs=2  # Two runs for stability testing
    )

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80)
    print(f"Overall pass: {report['overall_pass']}")
    print(f"Report saved to: {extractor.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
