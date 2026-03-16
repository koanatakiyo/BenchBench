#!/usr/bin/env python3
"""
Build a domain_card.yaml for a dataset from parsed Stage-1 outputs.

Example:
    python build_domain_card.py \
        --dataset tombench_en \
        --results /home/.../parsed_stage1_output/tombench_en/tombench_en_results.jsonl \
        --output /home/.../domain_cards/tombench_en_domain_card.yaml
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

import yaml

from tom_taxonomy import TOM_SUPER_TAGS
from cs_super_parents import CS_SUPER_PARENTS
from wemath_super_parents import WEMATH_SUPER_PARENTS
from med_specialty_taxonomy import SPECIALTY_MAP, TASK_AXIS_MAP

CURATED_BACKGROUND_SNIPPETS: Dict[str, Dict[str, List[str]]] = {
    "tombench_en": {
        "ToM.BeliefReasoning": [
            "Xiao Wang hides a tie in a suitcase; while he is away, Youyou secretly moves it to the cabinet. The item asks where Xiao Wang will look first.",
            "A family argues about who knows the real location of a hidden statue, probing second-order beliefs.",
            "Two classmates swap backpacks during lunch, forcing the reader to track what each person thinks happened."
        ],
        "ToM.EmotionRecognition": [
            "A drama club performs in front of peers; students must infer who feels nervous, jealous, or proud.",
            "A sibling receives an unexpected gift while another feels overlooked, requiring recognition of mixed emotions.",
            "A student plans a surprise party and the story asks how the guest of honor is likely to feel."
        ],
        "ToM.SocialInteraction": [
            "Neighbors negotiate over a borrowed bicycle, highlighting cooperation vs. conflict cues.",
            "A first date at a cafe explores expectations between Xiao Ming and Xiao Hong.",
            "Roommates discuss gossip about a mutual friend, requiring reasoning over social alliances."
        ],
        "ToM.NonLiteralCommunication": [
            "A manager tells a joke during a meeting, and the worker must detect sarcasm vs. sincerity.",
            "A parent scolds a child with indirect hints about responsibility.",
            "Friends exchange faux pas at a dinner party, testing pragmatic inference."
        ],
        "ToM.ObjectLocation": [
            "Xiao Li hides a handbag in the attic; Youyou later places it in the cabinet. Students must determine search behavior.",
            "Two kids move objects between lockers while the owner is absent.",
            "A pantry shuffle forces tracking of bowls and spoons across shelves."
        ],
        "ToM.IntentionReasoning": [
            "A teammate pretends to forget homework to test whether others will help.",
            "Someone irons a shirt and polishes shoes to hint at an interview plan.",
            "A neighbor plants false clues to conceal a surprise celebration."
        ],
        "ToM.MoralJudgment": [
            "Children witness a petty theft in their community and debate whether to report it.",
            "Students weigh fairness after a classmate cheats on an exam.",
            "Neighbors discuss whether breaking a promise can be justified."
        ]
    },
    "tombench_cn": {
        "ToM.BeliefReasoning": [
            "小王把领带藏在手提箱里，悠悠趁他不在时悄悄移到柜子里，问题询问小王先去哪找。",
            "两个同学在课堂上交换书包，需要判断彼此的信念和认知。",
            "邻居讨论隐蔽的礼物位置，考察谁掌握真实信息。"
        ],
        "ToM.EmotionRecognition": [
            "学校才艺表演前，同学们表现出紧张、羡慕与自豪等不同情绪。",
            "第一次约会时，小红对意外礼物的情感反应被追问。",
            "社区聚会中有人突然尴尬失态，问题要求识别情绪原因。"
        ],
        "ToM.SocialInteraction": [
            "邻里借用物品却迟迟不还，引发信任与冲突。",
            "朋友之间的生日惊喜牵动互相的期待与失望。",
            "家庭餐桌上父母与孩子的互动需要推断未说出口的意图。"
        ],
        "ToM.NonLiteralCommunication": [
            "同事会餐时有人开讽刺玩笑，考察谁听懂言外之意。",
            "课堂上老师用委婉语提醒学生注意行为。",
            "朋友间的不当言辞导致“失礼”，需要理解语用含义。"
        ],
        "ToM.ObjectLocation": [
            "李雷把胡萝卜放在花园盒子里，悠悠后来移到柜子中，考察错误信念。",
            "孩子们在厨房、阁楼之间来回搬动手提包与盒子。",
            "洗衣房出现被移位的海绵和标签，学生需推理搜索顺序。"
        ],
        "ToM.IntentionReasoning": [
            "小明假装生病以逃避测验，问题询问真实意图。",
            "邻居提前准备礼物、整理仪表，暗示重要会面。",
            "朋友故意隐瞒行程，为了给他人一个惊喜。"
        ],
        "ToM.MoralJudgment": [
            "新社区发生入室盗窃，居民讨论应该如何处理信息。",
            "同学之间存在隐瞒与撒谎，引导学生评估对错。",
            "邻里互助或推诿的场景考察公平与责任。"
        ]
    },
    "csbench_en_test": {
        "CS.Core.DataStructuresAlgorithms": [
            "Students compare recursion vs. iteration when traversing a tree or graph.",
            "A pseudo-code fragment manipulates stacks and queues, asking which structure fits best.",
            "Sorting and searching performance is analyzed under different constraints."
        ],
        "CS.Core.ComputerOrganization": [
            "A CPU pipeline diagram asks learners to reason about hazards and stalls.",
            "Memory hierarchy questions relate cache hit ratios to instruction timing.",
            "Assembly snippets illustrate register usage within fetch-decode-execute cycles."
        ],
        "CS.Core.ComputerNetwork": [
            "An email relay scenario probes transport-layer reliability vs. UDP trade-offs.",
            "Students map routing entries across network and data-link layers.",
            "Wireless handoff examples highlight collision avoidance protocols."
        ],
        "CS.Core.OperatingSystem": [
            "Threads contend for a shared file handle, requiring synchronization primitives.",
            "A virtual memory question links page-table entries to physical frames.",
            "IO schedulers prioritize workloads on disks with different service times."
        ]
    },
    "csbench_cn_test": {
        "CS.Core.DataStructuresAlgorithms": [
            "题目给出一段栈/队列伪代码，要求判断输出序列是否合法。",
            "分析树或图的遍历策略，比较递归与非递归实现。",
            "排序与查找算法在不同输入规模下的效率比较。"
        ],
        "CS.Core.ComputerOrganization": [
            "给定CPU五级流水线示意，询问结构与数据冒险如何解决。",
            "Cache、主存、寄存器之间的数据访问延迟对比题。",
            "指令系统与寄存器组织的选择题。"
        ],
        "CS.Core.ComputerNetwork": [
            "描述路由器转发表，让学生判断最优路径与下一跳。",
            "应用层协议如HTTP/SMTP的报文结构理解题。",
            "无线网络冲突避免与接入控制场景分析。"
        ],
        "CS.Core.OperatingSystem": [
            "描述分页与段页式内存管理的映射过程，考查虚拟地址转换。",
            "进程调度算法在不同优先级和时间片组合下的执行顺序。",
            "磁盘IO调度与设备管理策略的比较。"
        ]
    },
    "medxpertqa_text": {},
    "medxpertqa_mm": {},
    "wemath": {
        "Math.Geometry.PlaneFigures": [
            "A wire first forms a circle, then is reshaped into an isosceles trapezoid with the same perimeter.",
            "Students fold paper nets to reconstruct a particular polygon.",
            "Triangle ABC describes the front view of a cone, linking planar ratios to solid dimensions."
        ],
        "Math.Geometry.SolidFigures": [
            "A circular iron sheet is rolled into a cylindrical bucket, connecting circle radius to cylinder height and volume.",
            "Cubes and cuboids transform through cutting and reassembling, highlighting surface area changes.",
            "Students compare prisms and pyramids by matching base area with height to compute volume."
        ],
        "Math.Geometry.TransformAndMotion": [
            "Figure tessellations show how translations and rotations tile a plane.",
            "Partitioning puzzles ask where to cut and re-arrange shapes to form new silhouettes.",
            "Mirror symmetry problems require tracking reflections across multiple axes."
        ],
        "Math.Geometry.PositionDirection": [
            "Spatial reasoning challenges ask students to track relative positions after multiple turns.",
            "Coordinate-plane clues describe treasure hunts requiring directional interpretation.",
            "Perspective drawings force learners to infer left/right orientation from descriptions."
        ],
        "Math.Geometry.Measurement": [
            "A wire forms a circle and then another planar figure while keeping the same perimeter, prompting comparisons of area.",
            "Problems convert between square centimeters, square meters, and square kilometers.",
            "Students deduce the area or perimeter after scaling linear dimensions."
        ]
    }
}

# Extend curated backgrounds for MedXpert (shared across modalities)
for dataset in ("medxpertqa_text", "medxpertqa_mm"):
    CURATED_BACKGROUND_SNIPPETS[dataset] = {
        "neurology": [
            "A patient presents with unilateral weakness and slurred speech; learners triage stroke vs. seizure.",
            "Migraine aura descriptions are compared with occipital lobe lesions."
        ],
        "emergency medicine": [
            "A trauma bay receives multiple victims requiring ABC stabilization priorities.",
            "Paramedics relay vital signs that dictate shock management pathways."
        ],
        "diagnostic imaging": [
            "A CT head without contrast is examined for subtle hyperdense hemorrhage.",
            "MRI sequences are compared to identify demyelinating plaques."
        ],
        "musculoskeletal disorders": [
            "Lower back pain cases ask whether symptoms match herniated disk or muscle strain.",
            "Patients present with joint swelling that could signify rheumatoid arthritis."
        ],
        "orthopedic surgery": [
            "Femur fractures are stabilized with traction vs. intramedullary nails.",
            "Postoperative care after total knee arthroplasty is reviewed."
        ],
        "infectious diseases": [
            "Persistent fever with travel history raises suspicion for malaria vs. typhoid.",
            "Outbreak tracing determines whether transmission is airborne or contact-based."
        ],
        "digestive system": [
            "Right-upper-quadrant pain and ultrasound findings differentiate cholecystitis from hepatitis.",
            "Chronic reflux prompts comparisons between GERD and peptic ulcer disease."
        ],
        "pharmacologic therapy": [
            "Dose adjustments are calculated for renal impairment when prescribing antibiotics.",
            "Chemotherapy regimens are sequenced based on toxicity profiles."
        ],
        "histopathology": [
            "Slides show glandular vs. squamous differentiation for tumor classification.",
            "Biopsy of inflamed colon mucosa must be matched to Crohn's vs. ulcerative colitis."
        ],
        "cardiovascular": [
            "Chest pain workups contrast STEMI, NSTEMI, and pericarditis clues.",
            "Echocardiography snippets highlight valvular disease progression."
        ],
        "reproductive health": [
            "Prenatal ultrasound findings guide counseling on fetal growth restriction.",
            "Infertility evaluations compare hormonal vs. structural causes."
        ],
        "radiation oncology": [
            "Treatment plans describe IMRT fields for head and neck tumors.",
            "Side effects from stereotactic body radiotherapy are discussed."
        ],
        "pediatric pulmonology": [
            "A wheezing child is evaluated for asthma severity classification.",
            "Cystic fibrosis airway clearance regimens are compared."
        ],
        "cardiac electrophysiology": [
            "Holter monitor tracings differentiate atrial flutter from atrial fibrillation.",
            "Pacemaker programming scenarios test AV synchrony understanding."
        ],
        "clinical management": [
            "Hospitalists must prioritize consults, imaging, and discharge planning for complex patients."
        ],
        "physical examination": [
            "Learners interpret heart and lung findings gathered at the bedside."
        ],
        "laboratory testing": [
            "Sequential lab draws show trends in inflammatory markers and electrolytes."
        ],
        "differential diagnosis": [
            "Constellations of symptoms are parsed to rule in or out overlapping conditions."
        ]
    }

DATASET_DIR_MAP = {
    'csbench_en_test': 'csbench_en_test',
    'csbench_cn_test': 'csbench_cn_test',
    'csbench_fr_test': 'csbench_fr_test',
    'csbench_de_test': 'csbench_de_test',
    'medxpertqa_text': 'medxpertqa_text',
    'medxpertqa_mm': 'medxpertqa_mm',
    'tombench_en': 'tombench_en',
    'tombench_cn': 'tombench_cn',
    'wemath': 'wemath'
}


def load_items(dataset: str, project_root: Path) -> Dict[str, Dict[str, Any]]:
    dataset_dir = DATASET_DIR_MAP.get(dataset)
    if not dataset_dir:
        raise ValueError(f"Unknown dataset {dataset}")
    dataset_path = project_root / "outputs" / "phase0_datasets" / dataset_dir / "dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    items = {}
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            record = json.loads(line)
            item_id = str(record.get("id", idx))
            items[item_id] = record
    return items


def summarize_question(item: Dict[str, Any]) -> str:
    for key in ("question", "prompt", "text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            text = value.strip().replace("\n", " ")
            return text[:200]
    return "<no question text>"


def main():
    parser = argparse.ArgumentParser(description="Build domain card YAML from parsed outputs.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., tombench_en)")
    parser.add_argument("--results", required=True, help="Path to parsed Stage-1 results JSONL")
    parser.add_argument("--output", required=True, help="Destination YAML path")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[2], help="Project root (default: ../..)")
    parser.add_argument("--samples-per-parent", type=int, default=10, help="Sample question count per super parent")
    args = parser.parse_args()

    dataset = args.dataset.strip()
    results_path = Path(args.results)
    output_path = Path(args.output)
    project_root = Path(args.project_root)
    dataset_backgrounds = CURATED_BACKGROUND_SNIPPETS.get(dataset, {})

    if not results_path.exists():
        raise FileNotFoundError(results_path)

    items = load_items(dataset, project_root)
    parents = defaultdict(lambda: {
        "count": 0,
        "mid_level": Counter(),
        "terms": Counter(),
        "samples": []
    })
    modality_visual_examples: List[str] = []
    has_visual = False

    total_records = 0
    valid_records = 0

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            total_records += 1
            item_id = str(record.get("item_id", ""))
            extraction = record.get("extraction") or {}
            if extraction.get("is_valid"):
                valid_records += 1
            canon = extraction.get("subdomains_canonical") or extraction.get("subdomains") or []
            native = extraction.get("subdomains_native") or extraction.get("subdomains") or []
            terms = extraction.get("terms") or []
            visual = extraction.get("visual_facts") or []

            if visual:
                has_visual = True
                modality_visual_examples.extend([v for v in visual if isinstance(v, str)])

            for parent in canon:
                if not isinstance(parent, str):
                    continue
                entry = parents[parent]
                entry["count"] += 1
                for mid in native:
                    if isinstance(mid, str):
                        entry["mid_level"][mid] += 1
                for term in terms:
                    if isinstance(term, str):
                        entry["terms"][term] += 1

                item = items.get(item_id, {})
                background = summarize_question(item)
                if len(entry["samples"]) < args.samples_per_parent:
                    sample = {
                        "item_id": item_id,
                        "question": background,
                        "top_terms": [t for t in terms[:5] if isinstance(t, str)],
                    }
                    filtered_visual = [v for v in visual if isinstance(v, str)]
                    if filtered_visual:
                        sample["visual_facts"] = filtered_visual
                    entry["samples"].append(sample)

    modality_section = {
        "text": True,
        "multimodal": has_visual
    }
    if has_visual:
        unique_visual = []
        seen_visual = set()
        for fact in modality_visual_examples:
            if fact not in seen_visual:
                seen_visual.add(fact)
                unique_visual.append(fact)
            if len(unique_visual) >= 5:
                break
        modality_section["visual_examples"] = unique_visual

    super_parents = []
    glossary_entries = []
    samples_section = []
    for parent, data in sorted(parents.items(), key=lambda x: -x[1]["count"]):
        mid_level = [
            {"label": label, "count": count}
            for label, count in data["mid_level"].most_common(10)
        ]
        typical_terms = [term for term, _ in data["terms"].most_common(15)]
        curated_background = dataset_backgrounds.get(parent) if dataset.startswith("tombench") else None
        if curated_background:
            backgrounds = curated_background
        else:
            backgrounds = [sample["question"] for sample in data["samples"]]
        parent_entry = {
            "super_parent": parent,
            "count": data["count"],
            "mid_level_parents": mid_level
        }
        if dataset.startswith("tombench"):
            parent_entry["background_snippets"] = backgrounds[:args.samples_per_parent]
        super_parents.append(parent_entry)
        glossary_entries.append({
            "super_parent": parent,
            "typical_terms": typical_terms
        })
        samples_section.append({
            "super_parent": parent,
            "examples": data["samples"]
        })

    coverage = valid_records / total_records if total_records else 0.0

    diagnostics = {
        "coverage": round(coverage, 3),
        "jaccard": "n/a (single-run domain card)",
        "bias_notes": []
    }

    enums = {
        "recommended": []
    }
    if dataset.startswith("tombench"):
        enums["recommended"] = sorted(TOM_SUPER_TAGS.keys())
    elif dataset.startswith("csbench"):
        enums["recommended"] = sorted(CS_SUPER_PARENTS.keys())
    elif dataset.startswith("medxpertqa"):
        enums["recommended"] = sorted(set(SPECIALTY_MAP.keys()) | set(TASK_AXIS_MAP.keys()))
    elif dataset == "wemath":
        enums["recommended"] = sorted(WEMATH_SUPER_PARENTS.keys())

    domain_card = {
        "meta": {
            "dataset": dataset,
            "results_source": str(results_path),
            "oracle": "Stage-1 parsed outputs",
            "total_items": len(items),
            "parsed_records": total_records,
            "modality": modality_section
        },
        "ontology": super_parents,
        "glossary": glossary_entries,
        "samples": samples_section,
        "diagnostics": diagnostics,
        "enums": enums
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(domain_card, f, sort_keys=False, allow_unicode=True)

    print(f"Domain card saved to {output_path}")


if __name__ == "__main__":
    main()

