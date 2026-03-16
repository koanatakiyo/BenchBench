"""
Shared Theory-of-Mind taxonomy (EN + ZH) used to normalize ToMBench outputs.
"""

from typing import Dict, List

TOM_SUPER_TAGS: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "ToM.BeliefReasoning": {
        "aliases": {
            "en": [
                "belief reasoning",
                "false belief",
                "true belief",
                "second-order belief",
                "theory of mind",
                "perspective taking"
            ],
            "zh": ["信念推理", "错误信念", "假信念", "二级信念", "视角换位"]
        }
    },
    "ToM.EmotionRecognition": {
        "aliases": {
            "en": [
                "emotion recognition",
                "emotion attribution",
                "emotion inference",
                "affect labeling"
            ],
            "zh": ["情绪识别", "情绪判断", "情感识别", "情绪推断"]
        }
    },
    "ToM.SocialInteraction": {
        "aliases": {
            "en": [
                "social interaction",
                "social dynamics",
                "interpersonal relationship",
                "relationship reasoning"
            ],
            "zh": ["人际关系", "社交互动", "人际互动", "社交推理"]
        }
    },
    "ToM.NonLiteralCommunication": {
        "aliases": {
            "en": [
                "non-literal communication",
                "pragmatic inference",
                "pragmatics",
                "irony",
                "sarcasm",
                "faux pas"
            ],
            "zh": ["非字面交流", "语用推理", "语用", "讽刺", "反语", "失礼行为"]
        }
    },
    "ToM.ObjectLocation": {
        "aliases": {
            "en": [
                "object location",
                "object tracking",
                "spatial tracking",
                "location reasoning"
            ],
            "zh": ["物体位置", "物体追踪", "位置推理"]
        }
    },
    "ToM.IntentionReasoning": {
        "aliases": {
            "en": [
                "intention reasoning",
                "intention explanation",
                "goal inference",
                "intention inference"
            ],
            "zh": ["意图推理", "意图解释", "目标推断"]
        }
    },
    "ToM.MoralJudgment": {
        "aliases": {
            "en": [
                "moral judgment",
                "moral reasoning",
                "ethical reasoning",
                "fairness evaluation"
            ],
            "zh": ["道德判断", "道德推理", "公平判断", "伦理推理"]
        }
    }
}

SCENARIO_ALIASES = {
    "school life": ["school life", "classroom", "campus", "teacher", "students"],
    "restaurant scenario": ["restaurant", "cafe", "coffee shop", "diner", "dining hall"],
    "workplace dynamics": ["workplace", "office", "manager", "coworker", "company"],
    "project management": ["project", "project management", "team project"],
    "art exhibition": ["art exhibition", "gallery", "museum", "art show"],
    "outdoor activities": ["outdoor", "park", "playground", "camping", "hiking", "community park"]
}

BELIEF_KEYWORDS = {
    "en": [
        "belief", "believe", "think", "knows", "know", "false belief", "second-order",
        "perspective", "mindreading", "deception", "ignorant"
    ],
    "zh": ["信念", "以为", "认为", "觉得", "假信念", "错误信念", "视角", "知道", "不知道", "欺骗"]
}

EMOTION_KEYWORDS = {
    "en": [
        "happy", "sad", "angry", "afraid", "fear", "emotion", "feeling", "excited",
        "jealous", "worry", "anxious"
    ],
    "zh": ["开心", "高兴", "难过", "悲伤", "生气", "愤怒", "害怕", "紧张", "担心", "嫉妒", "情绪"]
}

SOCIAL_KEYWORDS = {
    "en": [
        "friend", "neighbor", "relationship", "interaction", "social",
        "family", "date", "romantic", "team"
    ],
    "zh": ["朋友", "邻居", "关系", "社交", "人际", "家庭", "约会", "同学", "同事"]
}

NON_LITERAL_KEYWORDS = {
    "en": ["irony", "sarcasm", "joke", "pragmatic", "faux pas", "metaphor"],
    "zh": ["讽刺", "反语", "玩笑", "语用", "失礼", "比喻"]
}

OBJECT_LOCATION_KEYWORDS = {
    "en": ["object", "location", "where", "search", "hide", "move", "box", "cabinet"],
    "zh": ["物体", "位置", "在哪里", "寻找", "藏", "移动", "盒子", "柜子"]
}

INTENTION_KEYWORDS = {
    "en": ["intend", "intention", "plan", "goal", "want", "purpose", "decide"],
    "zh": ["意图", "打算", "计划", "目的", "想要", "决定"]
}

MORAL_KEYWORDS = {
    "en": ["moral", "ethical", "fair", "unfair", "should", "shouldn't", "honest", "rule"],
    "zh": ["道德", "伦理", "公平", "不公平", "应该", "不应该", "诚实", "规则", "惩罚", "奖励"]
}

KEYWORD_MAP = {
    "ToM.BeliefReasoning": BELIEF_KEYWORDS,
    "ToM.EmotionRecognition": EMOTION_KEYWORDS,
    "ToM.SocialInteraction": SOCIAL_KEYWORDS,
    "ToM.NonLiteralCommunication": NON_LITERAL_KEYWORDS,
    "ToM.ObjectLocation": OBJECT_LOCATION_KEYWORDS,
    "ToM.IntentionReasoning": INTENTION_KEYWORDS,
    "ToM.MoralJudgment": MORAL_KEYWORDS
}

