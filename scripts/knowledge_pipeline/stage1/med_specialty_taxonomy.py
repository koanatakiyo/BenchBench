"""
Mappings for MedXpertQA canonical parents to specialty vs. task/modality axes.
"""

from typing import Dict, List

SPECIALTY_MAP: Dict[str, List[str]] = {
    "neurology": ["neurology", "neurocritical care", "neuro"],
    "emergency medicine": ["emergency medicine", "acute care", "trauma"],
    "musculoskeletal disorders": ["musculoskeletal disorders", "musculoskeletal", "rheumatology"],
    "orthopedic surgery": ["orthopedic surgery", "orthopedics"],
    "infectious diseases": ["infectious diseases", "infectious disease", "id"],
    "digestive system": ["digestive system", "gastroenterology"],
    "histopathology": ["histopathology", "pathology"],
    "cardiovascular": ["cardiovascular", "cardiology"],
    "reproductive health": ["reproductive health", "obstetrics", "gynecology"],
    "radiation oncology": ["radiation oncology", "rad onc"],
    "pediatric pulmonology": ["pediatric pulmonology", "pediatric respiratory"],
    "cardiac electrophysiology": ["cardiac electrophysiology", "electrophysiology"],
    "pediatrics": ["pediatrics", "pediatric medicine"],
    "nephrology": ["nephrology", "renal"],
    "endocrinology": ["endocrinology", "endocrine"],
    "dermatology": ["dermatology", "derm"]
}

TASK_AXIS_MAP: Dict[str, List[str]] = {
    "diagnostic imaging": ["diagnostic imaging", "medical imaging", "radiology", "imaging"],
    "pharmacologic therapy": ["pharmacologic therapy", "pharmacology", "drug therapy", "pharmacotherapy"],
    "clinical management": ["clinical management", "management plan", "treatment planning"],
    "physical examination": ["physical examination", "clinical examination", "phys exam"],
    "laboratory testing": ["laboratory testing", "lab diagnostics", "laboratory diagnostics"],
    "differential diagnosis": ["differential diagnosis", "differential reasoning"],
    "surgical planning": ["surgical planning", "perioperative management"]
}


def lookup_category(label: str, mapping: Dict[str, List[str]]) -> str:
    norm = label.lower().strip()
    for category, aliases in mapping.items():
        for alias in aliases:
            if alias in norm:
                return category
    return ""


