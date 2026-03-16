"""
WeMath geometry super-parent mapping.
"""

from typing import List, Set

WEMATH_SUPER_PARENTS = {
    "Math.Geometry.PlaneFigures": [
        "triangle", "quadrilateral", "trapezoid", "plane figure",
        "polygon", "folding", "net", "平面图形", "三角形", "四边形", "梯形", "展开图"
    ],
    "Math.Geometry.SolidFigures": [
        "solid figure", "solid geometry", "cube", "cuboid", "prism",
        "pyramid", "surface area", "volume", "solid mensuration",
        "立体图形", "立体几何", "正方体", "长方体", "棱柱", "棱锥", "表面积", "体积"
    ],
    "Math.Geometry.TransformAndMotion": [
        "transformation", "translation", "rotation", "reflection",
        "tessellation", "partition", "motion", "折叠", "旋转", "平移", "对称", "铺砌"
    ],
    "Math.Geometry.PositionDirection": [
        "spatial reasoning", "position", "direction", "coordinate",
        "location", "视角", "方位", "位置关系", "空间推理"
    ],
    "Math.Geometry.Measurement": [
        "area", "perimeter", "measurement", "unit conversion",
        "length", "area calculation", "perimeter calculation",
        "面积", "周长", "测量", "单位换算", "长度"
    ]
}


def map_wemath_parents(subdomains: List[str]) -> List[str]:
    tags: Set[str] = set()
    for sub in subdomains:
        if not isinstance(sub, str):
            continue
        norm = sub.lower()
        for super_parent, keywords in WEMATH_SUPER_PARENTS.items():
            if any(keyword.lower() in norm for keyword in keywords):
                tags.add(super_parent)
    return sorted(tags)

