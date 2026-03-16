# Stage 1 Domain Card Visualization
# Generates a sunburst chart (HTML + PNG) from a domain card YAML.

from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import yaml

try:
    import pandas as pd  # type: ignore
    import plotly.express as px  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    pd = None
    px = None


PASTEL = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
]


def load_domain_card(card_path: Path) -> Tuple[str, List[dict]]:
    """Load dataset name and ontology list from a domain card YAML."""
    with card_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    dataset = data.get("meta", {}).get("dataset", card_path.stem)
    ontology = data.get("ontology", [])
    if not ontology:
        raise ValueError(f"No ontology found in {card_path}")
    return dataset, ontology


def build_dataframe(ontology: List[dict], root_label: str) -> pd.DataFrame:
    """Build a flat dataframe suitable for a Plotly sunburst."""
    rows = [
        {
            "parent": "",
            "label": root_label,
            "value": sum(sp.get("count", 0) for sp in ontology),
            "level": "root",
        }
    ]
    for sp in ontology:
        super_parent = sp.get("super_parent", "unknown")
        count = sp.get("count", 0)
        rows.append(
            {
                "parent": root_label,
                "label": super_parent,
                "value": count,
                "level": "super_parent",
            }
        )
        for mid in sp.get("mid_level_parents", []):
            rows.append(
                {
                    "parent": super_parent,
                    "label": mid.get("label", "unknown"),
                    "value": mid.get("count", 0),
                    "level": "mid_level",
                }
            )
    return pd.DataFrame(rows)


def render_and_save(df: pd.DataFrame, out_dir: Path, dataset: str) -> None:
    if px is None:
        raise RuntimeError("plotly is not installed; cannot render via plotly.")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = px.sunburst(
        df,
        path=["parent", "label"],
        values="value",
        color="label",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title=f"{dataset} Domain Card Ontology",
        width=1200,
        height=800,
    )
    fig.update_traces(
        textinfo="label+value",
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>Cases: %{value}<extra></extra>",
    )
    fig.add_annotation(
        x=0.5,
        y=-0.15,
        xref="paper",
        yref="paper",
        text=(
            "<i>Note: No native ‘subdomain’ level. "
            "mid_level_parents are the official aggregation units for benchmark analysis.</i>"
        ),
        showarrow=False,
        font=dict(size=12, color="gray"),
    )

    html_path = out_dir / f"{dataset}_domain_card_sunburst.html"
    png_path = out_dir / f"{dataset}_domain_card.png"
    fig.write_html(html_path)
    try:
        fig.write_image(png_path, width=1200, height=800, scale=2)
        print(f"✅ Generated: {html_path}, {png_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ PNG export skipped for {dataset}: {exc}")
        print(f"✅ Generated: {html_path}")


def _svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _polar(cx: float, cy: float, r: float, ang: float) -> tuple[float, float]:
    return (cx + r * math.cos(ang), cy + r * math.sin(ang))


def _annular_sector_path(
    *,
    cx: float,
    cy: float,
    r_in: float,
    r_out: float,
    a0: float,
    a1: float,
) -> str:
    # Angles in radians; a1 > a0; angle increases clockwise (because y increases downward).
    if a1 <= a0:
        raise ValueError("Expected a1 > a0")
    large_arc = 1 if (a1 - a0) > math.pi else 0
    x0, y0 = _polar(cx, cy, r_out, a0)
    x1, y1 = _polar(cx, cy, r_out, a1)
    x2, y2 = _polar(cx, cy, r_in, a1)
    x3, y3 = _polar(cx, cy, r_in, a0)
    # Outer arc sweep=1 (clockwise), inner arc sweep=0 (counterclockwise back).
    return (
        f"M {x0:.3f} {y0:.3f} "
        f"A {r_out:.3f} {r_out:.3f} 0 {large_arc} 1 {x1:.3f} {y1:.3f} "
        f"L {x2:.3f} {y2:.3f} "
        f"A {r_in:.3f} {r_in:.3f} 0 {large_arc} 0 {x3:.3f} {y3:.3f} Z"
    )


def _svg_text(
    *,
    x: float,
    y: float,
    lines: list[str],
    size: int,
    fill: str,
    anchor: str = "middle",
    weight: int | None = None,
    rotate_deg: float | None = None,
) -> str:
    attrs = [
        f'x="{x:.3f}"',
        f'y="{y:.3f}"',
        f'font-size="{size}"',
        f'fill="{fill}"',
        f'text-anchor="{anchor}"',
        "font-family=\"Arial, Helvetica, 'DejaVu Sans', sans-serif\"",
    ]
    if weight is not None:
        attrs.append(f'font-weight="{weight}"')
    if rotate_deg is not None:
        attrs.append(f'transform="rotate({rotate_deg:.3f} {x:.3f} {y:.3f})"')
    out = [f"<text {' '.join(attrs)}>"]
    if not lines:
        out.append("</text>")
        return "".join(out)
    out.append(f"<tspan x=\"{x:.3f}\" dy=\"0\">{_svg_escape(lines[0])}</tspan>")
    for ln in lines[1:]:
        out.append(f"<tspan x=\"{x:.3f}\" dy=\"{size * 1.05:.1f}\">{_svg_escape(ln)}</tspan>")
    out.append("</text>")
    return "".join(out)


def _build_color_map(dataset: str, ontology: list[dict]) -> dict[str, str]:
    cmap: dict[str, str] = {}

    def add(label: str) -> None:
        if label not in cmap:
            cmap[label] = PASTEL[len(cmap) % len(PASTEL)]

    add(dataset)
    for sp in ontology:
        add(str(sp.get("super_parent", "unknown")))
        for mid in sp.get("mid_level_parents", []) or []:
            add(str(mid.get("label", "unknown")))
    return cmap


def _build_sunburst_svg(dataset: str, ontology: list[dict], *, width: int = 1200, height: int = 800) -> str:
    total = int(sum(float(sp.get("count", 0) or 0) for sp in ontology))
    # Layout tuned to roughly match the Plotly output size/margins.
    cx, cy = width * 0.55, height * 0.52
    r_root = 140.0
    r_sp_in, r_sp_out = r_root, 220.0
    r_mid_in, r_mid_out = r_sp_out, 300.0
    start_angle = -math.pi / 2  # start at top

    cmap = _build_color_map(dataset, ontology)

    svg: list[str] = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />')
    svg.append(
        _svg_text(
            x=45,
            y=44,
            lines=[f"{dataset} Domain Card Ontology"],
            size=22,
            fill="#2a3f5f",
            anchor="start",
            weight=600,
        )
    )

    # Root disc.
    svg.append(f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r_root:.3f}" fill="{cmap.get(dataset, "#66c2a5")}" stroke="#ffffff" stroke-width="1.5" />')
    svg.append(
        _svg_text(
            x=cx,
            y=cy - 10,
            lines=[dataset, f"{total}"],
            size=14,
            fill="#2a3f5f",
            anchor="middle",
            weight=600,
        )
    )

    # Super-parent ring.
    sp_total = sum(float(sp.get("count", 0) or 0) for sp in ontology) or 1.0
    a = start_angle
    for sp in ontology:
        sp_label = str(sp.get("super_parent", "unknown"))
        sp_val = float(sp.get("count", 0) or 0)
        if sp_val <= 0:
            continue
        span = (sp_val / sp_total) * 2 * math.pi
        a0, a1 = a, a + span
        a = a1
        d = _annular_sector_path(cx=cx, cy=cy, r_in=r_sp_in, r_out=r_sp_out, a0=a0, a1=a1)
        svg.append(f'<path d="{d}" fill="{cmap.get(sp_label, "#b3b3b3")}" stroke="#ffffff" stroke-width="1.2" />')

        # Mid-level ring, nested within this super-parent.
        mids = sp.get("mid_level_parents", []) or []
        mid_total = sum(float(m.get("count", 0) or 0) for m in mids) or 0.0
        if mid_total > 0:
            b = a0
            for mid in mids:
                mid_label = str(mid.get("label", "unknown"))
                mid_val = float(mid.get("count", 0) or 0)
                if mid_val <= 0:
                    continue
                mid_span = (mid_val / mid_total) * (a1 - a0)
                b0, b1 = b, min(a1, b + mid_span)
                b = b1
                if b1 - b0 <= 1e-6:
                    continue
                d2 = _annular_sector_path(cx=cx, cy=cy, r_in=r_mid_in, r_out=r_mid_out, a0=b0, a1=b1)
                svg.append(f'<path d="{d2}" fill="{cmap.get(mid_label, "#b3b3b3")}" stroke="#ffffff" stroke-width="1.0" />')

                # Label mid segments only if there's enough room.
                mid_deg = (b1 - b0) * 180.0 / math.pi
                arc_len = ((b1 - b0) * ((r_mid_in + r_mid_out) / 2.0))
                need = max(48.0, len(mid_label) * 6.2)
                if mid_deg >= 8.0 and arc_len >= need:
                    am = (b0 + b1) / 2.0
                    deg = (am * 180.0 / math.pi) % 360.0
                    rot = deg
                    if 90.0 < deg < 270.0:
                        rot = deg + 180.0
                    tx, ty = _polar(cx, cy, (r_mid_in + r_mid_out) / 2.0, am)
                    svg.append(
                        _svg_text(
                            x=tx,
                            y=ty,
                            lines=[mid_label, f"{int(mid_val)}"],
                            size=10,
                            fill="#444444",
                            anchor="middle",
                            rotate_deg=rot,
                            weight=500,
                        )
                    )

        # Label super-parent segments if there's enough room.
        sp_deg = (a1 - a0) * 180.0 / math.pi
        arc_len_sp = ((a1 - a0) * ((r_sp_in + r_sp_out) / 2.0))
        need_sp = max(70.0, len(sp_label) * 6.2)
        if sp_deg >= 10.0 and arc_len_sp >= need_sp:
            am = (a0 + a1) / 2.0
            deg = (am * 180.0 / math.pi) % 360.0
            rot = deg
            if 90.0 < deg < 270.0:
                rot = deg + 180.0
            tx, ty = _polar(cx, cy, (r_sp_in + r_sp_out) / 2.0, am)
            svg.append(
                _svg_text(
                    x=tx,
                    y=ty,
                    lines=[sp_label, f"{int(sp_val)}"],
                    size=11,
                    fill="#444444",
                    anchor="middle",
                    rotate_deg=rot,
                    weight=500,
                )
            )

    svg.append(
        _svg_text(
            x=width / 2,
            y=height - 28,
            lines=[
                "Note: No native ‘subdomain’ level. mid_level_parents are the official aggregation units for benchmark analysis."
            ],
            size=12,
            fill="#888888",
            anchor="middle",
            weight=400,
        )
    )
    svg.append("</svg>")
    return "\n".join(svg)


def render_and_save_minimal(ontology: list[dict], out_dir: Path, dataset: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{dataset}_domain_card_sunburst.html"
    png_path = out_dir / f"{dataset}_domain_card.png"

    svg = _build_sunburst_svg(dataset, ontology, width=1200, height=800)
    html = (
        "<!doctype html>\n"
        "<html>\n"
        "<head><meta charset=\"utf-8\" />"
        f"<title>{_svg_escape(dataset)} Domain Card Ontology</title></head>\n"
        "<body style=\"margin:0;background:#ffffff\">\n"
        f"{svg}\n"
        "</body>\n"
        "</html>\n"
    )
    html_path.write_text(html, encoding="utf-8")

    # Rasterize SVG to PNG using ImageMagick (convert). Use density=192 to get 2x (2400×1600).
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".svg", delete=False) as tmp:
            tmp.write(svg)
            tmp_path = Path(tmp.name)
        subprocess.run(
            ["convert", "-density", "192", str(tmp_path), str(png_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"✅ Generated (minimal): {html_path}, {png_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ PNG export skipped for {dataset}: {exc}")
        print(f"✅ Generated (minimal): {html_path}")
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def run_for_cards(cards: Iterable[Path], out_dir: Path | None) -> None:
    for card_path in cards:
        dataset, ontology = load_domain_card(card_path)
        target_dir = out_dir if out_dir else card_path.parent
        print(f"Rendering {dataset} from {card_path} -> {target_dir}")
        if px is not None and pd is not None:
            df = build_dataframe(ontology, root_label=dataset)
            render_and_save(df, target_dir, dataset)
        else:
            render_and_save_minimal(ontology, target_dir, dataset)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render domain card ontology sunburst.")
    parser.add_argument(
        "--cards",
        nargs="+",
        type=Path,
        help="Path(s) to domain_card YAML files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Optional output directory for figures (defaults to card location).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.cards:
        run_for_cards(args.cards, args.out_dir)
    else:
        # Fallback: render a small demo using the sample ontology if no cards provided.
        print("No --cards supplied; rendering demo figure.")
        sample_ontology = [
            {
                "super_parent": "demo_super",
                "count": 10,
                "mid_level_parents": [
                    {"label": "mid_a", "count": 6},
                    {"label": "mid_b", "count": 4},
                ],
            }
        ]
        df = build_dataframe(sample_ontology, root_label="demo")
        render_and_save(df, Path("."), dataset="demo")


if __name__ == "__main__":
    main()
