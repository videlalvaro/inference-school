#!/usr/bin/env python3
"""Generate the Inference School companion manuscript from curriculum source files."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import textwrap
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

BOOK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BOOK_DIRECTORY.parent
PROBLEMS_DIRECTORY = REPOSITORY_ROOT / "Problems"
EXERCISES_DIRECTORY = REPOSITORY_ROOT / "Sources" / "InferenceSchoolExercises"
SOLUTIONS_DIRECTORY = REPOSITORY_ROOT / "Sources" / "InferenceSchoolSolutions"
GITHUB_BLOB_ROOT = "https://github.com/videlalvaro/inference-school/blob/main"

PDF_FORMAT = "pdf"
EPUB_FORMAT = "epub"
SUPPORTED_EXTERNAL_LINK_SCHEMES = {"http", "https", "mailto"}
EPUB_ASSET_MARKER = ".generated"
EPUB_ASSET_MARKER_CONTENT = "Inference School EPUB assets\n"

ORIENTATION_CHAPTER_ID = "000"
COURSE_CHAPTER_IDS = tuple(f"{chapter:03d}" for chapter in range(1, 48))
FULL_BOOK_CHAPTER_IDS = (ORIENTATION_CHAPTER_ID, *COURSE_CHAPTER_IDS)
MAXIMUM_HORIZONTAL_DIAGRAM_ASPECT_RATIO = 6.0
HORIZONTAL_LEVEL_SPACING = 4.25
VERTICAL_GROUP_SPACING = 3.75
VERTICAL_LEVEL_SPACING = 2.35

APPENDICES = (
    ("one-token", REPOSITORY_ROOT / "docs" / "ONE-TOKEN.md"),
    ("math-primer", REPOSITORY_ROOT / "docs" / "MATH-PRIMER.md"),
)

PARTS = OrderedDict(
    [
        ("Foundations", range(1, 2)),
        ("Tensors and Dense Linear Algebra", range(2, 7)),
        ("Neural-Network Operators", range(7, 13)),
        ("Positions and Attention", range(13, 22)),
        ("KV-Cache Engineering", range(22, 29)),
        ("Weight Quantization", range(29, 35)),
        ("Assemble Inference", range(35, 43)),
        ("Tradeoffs from Evidence", range(43, 48)),
    ]
)

SHARED_SOLUTION_FILES = {
    "029": [SOLUTIONS_DIRECTORY / "WeightQuantizationSolutionSupport.swift"],
    "039": [SOLUTIONS_DIRECTORY / "MiniDecoderCPUEngine.swift"],
}

DETAIL_PATTERN = re.compile(
    r"<details>\s*<summary>(.*?)</summary>\s*(.*?)\s*</details>",
    re.DOTALL,
)
MERMAID_PATTERN = re.compile(
    r"```mermaid(?:\s+\{#(?P<id>[A-Za-z0-9_-]+)\})?\s*\n"
    r"(?P<source>.*?)\n```",
    re.DOTALL,
)
EDGE_PATTERN = re.compile(r"\s*-->\s*(?:\|\"(.*?)\"\|\s*)?")
NODE_PATTERN = re.compile(
    r"^(?P<id>[A-Za-z][A-Za-z0-9_]*)\s*"
    r"(?:(?:\[\"(?P<rect>.*)\"\])|(?:\{\"(?P<diamond>.*)\"\}))?$"
)
MARKDOWN_LINK_PATTERN = re.compile(
    r"(?P<prefix>!?\[[^\]]*\]\()"
    r"(?P<target>[^)\s]+)"
    r"(?P<suffix>(?:\s+[\"'][^)\n]*[\"'])?\))"
)

EPUB_INTERNAL_DOCUMENTS = {
    (REPOSITORY_ROOT / "docs" / "ONE-TOKEN.md").resolve(): "appendix-one-token",
    (REPOSITORY_ROOT / "docs" / "MATH-PRIMER.md").resolve(): "appendix-math-primer",
}


@dataclass(frozen=True)
class DiagramNode:
    identifier: str
    label: str
    decision: bool = False


@dataclass(frozen=True)
class DiagramEdge:
    source: str
    target: str
    label: str | None = None


@dataclass(frozen=True)
class Flowchart:
    identifier: str
    direction: str
    nodes: OrderedDict[str, DiagramNode]
    edges: list[DiagramEdge]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chapters",
        nargs="*",
        help="Three-digit chapter IDs to generate. Defaults to all chapters.",
    )
    parser.add_argument(
        "--format",
        choices=(PDF_FORMAT, EPUB_FORMAT),
        default=PDF_FORMAT,
        help="Output format. Defaults to pdf.",
    )
    parser.add_argument(
        "--asset-directory",
        type=Path,
        help="Directory for generated EPUB assets. Required for EPUB output.",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def chapter_directories() -> dict[str, Path]:
    chapters: dict[str, Path] = {}
    for directory in sorted(PROBLEMS_DIRECTORY.glob("[0-9][0-9][0-9]-*")):
        chapter_id = directory.name[:3]
        readme = directory / "README.md"
        if readme.is_file():
            chapters[chapter_id] = directory
    return chapters


def selected_chapters(requested: list[str] | None) -> list[tuple[str, Path]]:
    available = chapter_directories()
    if not requested:
        requested = sorted(available)
    normalized = [chapter.zfill(3) for chapter in requested]
    unknown = sorted(set(normalized) - set(available))
    if unknown:
        raise SystemExit(f"Unknown chapter IDs: {', '.join(unknown)}")
    return [(chapter_id, available[chapter_id]) for chapter_id in normalized]


def load_solution_notes() -> dict[str, dict[str, object]]:
    note_files = [BOOK_DIRECTORY / "solution_notes.json"]
    note_files.extend(sorted((BOOK_DIRECTORY / "solution_notes").glob("*.json")))
    notes: dict[str, dict[str, object]] = {}
    for path in note_files:
        fragment = json.loads(path.read_text(encoding="utf-8"))
        duplicates = sorted(set(notes) & set(fragment))
        if duplicates:
            raise ValueError(
                f"Duplicate solution notes in {path.name}: {', '.join(duplicates)}"
            )
        notes.update(fragment)
    return notes


def normalize_chapter_title(markdown: str, chapter_id: str) -> tuple[str, str]:
    match = re.match(
        r"^#\s+(?:Problem\s+)?(?:\d{3}:\s*)?([^\n]+?)[ \t]*(?:\n|$)",
        markdown,
    )
    if not match:
        raise ValueError(f"Chapter {chapter_id} has no level-one title.")
    title = match.group(1)
    replacement = f"# {title} {{#chapter-{chapter_id}}}\n"
    return title, replacement + markdown[match.end() :]


def remove_canonical_section(markdown: str) -> str:
    markdown = re.sub(
        r"\n## Canonical solution\s*\n.*?(?=\n## |\Z)",
        "\n",
        markdown,
        flags=re.DOTALL,
    )
    return markdown.replace("## Hints and canonical solution", "## Hints")


def convert_details(markdown: str, output_format: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        title = html.unescape(match.group(1).strip())
        body = match.group(2).strip()
        if "canonical" in title.lower():
            return ""
        if output_format == EPUB_FORMAT:
            return epub_callout("hint", body, title)
        return (
            f"\\begin{{hintbox}}[title={{{tex_escape(title)}}}]\n\n"
            f"{body}\n\n"
            "\\end{hintbox}"
        )

    return DETAIL_PATTERN.sub(replacement, markdown)


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def epub_callout(class_name: str, body: str, title: str) -> str:
    escaped_title = html.escape(title, quote=False)
    return (
        f"::: {{.{class_name}}}\n\n"
        f'<p class="callout-title">{escaped_title}</p>\n\n'
        f"{body.strip()}\n\n"
        ":::"
    )


def internal_document_anchor(
    path: Path,
    included_chapter_ids: set[str],
    include_appendices: bool,
) -> str | None:
    resolved = path.resolve()
    if include_appendices and resolved in EPUB_INTERNAL_DOCUMENTS:
        return EPUB_INTERNAL_DOCUMENTS[resolved]
    try:
        relative = resolved.relative_to(PROBLEMS_DIRECTORY.resolve())
    except ValueError:
        return None
    if relative.name != "README.md" or len(relative.parts) != 2:
        return None
    chapter_id = relative.parts[0][:3]
    if chapter_id not in included_chapter_ids:
        return None
    return f"chapter-{chapter_id}"


def rewrite_epub_links(
    markdown: str,
    source_path: Path,
    included_chapter_ids: set[str],
    include_appendices: bool,
) -> str:
    def replacement(match: re.Match[str]) -> str:
        target = match.group("target")
        parsed = urlsplit(target)
        is_image = match.group("prefix").startswith("![")
        if target.startswith("#"):
            return match.group(0)
        if parsed.scheme or parsed.netloc:
            if is_image:
                raise ValueError(
                    f"EPUB image in {relative_path(source_path)} must be "
                    f"repository-local: {target}"
                )
            valid_web_link = (
                parsed.scheme in SUPPORTED_EXTERNAL_LINK_SCHEMES - {"mailto"}
                and bool(parsed.netloc)
            )
            valid_mail_link = parsed.scheme == "mailto" and not parsed.netloc
            if valid_web_link or valid_mail_link:
                return match.group(0)
            raise ValueError(
                f"EPUB link in {relative_path(source_path)} uses an unsupported "
                f"scheme: {target}"
            )
        if parsed.path.startswith("/"):
            raise ValueError(
                f"EPUB link in {relative_path(source_path)} is root-relative: {target}"
            )

        decoded_path = unquote(parsed.path)
        resolved = (source_path.parent / decoded_path).resolve()
        try:
            repository_path = resolved.relative_to(REPOSITORY_ROOT.resolve())
        except ValueError as error:
            raise ValueError(
                f"EPUB link in {relative_path(source_path)} escapes the repository: "
                f"{target}"
            ) from error
        if not resolved.exists():
            raise ValueError(
                f"EPUB link in {relative_path(source_path)} is missing: {target}"
            )

        if is_image:
            rewritten = f"../{quote(repository_path.as_posix(), safe='/')}"
        else:
            anchor = internal_document_anchor(
                resolved,
                included_chapter_ids,
                include_appendices,
            )
            if anchor is not None:
                rewritten = f"#{parsed.fragment or anchor}"
            else:
                rewritten = f"{GITHUB_BLOB_ROOT}/{repository_path.as_posix()}"
        if not (not is_image and rewritten.startswith("#")):
            if parsed.query:
                rewritten += f"?{parsed.query}"
            if parsed.fragment:
                rewritten += f"#{parsed.fragment}"
        return f"{match.group('prefix')}{rewritten}{match.group('suffix')}"

    return MARKDOWN_LINK_PATTERN.sub(replacement, markdown)


def render_frontmatter(
    output_format: str,
    included_chapter_ids: set[str],
    include_appendices: bool,
) -> str:
    path = BOOK_DIRECTORY / "frontmatter.md"
    frontmatter = path.read_text(encoding="utf-8").strip()
    if output_format == PDF_FORMAT:
        return frontmatter

    frontmatter = re.sub(
        r"\\begin\{chapterfocus\}\s*(.*?)\s*\\end\{chapterfocus\}",
        lambda match: epub_callout(
            "chapter-focus",
            match.group(1),
            "Chapter focus",
        ),
        frontmatter,
        flags=re.DOTALL,
    )
    frontmatter = re.sub(
        r"The PDF is vector-first\..*?under magnification\.",
        (
            "The EPUB preserves selectable equations and vector diagrams while "
            "adapting its layout to the reader's chosen text size and theme."
        ),
        frontmatter,
        flags=re.DOTALL,
    )
    return rewrite_epub_links(
        frontmatter,
        path,
        included_chapter_ids,
        include_appendices,
    )


def tex_diagram_label(value: str) -> str:
    label = tex_escape(value)
    for character, digit in zip("₀₁₂₃₄₅₆₇₈₉", "0123456789"):
        label = label.replace(character, rf"\textsubscript{{{digit}}}")
    return label


def parse_node(fragment: str, nodes: OrderedDict[str, DiagramNode]) -> str:
    match = NODE_PATTERN.match(fragment.strip())
    if not match:
        raise ValueError(f"Unsupported Mermaid node: {fragment!r}")
    identifier = match.group("id")
    label = match.group("rect") or match.group("diamond")
    if label is not None:
        label = label.replace(r"\n", "\n")
        nodes[identifier] = DiagramNode(
            identifier=identifier,
            label=label,
            decision=match.group("diamond") is not None,
        )
    elif identifier not in nodes:
        nodes[identifier] = DiagramNode(identifier=identifier, label=identifier)
    return identifier


def parse_flowchart(identifier: str, source: str) -> Flowchart:
    lines = [line.strip() for line in source.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"Diagram {identifier} is empty.")
    declaration = re.fullmatch(r"flowchart\s+(LR|RL|TD|TB)", lines[0])
    if not declaration:
        raise ValueError(f"Diagram {identifier} is not a supported flowchart.")
    nodes: OrderedDict[str, DiagramNode] = OrderedDict()
    edges: list[DiagramEdge] = []

    for line in lines[1:]:
        parts = EDGE_PATTERN.split(line)
        if len(parts) < 3:
            raise ValueError(f"Unsupported Mermaid statement in {identifier}: {line}")
        source_id = parse_node(parts[0], nodes)
        cursor = 1
        while cursor < len(parts):
            edge_label = parts[cursor] or None
            target_id = parse_node(parts[cursor + 1], nodes)
            edges.append(DiagramEdge(source=source_id, target=target_id, label=edge_label))
            source_id = target_id
            cursor += 2

    return Flowchart(
        identifier=identifier,
        direction=declaration.group(1),
        nodes=nodes,
        edges=edges,
    )


def flowchart_levels(flowchart: Flowchart) -> dict[str, int]:
    outgoing: dict[str, list[str]] = {identifier: [] for identifier in flowchart.nodes}
    incoming_count = {identifier: 0 for identifier in flowchart.nodes}
    for edge in flowchart.edges:
        outgoing[edge.source].append(edge.target)
        incoming_count[edge.target] += 1

    roots = [identifier for identifier in flowchart.nodes if incoming_count[identifier] == 0]
    if not roots:
        return breadth_first_flowchart_levels(flowchart, outgoing)
    levels: dict[str, int] = {root: 0 for root in roots}
    queue = deque(roots)
    remaining_incoming = incoming_count.copy()
    visited: set[str] = set()
    while queue:
        source = queue.popleft()
        visited.add(source)
        for target in outgoing[source]:
            levels[target] = max(levels.get(target, 0), levels[source] + 1)
            remaining_incoming[target] -= 1
            if remaining_incoming[target] == 0:
                queue.append(target)

    if len(visited) != len(flowchart.nodes):
        return breadth_first_flowchart_levels(flowchart, outgoing)

    return levels


def breadth_first_flowchart_levels(
    flowchart: Flowchart,
    outgoing: dict[str, list[str]],
) -> dict[str, int]:
    root = next(iter(flowchart.nodes))
    levels: dict[str, int] = {root: 0}
    queue = deque([root])
    while queue:
        source = queue.popleft()
        for target in outgoing[source]:
            if target not in levels:
                levels[target] = levels[source] + 1
                queue.append(target)

    next_level = max(levels.values(), default=-1) + 1
    for identifier in flowchart.nodes:
        if identifier not in levels:
            levels[identifier] = next_level
            next_level += 1
    return levels


def diagram_caption(identifier: str) -> str:
    caption = re.sub(r"^p\d{3}-", "", identifier).replace("-", " ")
    return caption[:1].upper() + caption[1:]


def horizontal_layout_is_readable(groups: OrderedDict[int, list[str]]) -> bool:
    level_span = max(groups, default=0) - min(groups, default=0)
    vertical_span = max((len(identifiers) - 1) * 1.55 for identifiers in groups.values())
    return vertical_span > 0 and (
        level_span * HORIZONTAL_LEVEL_SPACING / vertical_span
        <= MAXIMUM_HORIZONTAL_DIAGRAM_ASPECT_RATIO
    )


def flowchart_layout(
    flowchart: Flowchart,
) -> tuple[dict[str, int], dict[str, tuple[float, float]]]:
    levels = flowchart_levels(flowchart)
    groups: OrderedDict[int, list[str]] = OrderedDict()
    for identifier in flowchart.nodes:
        groups.setdefault(levels[identifier], []).append(identifier)

    coordinates: dict[str, tuple[float, float]] = {}
    horizontal = (
        flowchart.direction in {"LR", "RL"}
        and horizontal_layout_is_readable(groups)
    )
    reverse = flowchart.direction in {"RL"}
    for level, identifiers in groups.items():
        center = (len(identifiers) - 1) / 2
        for index, identifier in enumerate(identifiers):
            if horizontal:
                x = level * HORIZONTAL_LEVEL_SPACING * (-1 if reverse else 1)
                y = (center - index) * 1.55
            else:
                x = (index - center) * VERTICAL_GROUP_SPACING
                y = -level * VERTICAL_LEVEL_SPACING
            coordinates[identifier] = (x, y)
    return levels, coordinates


def tikz_flowchart(flowchart: Flowchart) -> str:
    levels, coordinates = flowchart_layout(flowchart)

    lines = [
        r"\begin{figure}[htbp]",
        r"\centering",
        r"\begin{adjustbox}{max width=\textwidth,max totalheight=0.68\textheight,keepaspectratio}",
        r"\begin{tikzpicture}",
    ]
    for identifier, node in flowchart.nodes.items():
        style = "diagram decision" if node.decision else "diagram node"
        x, y = coordinates[identifier]
        label = tex_diagram_label(node.label).replace("\n", r" \\ ")
        lines.append(
            f"\\node[{style}] ({identifier}) at ({x:.2f},{y:.2f}) {{{label}}};"
        )

    for edge in flowchart.edges:
        source_level = levels[edge.source]
        target_level = levels[edge.target]
        route = " to[bend left=18] " if target_level <= source_level else " -- "
        label = ""
        if edge.label:
            label = (
                " node[diagram edge label,pos=0.62] "
                f"{{{tex_escape(edge.label)}}} "
            )
        lines.append(
            f"\\draw[diagram edge] ({edge.source}){route}{label}({edge.target});"
        )

    lines.extend(
        [
            r"\end{tikzpicture}",
            r"\end{adjustbox}",
            f"\\caption{{{tex_escape(diagram_caption(flowchart.identifier))}.}}",
            f"\\label{{fig:{flowchart.identifier}}}",
            r"\end{figure}",
        ]
    )
    return "\n".join(lines)


def svg_label_lines(label: str) -> list[str]:
    lines: list[str] = []
    for source_line in label.splitlines() or [label]:
        lines.extend(
            textwrap.wrap(
                source_line,
                width=28,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    return lines


def svg_node_size(node: DiagramNode) -> tuple[float, float, list[str]]:
    lines = svg_label_lines(node.label)
    longest = max((len(line) for line in lines), default=1)
    width = max(150.0, min(270.0, longest * 7.4 + 36.0))
    height = max(58.0, len(lines) * 20.0 + 26.0)
    if node.decision:
        width += 34.0
        height += 18.0
    return width, height, lines


def svg_node_centers(
    flowchart: Flowchart,
    levels: dict[str, int],
    logical_coordinates: dict[str, tuple[float, float]],
    metrics: dict[str, tuple[float, float, list[str]]],
) -> dict[str, tuple[float, float]]:
    ordered_levels = sorted(set(levels.values()))
    if flowchart.direction not in {"TD", "TB"} or len(ordered_levels) <= 8:
        scale = 90.0
        return {
            identifier: (x * scale, -y * scale)
            for identifier, (x, y) in logical_coordinates.items()
        }

    identifiers_by_level: OrderedDict[int, list[str]] = OrderedDict()
    for identifier in flowchart.nodes:
        identifiers_by_level.setdefault(levels[identifier], []).append(identifier)

    levels_per_column = 6
    horizontal_node_gap = 52.0
    column_gap = 155.0
    row_gap = 205.0
    column_widths: list[float] = []
    column_count = (len(ordered_levels) + levels_per_column - 1) // levels_per_column
    for column in range(column_count):
        group_widths: list[float] = []
        for level in ordered_levels[
            column * levels_per_column : (column + 1) * levels_per_column
        ]:
            identifiers = identifiers_by_level[level]
            group_widths.append(
                sum(metrics[identifier][0] for identifier in identifiers)
                + max(0, len(identifiers) - 1) * horizontal_node_gap
            )
        column_widths.append(max(group_widths, default=150.0))

    column_centers: list[float] = []
    cursor = 0.0
    for width in column_widths:
        column_centers.append(cursor + width / 2)
        cursor += width + column_gap

    centers: dict[str, tuple[float, float]] = {}
    for level_index, level in enumerate(ordered_levels):
        column = level_index // levels_per_column
        row = level_index % levels_per_column
        display_row = row if column % 2 == 0 else levels_per_column - 1 - row
        identifiers = identifiers_by_level[level]
        group_width = (
            sum(metrics[identifier][0] for identifier in identifiers)
            + max(0, len(identifiers) - 1) * horizontal_node_gap
        )
        x = column_centers[column] - group_width / 2
        for identifier in identifiers:
            width = metrics[identifier][0]
            centers[identifier] = (x + width / 2, display_row * row_gap)
            x += width + horizontal_node_gap
    return centers


def svg_boundary_point(
    center: tuple[float, float],
    toward: tuple[float, float],
    size: tuple[float, float],
    decision: bool,
) -> tuple[float, float]:
    x, y = center
    dx = toward[0] - x
    dy = toward[1] - y
    if abs(dx) < 0.001 and abs(dy) < 0.001:
        return center
    half_width = size[0] / 2
    half_height = size[1] / 2
    if decision:
        scale = 1.0 / (
            abs(dx) / max(half_width, 0.001)
            + abs(dy) / max(half_height, 0.001)
        )
    else:
        x_scale = half_width / abs(dx) if abs(dx) >= 0.001 else float("inf")
        y_scale = half_height / abs(dy) if abs(dy) >= 0.001 else float("inf")
        scale = min(x_scale, y_scale)
    return x + dx * scale, y + dy * scale


def svg_flowchart(flowchart: Flowchart) -> str:
    levels, logical_coordinates = flowchart_layout(flowchart)
    metrics = {
        identifier: svg_node_size(node)
        for identifier, node in flowchart.nodes.items()
    }
    centers = svg_node_centers(flowchart, levels, logical_coordinates, metrics)
    padding = 74.0
    minimum_x = min(
        centers[identifier][0] - metrics[identifier][0] / 2
        for identifier in centers
    ) - padding
    maximum_x = max(
        centers[identifier][0] + metrics[identifier][0] / 2
        for identifier in centers
    ) + padding
    minimum_y = min(
        centers[identifier][1] - metrics[identifier][1] / 2
        for identifier in centers
    ) - padding
    maximum_y = max(
        centers[identifier][1] + metrics[identifier][1] / 2
        for identifier in centers
    ) + padding

    caption = f"{diagram_caption(flowchart.identifier)}."
    edge_summary = "; ".join(
        f"{flowchart.nodes[edge.source].label.replace(chr(10), ' ')} to "
        f"{flowchart.nodes[edge.target].label.replace(chr(10), ' ')}"
        + (f" ({edge.label})" if edge.label else "")
        for edge in flowchart.edges
    )
    title_id = f"{flowchart.identifier}-title"
    description_id = f"{flowchart.identifier}-description"
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<svg xmlns="http://www.w3.org/2000/svg" role="img" '
            f'aria-labelledby="{title_id} {description_id}" '
            f'viewBox="{minimum_x:.1f} {minimum_y:.1f} '
            f'{maximum_x - minimum_x:.1f} {maximum_y - minimum_y:.1f}">'
        ),
        f'<title id="{title_id}">{html.escape(caption)}</title>',
        (
            f'<desc id="{description_id}">'
            f"{html.escape('Flowchart: ' + edge_summary)}</desc>"
        ),
        "<defs>",
        (
            '<marker id="arrowhead" markerWidth="10" markerHeight="8" '
            'refX="9" refY="4" orient="auto" markerUnits="strokeWidth">'
        ),
        '<path d="M 0 0 L 10 4 L 0 8 z" fill="#3465a4"/>',
        "</marker>",
        "</defs>",
        "<style>",
        ".edge{fill:none;stroke:#3465a4;stroke-width:2.4}",
        ".node{fill:#f5f8fc;stroke:#204a87;stroke-width:2.2}",
        ".decision{fill:#fff7df;stroke:#8f5902;stroke-width:2.2}",
        ".node-label{fill:#17202a;font-family:sans-serif;font-size:16px;text-anchor:middle}",
        ".edge-label{fill:#204a87;font-family:sans-serif;font-size:14px;text-anchor:middle;paint-order:stroke;stroke:#fff;stroke-width:5px;stroke-linejoin:round}",
        "</style>",
        '<g class="edges">',
    ]

    for edge in flowchart.edges:
        source_center = centers[edge.source]
        target_center = centers[edge.target]
        source_node = flowchart.nodes[edge.source]
        target_node = flowchart.nodes[edge.target]
        source_size = metrics[edge.source][:2]
        target_size = metrics[edge.target][:2]
        start = svg_boundary_point(
            source_center,
            target_center,
            source_size,
            source_node.decision,
        )
        end = svg_boundary_point(
            target_center,
            source_center,
            target_size,
            target_node.decision,
        )
        if levels[edge.target] <= levels[edge.source]:
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = max((dx * dx + dy * dy) ** 0.5, 1.0)
            control = (
                (start[0] + end[0]) / 2 - dy / length * 62.0,
                (start[1] + end[1]) / 2 + dx / length * 62.0,
            )
            path = (
                f"M {start[0]:.1f} {start[1]:.1f} "
                f"Q {control[0]:.1f} {control[1]:.1f} {end[0]:.1f} {end[1]:.1f}"
            )
            label_position = (
                0.25 * start[0] + 0.5 * control[0] + 0.25 * end[0],
                0.25 * start[1] + 0.5 * control[1] + 0.25 * end[1] - 8.0,
            )
        else:
            path = f"M {start[0]:.1f} {start[1]:.1f} L {end[0]:.1f} {end[1]:.1f}"
            label_position = (
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 - 8.0,
            )
        lines.append(
            f'<path class="edge" d="{path}" fill="none" stroke="#3465a4" '
            'stroke-width="2.4" marker-end="url(#arrowhead)"/>'
        )
        if edge.label:
            lines.append(
                f'<text class="edge-label" fill="#204a87" font-family="sans-serif" '
                f'font-size="14" text-anchor="middle" paint-order="stroke" '
                f'stroke="#fff" stroke-width="5" stroke-linejoin="round" '
                f'x="{label_position[0]:.1f}" '
                f'y="{label_position[1]:.1f}">{html.escape(edge.label)}</text>'
            )
    lines.extend(["</g>", '<g class="nodes">'])

    for identifier, node in flowchart.nodes.items():
        x, y = centers[identifier]
        width, height, label_lines = metrics[identifier]
        if node.decision:
            points = " ".join(
                [
                    f"{x:.1f},{y - height / 2:.1f}",
                    f"{x + width / 2:.1f},{y:.1f}",
                    f"{x:.1f},{y + height / 2:.1f}",
                    f"{x - width / 2:.1f},{y:.1f}",
                ]
            )
            lines.append(
                f'<polygon class="decision" points="{points}" fill="#fff7df" '
                'stroke="#8f5902" stroke-width="2.2"/>'
            )
        else:
            lines.append(
                f'<rect class="node" x="{x - width / 2:.1f}" '
                f'y="{y - height / 2:.1f}" width="{width:.1f}" '
                f'height="{height:.1f}" rx="12" fill="#f5f8fc" '
                'stroke="#204a87" stroke-width="2.2"/>'
            )
        first_y = y - (len(label_lines) - 1) * 10.0 + 5.0
        lines.append(
            f'<text class="node-label" fill="#17202a" font-family="sans-serif" '
            f'font-size="16" text-anchor="middle" x="{x:.1f}" '
            f'y="{first_y:.1f}">'
        )
        for index, label_line in enumerate(label_lines):
            dy = "0" if index == 0 else "20"
            lines.append(
                f'<tspan x="{x:.1f}" dy="{dy}">{html.escape(label_line)}</tspan>'
            )
        lines.append("</text>")
    lines.extend(["</g>", "</svg>"])
    return "\n".join(lines) + "\n"


def convert_mermaid(
    markdown: str,
    chapter_id: str,
    output_format: str,
    asset_directory: Path | None,
    asset_url_prefix: str | None,
) -> str:
    diagram_index = 0

    def replacement(match: re.Match[str]) -> str:
        nonlocal diagram_index
        diagram_index += 1
        identifier = match.group("id") or f"p{chapter_id}-diagram-{diagram_index}"
        flowchart = parse_flowchart(identifier, match.group("source"))
        if output_format == PDF_FORMAT:
            return tikz_flowchart(flowchart)
        if asset_directory is None or asset_url_prefix is None:
            raise ValueError("EPUB diagram generation requires an asset directory.")
        diagram_directory = asset_directory / "diagrams"
        diagram_directory.mkdir(parents=True, exist_ok=True)
        diagram_path = diagram_directory / f"{identifier}.svg"
        diagram_path.write_text(svg_flowchart(flowchart), encoding="utf-8")
        caption = f"{diagram_caption(identifier)}."
        image_path = f"{asset_url_prefix}/diagrams/{identifier}.svg"
        return f"![{caption}]({image_path}){{#fig:{identifier}}}"

    return MERMAID_PATTERN.sub(replacement, markdown)


def chapter_owned_files(directory: Path, chapter_id: str) -> list[Path]:
    files = sorted(directory.glob(f"P{chapter_id}*.swift"))
    metal_directory = directory / "Metal"
    if metal_directory.is_dir():
        files.extend(sorted(metal_directory.glob(f"P{chapter_id}*.metal")))
    return files


def source_language(path: Path) -> str:
    return "swift" if path.suffix == ".swift" else "cpp"


def relative_path(path: Path) -> str:
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def listing(path: Path) -> str:
    source = path.read_text(encoding="utf-8").rstrip()
    return (
        f"### `{relative_path(path)}`\n\n"
        f"```{source_language(path)}\n{source}\n```\n"
    )


def exercise_section(chapter_id: str, output_format: str) -> str:
    files = chapter_owned_files(EXERCISES_DIRECTORY, chapter_id)
    if not files:
        raise ValueError(f"Chapter {chapter_id} has no learner exercise files.")
    if output_format == EPUB_FORMAT:
        output = [
            "## Exercise source files",
            "",
            epub_callout(
                "exercise",
                (
                    "These are the complete chapter-owned starter files. Work in "
                    "the repository copies so the shared judge, SwiftPM build, and "
                    "Metal host pipeline remain authoritative. Placeholder bodies "
                    "and deliberately incomplete kernels are part of the exercise."
                ),
                "Exercise",
            ),
            "",
        ]
        output.extend(listing(path) for path in files)
        return "\n".join(output)
    output = [
        "## Exercise source files",
        "",
        r"\begin{exercisebox}",
        "",
        "These are the complete chapter-owned starter files. Work in the repository copies so the shared judge, SwiftPM build, and Metal host pipeline remain authoritative. Placeholder bodies and deliberately incomplete kernels are part of the exercise.",
        "",
        r"\end{exercisebox}",
        "",
    ]
    output.extend(listing(path) for path in files)
    return "\n".join(output)


def declaration_map(path: Path) -> list[tuple[int, str]]:
    declarations: list[tuple[int, str]] = []
    pattern = re.compile(
        r"^\s*(?:(?:public|private|internal|fileprivate|open|final)\s+)*"
        r"(?:(?:static|class|mutating|nonmutating)\s+)*"
        r"(?:struct|enum|class|actor|protocol|extension|func|init)\b"
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if pattern.match(line):
            declarations.append((line_number, stripped[:120]))
    if len(declarations) > 18:
        declarations = declarations[:15] + declarations[-3:]
    return declarations


def solution_files(chapter_id: str) -> list[Path]:
    files = chapter_owned_files(SOLUTIONS_DIRECTORY, chapter_id)
    files.extend(SHARED_SOLUTION_FILES.get(chapter_id, []))
    unique: list[Path] = []
    for path in files:
        if path not in unique:
            unique.append(path)
    if not unique:
        raise ValueError(f"Chapter {chapter_id} has no canonical solution files.")
    return unique


def solution_section(
    chapter_id: str,
    notes: dict[str, object],
    output_format: str,
) -> str:
    required_fields = {"strategy", "invariant", "pitfall", "points"}
    missing = required_fields - notes.keys()
    if missing:
        raise ValueError(
            f"Chapter {chapter_id} solution notes are missing: {', '.join(sorted(missing))}"
        )
    points = notes["points"]
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(f"Chapter {chapter_id} needs at least three implementation points.")
    for point in points:
        if not isinstance(point, dict) or "title" not in point or "body" not in point:
            raise ValueError(f"Chapter {chapter_id} has an invalid implementation point.")

    files = solution_files(chapter_id)
    if output_format == EPUB_FORMAT:
        return epub_solution_section(notes, points, files)
    output = [
        "## Worked solution",
        "",
        r"\begin{solutionstrategy}",
        "",
        tex_escape(str(notes["strategy"])),
        "",
        r"\end{solutionstrategy}",
        "",
        r"\begin{invariantbox}",
        "",
        tex_escape(str(notes["invariant"])),
        "",
        r"\end{invariantbox}",
        "",
        r"\begin{pitfallbox}",
        "",
        tex_escape(str(notes["pitfall"])),
        "",
        r"\end{pitfallbox}",
        "",
        "### Implementation points",
        "",
    ]
    for point in points:
        output.extend(
            [
                f"\\begin{{implementationpoint}}[title={{{tex_escape(str(point['title']))}}}]",
                "",
                tex_escape(str(point["body"])),
                "",
                r"\end{implementationpoint}",
                "",
            ]
        )

    output.extend(
        [
            "### Code map",
            "",
            r"\begin{readingguide}",
            "",
            "Use this map to connect declarations to the strategy above before reading each full listing. Line numbers refer to the generated listing's source file.",
            "",
        ]
    )
    for path in files:
        output.append(
            f"\\textbf{{\\texttt{{{tex_escape(relative_path(path))}}}}}"
        )
        output.append("")
        declarations = declaration_map(path)
        if declarations:
            output.append(r"\begin{itemize}[leftmargin=*,nosep]")
            for line_number, declaration in declarations:
                output.append(
                    f"\\item Line {line_number}: "
                    f"\\texttt{{{tex_escape(declaration)}}}"
                )
            output.append(r"\end{itemize}")
        else:
            output.append(
                "The Metal file defines the chapter's complete kernel entry "
                "points and synchronization schedule."
            )
        output.append("")
    output.extend([r"\end{readingguide}", "", "### Complete canonical listings", ""])
    output.extend(listing(path) for path in files)
    return "\n".join(output)


def epub_solution_section(
    notes: dict[str, object],
    points: list[object],
    files: list[Path],
) -> str:
    output = [
        "## Worked solution",
        "",
        epub_callout(
            "solution-strategy",
            str(notes["strategy"]),
            "Solution strategy",
        ),
        "",
        epub_callout("invariant", str(notes["invariant"]), "Invariant"),
        "",
        epub_callout("pitfall", str(notes["pitfall"]), "Pitfall"),
        "",
        "### Implementation points",
        "",
    ]
    for point in points:
        output.extend(
            [
                epub_callout(
                    "implementation-point",
                    str(point["body"]),
                    str(point["title"]),
                ),
                "",
            ]
        )

    reading_guide = [
        "Use this map to connect declarations to the strategy above before reading "
        "each full listing. Line numbers refer to the generated listing's source "
        "file.",
        "",
    ]
    for path in files:
        reading_guide.extend([f"**`{relative_path(path)}`**", ""])
        declarations = declaration_map(path)
        if declarations:
            reading_guide.extend(
                f"- Line {line_number}: `{declaration}`"
                for line_number, declaration in declarations
            )
        else:
            reading_guide.append(
                "The Metal file defines the chapter's complete kernel entry points "
                "and synchronization schedule."
            )
        reading_guide.append("")

    output.extend(
        [
            "### Code map",
            "",
            epub_callout(
                "reading-guide",
                "\n".join(reading_guide),
                "Reading guide",
            ),
            "",
            "### Complete canonical listings",
            "",
        ]
    )
    output.extend(listing(path) for path in files)
    return "\n".join(output)


def preprocess_chapter(
    markdown: str,
    chapter_id: str,
    source_path: Path,
    output_format: str,
    asset_directory: Path | None,
    asset_url_prefix: str | None,
    included_chapter_ids: set[str],
    include_appendices: bool,
) -> tuple[str, str]:
    title, markdown = normalize_chapter_title(markdown, chapter_id)
    markdown = remove_canonical_section(markdown)
    if output_format == EPUB_FORMAT:
        markdown = rewrite_epub_links(
            markdown,
            source_path,
            included_chapter_ids,
            include_appendices,
        )
    markdown = convert_details(markdown, output_format)
    markdown = convert_mermaid(
        markdown,
        chapter_id,
        output_format,
        asset_directory,
        asset_url_prefix,
    )
    return title, markdown.strip()


def preprocess_appendix(
    path: Path,
    identifier: str,
    output_format: str,
    included_chapter_ids: set[str],
) -> str:
    markdown = path.read_text(encoding="utf-8")
    match = re.match(r"^#\s+([^\n]+?)[ \t]*(?:\n|$)", markdown)
    if not match:
        raise ValueError(f"Appendix {path.name} has no level-one title.")
    title = match.group(1)
    body = markdown[match.end() :]
    if output_format == EPUB_FORMAT:
        body = rewrite_epub_links(body, path, included_chapter_ids, True)
    return (
        f"# {title} {{#appendix-{identifier}}}\n"
        f"{body}"
    ).strip()


def part_for_chapter(chapter_number: int) -> str:
    for part, chapter_range in PARTS.items():
        if chapter_number in chapter_range:
            return part
    raise ValueError(f"Chapter {chapter_number:03d} is not assigned to a part.")


def diagram_count(manuscript: str, output_format: str) -> int:
    if output_format == PDF_FORMAT:
        return manuscript.count(r"\begin{figure}")
    return len(re.findall(r"\(.*?/diagrams/[^)]+\.svg\)\{#fig:", manuscript))


def generate(
    chapters: list[tuple[str, Path]],
    notes: dict[str, dict[str, object]],
    output_format: str = PDF_FORMAT,
    asset_directory: Path | None = None,
    asset_url_prefix: str | None = None,
) -> str:
    generated_ids = [chapter_id for chapter_id, _ in chapters]
    is_full_book = (
        len(generated_ids) == len(FULL_BOOK_CHAPTER_IDS)
        and set(generated_ids) == set(FULL_BOOK_CHAPTER_IDS)
    )
    included_chapter_ids = set(generated_ids)
    output = [
        render_frontmatter(
            output_format,
            included_chapter_ids,
            is_full_book,
        )
    ]
    current_part: str | None = None
    for chapter_id, directory in chapters:
        source_path = directory / "README.md"
        if chapter_id == ORIENTATION_CHAPTER_ID:
            _, chapter = preprocess_chapter(
                source_path.read_text(encoding="utf-8"),
                chapter_id,
                source_path,
                output_format,
                asset_directory,
                asset_url_prefix,
                included_chapter_ids,
                is_full_book,
            )
            output.append(chapter)
            continue

        if chapter_id not in notes:
            raise ValueError(f"Chapter {chapter_id} has no worked-solution explanation notes.")
        part = part_for_chapter(int(chapter_id))
        if part != current_part:
            if output_format == PDF_FORMAT:
                output.append(f"\\part{{{tex_escape(part)}}}")
            else:
                output.append(f"# {part} {{.part .unnumbered}}")
            current_part = part

        _, chapter = preprocess_chapter(
            source_path.read_text(encoding="utf-8"),
            chapter_id,
            source_path,
            output_format,
            asset_directory,
            asset_url_prefix,
            included_chapter_ids,
            is_full_book,
        )
        output.extend(
            [
                chapter,
                exercise_section(chapter_id, output_format),
                solution_section(chapter_id, notes[chapter_id], output_format),
            ]
        )
    if is_full_book:
        if output_format == PDF_FORMAT:
            output.extend([r"\appendix", r"\part{Reference}"])
        else:
            output.append("# Reference {.part .unnumbered}")
        output.extend(
            preprocess_appendix(
                path,
                identifier,
                output_format,
                included_chapter_ids,
            )
            for identifier, path in APPENDICES
        )
    separator = "\n\n\\clearpage\n\n" if output_format == PDF_FORMAT else "\n\n"
    manuscript = separator.join(output) + "\n"
    if is_full_book:
        expected_ids = set(COURSE_CHAPTER_IDS)
        if set(notes) != expected_ids:
            missing = sorted(expected_ids - set(notes))
            extra = sorted(set(notes) - expected_ids)
            raise ValueError(
                "Full-book solution-note coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        observed_diagram_count = diagram_count(manuscript, output_format)
        if observed_diagram_count != 49:
            raise ValueError(
                "Full book must contain 49 diagrams; generated "
                f"{observed_diagram_count}."
            )
    return manuscript


def prepare_epub_asset_directory(
    requested_directory: Path,
    output_path: Path,
) -> tuple[Path, str]:
    output_parent = output_path.parent.resolve()
    asset_directory = requested_directory.resolve()
    try:
        relative_asset_directory = asset_directory.relative_to(output_parent)
    except ValueError as error:
        raise SystemExit(
            "EPUB assets must be stored below the manuscript output directory."
        ) from error
    if not relative_asset_directory.parts:
        raise SystemExit(
            "EPUB assets must use a dedicated child of the manuscript output directory."
        )

    resolved_output = output_path.resolve()
    if asset_directory == resolved_output or asset_directory in resolved_output.parents:
        raise SystemExit("The EPUB manuscript cannot be stored inside its asset directory.")

    marker = asset_directory / EPUB_ASSET_MARKER
    if asset_directory.exists():
        recognized = (
            asset_directory.is_dir()
            and marker.is_file()
            and marker.read_text(encoding="utf-8") == EPUB_ASSET_MARKER_CONTENT
        )
        if not recognized:
            raise SystemExit(
                f"Refusing to replace unrecognized EPUB asset directory: "
                f"{asset_directory}"
            )
        shutil.rmtree(asset_directory)

    asset_directory.mkdir(parents=True)
    marker.write_text(EPUB_ASSET_MARKER_CONTENT, encoding="utf-8")
    return asset_directory, relative_asset_directory.as_posix()


def main() -> None:
    arguments = parse_arguments()
    asset_directory: Path | None = None
    asset_url_prefix: str | None = None
    if arguments.format == EPUB_FORMAT:
        if arguments.asset_directory is None:
            raise SystemExit("--asset-directory is required for EPUB output.")
        asset_directory, asset_url_prefix = prepare_epub_asset_directory(
            arguments.asset_directory,
            arguments.output,
        )
    chapters = selected_chapters(arguments.chapters)
    notes = load_solution_notes()
    manuscript = generate(
        chapters,
        notes,
        arguments.format,
        asset_directory,
        asset_url_prefix,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(manuscript, encoding="utf-8")
    observed_diagram_count = diagram_count(manuscript, arguments.format)
    solution_count = manuscript.count("## Worked solution")
    print(
        f"Generated {len(chapters)} chapters, "
        f"{observed_diagram_count} diagrams, and "
        f"{solution_count} worked solutions."
    )


if __name__ == "__main__":
    main()
