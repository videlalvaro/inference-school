#!/usr/bin/env python3
"""Generate the Inference School companion manuscript from curriculum source files."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

BOOK_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = BOOK_DIRECTORY.parent
PROBLEMS_DIRECTORY = REPOSITORY_ROOT / "Problems"
EXERCISES_DIRECTORY = REPOSITORY_ROOT / "Sources" / "InferenceSchoolExercises"
SOLUTIONS_DIRECTORY = REPOSITORY_ROOT / "Sources" / "InferenceSchoolSolutions"

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


def convert_details(markdown: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        title = html.unescape(match.group(1).strip())
        body = match.group(2).strip()
        if "canonical" in title.lower():
            return ""
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


def tikz_flowchart(flowchart: Flowchart) -> str:
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


def convert_mermaid(markdown: str, chapter_id: str) -> str:
    diagram_index = 0

    def replacement(match: re.Match[str]) -> str:
        nonlocal diagram_index
        diagram_index += 1
        identifier = match.group("id") or f"p{chapter_id}-diagram-{diagram_index}"
        return tikz_flowchart(parse_flowchart(identifier, match.group("source")))

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


def exercise_section(chapter_id: str) -> str:
    files = chapter_owned_files(EXERCISES_DIRECTORY, chapter_id)
    if not files:
        raise ValueError(f"Chapter {chapter_id} has no learner exercise files.")
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


def solution_section(chapter_id: str, notes: dict[str, object]) -> str:
    required_fields = {"strategy", "invariant", "pitfall", "points"}
    missing = required_fields - notes.keys()
    if missing:
        raise ValueError(
            f"Chapter {chapter_id} solution notes are missing: {', '.join(sorted(missing))}"
        )
    points = notes["points"]
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError(f"Chapter {chapter_id} needs at least three implementation points.")

    files = solution_files(chapter_id)
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
        if not isinstance(point, dict) or "title" not in point or "body" not in point:
            raise ValueError(f"Chapter {chapter_id} has an invalid implementation point.")
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


def preprocess_chapter(markdown: str, chapter_id: str) -> tuple[str, str]:
    title, markdown = normalize_chapter_title(markdown, chapter_id)
    markdown = remove_canonical_section(markdown)
    markdown = convert_details(markdown)
    markdown = convert_mermaid(markdown, chapter_id)
    return title, markdown.strip()


def preprocess_appendix(path: Path, identifier: str) -> str:
    markdown = path.read_text(encoding="utf-8")
    match = re.match(r"^#\s+([^\n]+?)[ \t]*(?:\n|$)", markdown)
    if not match:
        raise ValueError(f"Appendix {path.name} has no level-one title.")
    title = match.group(1)
    return (
        f"# {title} {{#appendix-{identifier}}}\n"
        f"{markdown[match.end():]}"
    ).strip()


def part_for_chapter(chapter_number: int) -> str:
    for part, chapter_range in PARTS.items():
        if chapter_number in chapter_range:
            return part
    raise ValueError(f"Chapter {chapter_number:03d} is not assigned to a part.")


def generate(chapters: list[tuple[str, Path]], notes: dict[str, dict[str, object]]) -> str:
    output = [(BOOK_DIRECTORY / "frontmatter.md").read_text(encoding="utf-8").strip()]
    current_part: str | None = None
    for chapter_id, directory in chapters:
        if chapter_id == ORIENTATION_CHAPTER_ID:
            _, chapter = preprocess_chapter(
                (directory / "README.md").read_text(encoding="utf-8"),
                chapter_id,
            )
            output.append(chapter)
            continue

        if chapter_id not in notes:
            raise ValueError(f"Chapter {chapter_id} has no worked-solution explanation notes.")
        part = part_for_chapter(int(chapter_id))
        if part != current_part:
            output.append(f"\\part{{{tex_escape(part)}}}")
            current_part = part

        _, chapter = preprocess_chapter(
            (directory / "README.md").read_text(encoding="utf-8"),
            chapter_id,
        )
        output.extend(
            [
                chapter,
                exercise_section(chapter_id),
                solution_section(chapter_id, notes[chapter_id]),
            ]
        )
    generated_ids = [chapter_id for chapter_id, _ in chapters]
    is_full_book = (
        len(generated_ids) == len(FULL_BOOK_CHAPTER_IDS)
        and set(generated_ids) == set(FULL_BOOK_CHAPTER_IDS)
    )
    if is_full_book:
        output.extend([r"\appendix", r"\part{Reference}"])
        output.extend(
            preprocess_appendix(path, identifier)
            for identifier, path in APPENDICES
        )
    manuscript = "\n\n\\clearpage\n\n".join(output) + "\n"
    if is_full_book:
        expected_ids = set(COURSE_CHAPTER_IDS)
        if set(notes) != expected_ids:
            missing = sorted(expected_ids - set(notes))
            extra = sorted(set(notes) - expected_ids)
            raise ValueError(
                "Full-book solution-note coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        diagram_count = manuscript.count(r"\begin{figure}")
        if diagram_count != 49:
            raise ValueError(
                f"Full book must contain 49 diagrams; generated {diagram_count}."
            )
    return manuscript


def main() -> None:
    arguments = parse_arguments()
    chapters = selected_chapters(arguments.chapters)
    notes = load_solution_notes()
    manuscript = generate(chapters, notes)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(manuscript, encoding="utf-8")
    diagram_count = manuscript.count(r"\begin{figure}")
    solution_count = manuscript.count("## Worked solution")
    print(
        f"Generated {len(chapters)} chapters, "
        f"{diagram_count} diagrams, and "
        f"{solution_count} worked solutions."
    )


if __name__ == "__main__":
    main()
