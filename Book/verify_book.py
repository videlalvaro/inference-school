#!/usr/bin/env python3
"""Verify the structure and publication quality of the companion book artifacts."""

from __future__ import annotations

import argparse
import posixpath
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree

EXPECTED_COUNTS = {
    "chapters": (re.compile(r"^\\chapter\{", re.MULTILINE), 50),
    "readable chapters": (re.compile(r"\\label\{chapter-\d{3}\}"), 48),
    "parts": (re.compile(r"^\\part\{", re.MULTILINE), 9),
    "figures": (re.compile(r"^\\begin\{figure\}", re.MULTILINE), 49),
    "exercises": (re.compile(r"^\\begin\{exercisebox\}", re.MULTILINE), 47),
    "worked solutions": (re.compile(r"^\\begin\{solutionstrategy\}", re.MULTILINE), 47),
    "implementation points": (re.compile(r"^\\begin\{implementationpoint\}", re.MULTILINE), 206),
}

FORBIDDEN_LOG_PATTERNS = {
    "fatal LaTeX diagnostics": re.compile(
        r"Undefined control sequence|Missing \$ inserted|Fatal error|Emergency stop|"
        r"Package .* Error|LaTeX Error"
    ),
    "missing glyphs": re.compile(r"Missing character:"),
    "horizontal overflow": re.compile(r"Overfull \\hbox"),
    "vertical overflow": re.compile(r"Overfull \\vbox"),
}

REQUIRED_PDF_TEXT = (
    "Start Here: Build an LLM Inference Engine",
    "Vector Dot Product",
    "Capstone Inference Engine",
    "Anatomy of One Generated Token",
    "Math Primer",
    "Complete canonical listings",
)
MAXIMUM_HORIZONTAL_DIAGRAM_ASPECT_RATIO = 6.0
CANONICAL_GITHUB_BLOB_ROOT = (
    "https://github.com/videlalvaro/inference-school/blob/main"
)
SUPPORTED_EXTERNAL_LINK_SCHEMES = {"http", "https", "mailto"}
REPOSITORY_SOURCE_PATH_PREFIXES = (
    "Sources/InferenceSchool",
    "Tests/InferenceSchool",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", required=True, type=Path)
    parser.add_argument("--tex", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--published", required=True, type=Path)
    parser.add_argument("--epub", type=Path)
    parser.add_argument("--published-epub", type=Path)
    arguments = parser.parse_args()
    if (arguments.epub is None) != (arguments.published_epub is None):
        parser.error("--epub and --published-epub must be supplied together")
    return arguments


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def extract_pdf_text(pdf: Path) -> str:
    result = subprocess.run(
        [
            "gs",
            "-q",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=txtwrite",
            "-sOutputFile=-",
            str(pdf),
        ],
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def diagram_coordinate_aspects(tex: str) -> list[float]:
    aspects: list[float] = []
    for body in re.findall(
        r"\\begin\{tikzpicture\}(.*?)\\end\{tikzpicture\}",
        tex,
        flags=re.DOTALL,
    ):
        coordinates = [
            (float(x), float(y))
            for x, y in re.findall(
                r"at \((-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)\)",
                body,
            )
        ]
        if not coordinates:
            continue
        xs, ys = zip(*coordinates)
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        aspects.append(width / max(height, 0.01))
    return aspects


def element_name(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def archive_target_path(source_path: str, target: str) -> str | None:
    parsed = urlsplit(target)
    if parsed.path:
        candidate = posixpath.normpath(
            posixpath.join(posixpath.dirname(source_path), unquote(parsed.path))
        )
    else:
        candidate = source_path
    if candidate in {".", ".."} or candidate.startswith("../") or candidate.startswith("/"):
        return None
    return candidate


def repository_source_path(target: str) -> str | None:
    parsed = urlsplit(target)
    if parsed.netloc.lower() != "github.com":
        return None
    components = unquote(parsed.path).strip("/").split("/")
    if len(components) < 5 or components[2] != "blob":
        return None
    path = "/".join(components[4:])
    if not path.startswith(REPOSITORY_SOURCE_PATH_PREFIXES):
        return None
    return path


def external_reference_failure(attribute: str, target: str) -> str | None:
    parsed = urlsplit(target)
    if attribute == "src":
        return f"EPUB contains a non-package {attribute}: {target}"
    valid_web_link = (
        parsed.scheme in SUPPORTED_EXTERNAL_LINK_SCHEMES - {"mailto"}
        and bool(parsed.netloc)
    )
    valid_mail_link = parsed.scheme == "mailto" and not parsed.netloc
    if not (valid_web_link or valid_mail_link):
        return f"EPUB contains an unsupported {attribute}: {target}"
    source_path = repository_source_path(target)
    if source_path is not None and not target.startswith(
        f"{CANONICAL_GITHUB_BLOB_ROOT}/"
    ):
        return f"EPUB repository source link is not canonical: {target}"
    return None


def verify_epub(epub_path: Path, failures: list[str]) -> Counter[str]:
    observed: Counter[str] = Counter()
    try:
        archive = zipfile.ZipFile(epub_path)
    except zipfile.BadZipFile as error:
        failures.append(f"EPUB is not a valid ZIP archive: {error}")
        return observed

    with archive:
        entries = archive.infolist()
        names = set(archive.namelist())
        require(bool(entries), "EPUB archive is empty.", failures)
        if not entries:
            return observed
        require(
            entries[0].filename == "mimetype",
            "EPUB mimetype must be the first archive entry.",
            failures,
        )
        require(
            entries[0].compress_type == zipfile.ZIP_STORED,
            "EPUB mimetype entry must not be compressed.",
            failures,
        )
        require("mimetype" in names, "EPUB is missing its mimetype entry.", failures)
        if "mimetype" in names:
            require(
                archive.read("mimetype") == b"application/epub+zip",
                "EPUB has an invalid mimetype value.",
                failures,
            )

        container_path = "META-INF/container.xml"
        require(
            container_path in names,
            "EPUB is missing META-INF/container.xml.",
            failures,
        )
        if container_path not in names:
            return observed
        try:
            container = ElementTree.fromstring(archive.read(container_path))
        except ElementTree.ParseError as error:
            failures.append(f"EPUB container.xml is invalid XML: {error}")
            return observed
        rootfile = next(
            (
                element
                for element in container.iter()
                if element_name(element) == "rootfile"
            ),
            None,
        )
        opf_path = rootfile.get("full-path") if rootfile is not None else None
        require(bool(opf_path), "EPUB container has no OPF rootfile.", failures)
        if not opf_path:
            return observed
        opf_path = posixpath.normpath(opf_path)
        require(opf_path in names, f"EPUB is missing OPF package: {opf_path}", failures)
        if opf_path not in names:
            return observed
        try:
            package = ElementTree.fromstring(archive.read(opf_path))
        except ElementTree.ParseError as error:
            failures.append(f"EPUB OPF package is invalid XML: {error}")
            return observed

        unique_identifier_id = package.get("unique-identifier")
        identifier_elements = [
            element
            for element in package.iter()
            if element_name(element) == "identifier"
        ]
        matching_identifiers = [
            element
            for element in identifier_elements
            if unique_identifier_id and element.get("id") == unique_identifier_id
        ]
        require(
            bool(matching_identifiers)
            and bool((matching_identifiers[0].text or "").strip()),
            "EPUB package has no stable unique identifier.",
            failures,
        )

        manifest: dict[str, tuple[str, str, set[str]]] = {}
        for item in package.iter():
            if element_name(item) != "item":
                continue
            identifier = item.get("id")
            href = item.get("href")
            media_type = item.get("media-type", "")
            if not identifier or not href:
                failures.append("EPUB manifest contains an item without id or href.")
                continue
            parsed_href = urlsplit(href)
            if parsed_href.scheme or parsed_href.netloc:
                failures.append(f"EPUB manifest contains a remote resource: {href}")
                continue
            item_path = archive_target_path(opf_path, href)
            if item_path is None:
                failures.append(f"EPUB manifest resource escapes the package: {href}")
                continue
            require(
                item_path in names,
                f"EPUB manifest resource is missing: {item_path}",
                failures,
            )
            manifest[identifier] = (
                item_path,
                media_type,
                set(item.get("properties", "").split()),
            )

        navigation_items = [
            path
            for path, _, properties in manifest.values()
            if "nav" in properties
        ]
        require(
            len(navigation_items) == 1,
            f"EPUB must contain one navigation document; found {len(navigation_items)}.",
            failures,
        )

        xhtml_paths = [
            path
            for path, media_type, _ in manifest.values()
            if media_type == "application/xhtml+xml" and path in names
        ]
        svg_paths = [
            path
            for path, media_type, _ in manifest.values()
            if media_type == "image/svg+xml" and path in names
        ]
        require(
            len(svg_paths) == 49,
            f"Expected 49 packaged SVG diagrams; found {len(svg_paths)}.",
            failures,
        )

        document_roots: dict[str, ElementTree.Element] = {}
        document_source: dict[str, str] = {}
        document_ids: dict[str, set[str]] = {}
        for path in xhtml_paths:
            source = archive.read(path).decode("utf-8", errors="replace")
            document_source[path] = source
            try:
                root = ElementTree.fromstring(source)
            except ElementTree.ParseError as error:
                failures.append(f"EPUB XHTML is invalid XML ({path}): {error}")
                continue
            document_roots[path] = root
            document_ids[path] = {
                identifier
                for element in root.iter()
                if (identifier := element.get("id"))
            }

        all_ids = set().union(*document_ids.values()) if document_ids else set()
        class_counts: Counter[str] = Counter()
        referenced_svgs = 0
        book_text_parts: list[str] = []
        for path, root in document_roots.items():
            book_text_parts.extend(root.itertext())
            for element in root.iter():
                class_counts.update(element.get("class", "").split())
                if element_name(element) == "img":
                    source = element.get("src", "")
                    if urlsplit(source).path.endswith(".svg"):
                        referenced_svgs += 1

                for attribute in ("href", "src"):
                    target = element.get(attribute)
                    if not target:
                        continue
                    parsed_target = urlsplit(target)
                    if parsed_target.scheme or parsed_target.netloc:
                        if failure := external_reference_failure(attribute, target):
                            failures.append(failure)
                        continue
                    target_path = archive_target_path(path, target)
                    if target_path is None:
                        failures.append(
                            f"EPUB {attribute} escapes the package in {path}: {target}"
                        )
                        continue
                    require(
                        target_path in names,
                        f"EPUB {attribute} target is missing in {path}: {target}",
                        failures,
                    )
                    fragment = unquote(parsed_target.fragment)
                    if fragment and target_path in document_ids:
                        require(
                            fragment in document_ids[target_path],
                            f"EPUB fragment target is missing in {path}: {target}",
                            failures,
                        )

        observed["readable chapters"] = len(
            [identifier for identifier in all_ids if re.fullmatch(r"chapter-\d{3}", identifier)]
        )
        observed["exercises"] = class_counts["exercise"]
        observed["worked solutions"] = class_counts["solution-strategy"]
        observed["appendices"] = len(
            [identifier for identifier in all_ids if identifier.startswith("appendix-")]
        )
        observed["diagrams"] = referenced_svgs
        expected_epub_counts = {
            "readable chapters": 48,
            "exercises": 47,
            "worked solutions": 47,
            "appendices": 2,
            "diagrams": 49,
        }
        for label, expected in expected_epub_counts.items():
            require(
                observed[label] == expected,
                f"Expected {expected} EPUB {label}; found {observed[label]}.",
                failures,
            )

        book_text = re.sub(r"\s+", " ", " ".join(book_text_parts))
        for required_text in REQUIRED_PDF_TEXT:
            require(
                required_text in book_text,
                f"EPUB text is missing: {required_text}",
                failures,
            )

        pdf_only_pattern = re.compile(
            r"\\(?:begin|end)\{(?:hintbox|exercisebox|solutionstrategy|"
            r"invariantbox|pitfallbox|implementationpoint|readingguide|"
            r"chapterfocus|figure|tikzpicture|itemize|adjustbox)\}|"
            r"\\part\{|\\appendix|\\clearpage|\\textbf\{|\\texttt\{"
        )
        leaked_documents = [
            path
            for path, source in document_source.items()
            if pdf_only_pattern.search(source)
        ]
        require(
            not leaked_documents,
            "EPUB XHTML contains PDF-only TeX in: " + ", ".join(leaked_documents),
            failures,
        )
    return observed


def run_epubcheck(epub_path: Path, failures: list[str]) -> None:
    try:
        result = subprocess.run(
            ["epubcheck", str(epub_path)],
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        failures.append(
            "epubcheck is required by make check but was not found on PATH."
        )
        return
    if result.returncode != 0:
        output = "\n".join(
            line for line in (result.stdout, result.stderr) if line.strip()
        ).strip()
        failures.append(
            "epubcheck rejected the EPUB"
            + (f":\n{output}" if output else ".")
        )


def main() -> None:
    arguments = parse_arguments()
    failures: list[str] = []

    required_paths = [
        arguments.manuscript,
        arguments.tex,
        arguments.log,
        arguments.pdf,
        arguments.published,
    ]
    if arguments.epub is not None and arguments.published_epub is not None:
        required_paths.extend([arguments.epub, arguments.published_epub])
    for path in required_paths:
        require(path.is_file(), f"Missing artifact: {path}", failures)
    if failures:
        raise SystemExit("\n".join(failures))

    manuscript = arguments.manuscript.read_text(encoding="utf-8")
    tex = arguments.tex.read_text(encoding="utf-8")
    log = arguments.log.read_text(encoding="utf-8", errors="replace")

    require(
        manuscript.count(r"\begin{figure}") == 49,
        "Manuscript does not contain exactly 49 generated diagrams.",
        failures,
    )
    require(
        manuscript.count("## Worked solution") == 47,
        "Manuscript does not contain exactly 47 worked solutions.",
        failures,
    )
    require(
        manuscript.count(r"\appendix") == 1,
        "Manuscript must contain one appendix marker.",
        failures,
    )

    observed_counts: dict[str, int] = {}
    for label, (pattern, expected) in EXPECTED_COUNTS.items():
        observed = len(pattern.findall(tex))
        observed_counts[label] = observed
        require(
            observed == expected,
            f"Expected {expected} {label}; generated {observed}.",
            failures,
        )

    diagram_aspects = diagram_coordinate_aspects(tex)
    require(
        len(diagram_aspects) == EXPECTED_COUNTS["figures"][1],
        "Could not measure every generated diagram.",
        failures,
    )
    require(
        all(aspect <= MAXIMUM_HORIZONTAL_DIAGRAM_ASPECT_RATIO for aspect in diagram_aspects),
        "Generated diagrams contain an unreadably wide layout.",
        failures,
    )

    page_match = re.search(
        r"Output written on .*?\((\d+) pages,\s*(\d+) bytes\)\.",
        log,
    )
    require(page_match is not None, "LuaLaTeX log has no completed PDF record.", failures)
    pages = int(page_match.group(1)) if page_match else 0
    require(pages >= 500, f"Expected a complete book of at least 500 pages; got {pages}.", failures)

    for label, pattern in FORBIDDEN_LOG_PATTERNS.items():
        matches = pattern.findall(log)
        require(not matches, f"LaTeX log contains {len(matches)} {label}.", failures)

    require(
        arguments.pdf.read_bytes() == arguments.published.read_bytes(),
        "Published PDF differs from the verified build PDF.",
        failures,
    )

    try:
        pdf_text = extract_pdf_text(arguments.pdf)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        failures.append(f"Ghostscript text extraction failed: {error}")
        pdf_text = ""
    for required_text in REQUIRED_PDF_TEXT:
        require(
            required_text in pdf_text,
            f"PDF text layer is missing: {required_text}",
            failures,
        )

    epub_counts: Counter[str] = Counter()
    if arguments.epub is not None and arguments.published_epub is not None:
        require(
            arguments.epub.read_bytes() == arguments.published_epub.read_bytes(),
            "Published EPUB differs from the verified build EPUB.",
            failures,
        )
        epub_counts = verify_epub(arguments.epub, failures)
        run_epubcheck(arguments.epub, failures)

    if failures:
        print("Book verification failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    summary = (
        "Verified "
        f"{pages} pages, {observed_counts['readable chapters']} readable chapters, "
        f"{observed_counts['figures']} diagrams, "
        f"{observed_counts['worked solutions']} worked solutions, and "
        f"{observed_counts['implementation points']} implementation points"
    )
    if arguments.epub is not None:
        summary += (
            "; EPUB contains "
            f"{epub_counts['readable chapters']} readable chapters, "
            f"{epub_counts['diagrams']} diagrams, and "
            f"{epub_counts['worked solutions']} worked solutions"
        )
    print(summary + ".")


if __name__ == "__main__":
    main()
