#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import generate_book
import verify_book


class EpubLinkTests(unittest.TestCase):
    source_path = (
        generate_book.REPOSITORY_ROOT
        / "Problems"
        / "001-vector-dot"
        / "README.md"
    )

    def rewrite(self, markdown: str) -> str:
        return generate_book.rewrite_epub_links(
            markdown,
            self.source_path,
            {"001"},
            False,
        )

    def test_repository_source_link_uses_canonical_origin(self) -> None:
        rewritten = self.rewrite(
            "[source](../../Sources/InferenceSchoolExercises/"
            "P001VectorDotExercise.swift)"
        )
        self.assertEqual(
            rewritten,
            "[source](https://github.com/videlalvaro/inference-school/blob/main/"
            "Sources/InferenceSchoolExercises/P001VectorDotExercise.swift)",
        )

    def test_external_images_are_rejected(self) -> None:
        for target in ("file:///etc/passwd", "https://example.com/image.png"):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "must be repository-local"):
                    self.rewrite(f"![image]({target})")

    def test_unsupported_link_scheme_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported scheme"):
            self.rewrite("[payload](data:text/plain,hello)")

    def test_web_link_is_preserved(self) -> None:
        link = "[Pandoc](https://pandoc.org/)"
        self.assertEqual(self.rewrite(link), link)

    def test_repository_local_image_stays_inside_repository(self) -> None:
        rewritten = generate_book.rewrite_epub_links(
            "![preview](docs/assets/social-preview.png)",
            generate_book.REPOSITORY_ROOT / "README.md",
            set(),
            False,
        )
        self.assertEqual(
            rewritten,
            "![preview](../docs/assets/social-preview.png)",
        )


class EpubAssetDirectoryTests(unittest.TestCase):
    def test_unrecognized_directory_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assets = root / "assets"
            assets.mkdir()
            sentinel = assets / "sentinel.txt"
            sentinel.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "Refusing to replace"):
                generate_book.prepare_epub_asset_directory(
                    assets,
                    root / "manuscript.md",
                )

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_owned_directory_can_be_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            assets = root / "assets"
            output = root / "manuscript.md"
            prepared, prefix = generate_book.prepare_epub_asset_directory(
                assets,
                output,
            )
            generated_file = prepared / "stale.svg"
            generated_file.write_text("stale", encoding="utf-8")

            regenerated, regenerated_prefix = (
                generate_book.prepare_epub_asset_directory(assets, output)
            )

            self.assertEqual(prefix, "assets")
            self.assertEqual(regenerated_prefix, prefix)
            self.assertFalse(generated_file.exists())
            self.assertEqual(
                (regenerated / generate_book.EPUB_ASSET_MARKER).read_text(
                    encoding="utf-8"
                ),
                generate_book.EPUB_ASSET_MARKER_CONTENT,
            )

    def test_output_directory_itself_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(SystemExit, "dedicated child"):
                generate_book.prepare_epub_asset_directory(
                    root,
                    root / "manuscript.md",
                )


class EpubVerifierLinkTests(unittest.TestCase):
    def test_archive_target_normalizing_to_package_root_is_rejected(self) -> None:
        self.assertIsNone(
            verify_book.archive_target_path("OEBPS/content.opf", "a/../..")
        )

    def test_noncanonical_repository_source_link_is_rejected(self) -> None:
        target = (
            "https://github.com/contributor/inference-school/blob/main/"
            "Sources/InferenceSchoolExercises/P001VectorDotExercise.swift"
        )
        self.assertIn(
            "not canonical",
            verify_book.external_reference_failure("href", target) or "",
        )

    def test_canonical_repository_source_link_is_allowed(self) -> None:
        target = (
            f"{verify_book.CANONICAL_GITHUB_BLOB_ROOT}/"
            "Sources/InferenceSchoolExercises/P001VectorDotExercise.swift"
        )
        self.assertIsNone(verify_book.external_reference_failure("href", target))

    def test_external_image_source_is_rejected(self) -> None:
        self.assertIn(
            "non-package src",
            verify_book.external_reference_failure(
                "src",
                "https://example.com/image.png",
            )
            or "",
        )


if __name__ == "__main__":
    unittest.main()