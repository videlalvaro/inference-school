# Contributing

## Development setup

Inference School requires an Apple Silicon Mac running macOS 15 or newer, plus Xcode or
the Xcode command-line tools with Swift and Metal.

Install the locked web dependencies before rebuilding embedded browser assets:

```sh
npm ci --prefix Web/Editor
npm ci --prefix Web/Diagram
```

The diagram test renders every lesson image in a Chromium browser and checks
the resulting pixels. It detects Microsoft Edge, Google Chrome, or Chromium in
their standard locations. Set `INFERENCE_SCHOOL_BROWSER_PATH` to the browser executable
when it is installed elsewhere.

## Before opening a pull request

Run the checks relevant to your change. For a complete validation pass, run:

```sh
swift test
npm run build --prefix Web/Editor
npm run build --prefix Web/Diagram
npm test --prefix Web/Diagram
node scripts/generate-third-party-notices.mjs
node scripts/generate-third-party-notices.mjs --check
make -C Book check
scripts/package-studio.sh debug
codesign --verify --deep --strict --verbose=2 "dist/Inference School Studio.app"
```

Rebuilding either web bundle changes committed files under
`Sources/InferenceSchoolStudio/Resources`. Commit those generated resources together
with their source changes. Rebuilding dependencies may also change
`THIRD_PARTY_NOTICES.md`; regenerate and commit it with any lockfile update.
Rebuilding the book changes `dist/Inference-School-Companion.pdf` and
`dist/Inference-School-Companion.epub`; commit both published artifacts with curriculum
or book-source changes that affect them.

Keep changes focused, add tests for behavior changes, and update the lessons or
book when a concept or public command changes. Do not commit generated files
from `.build/`, `Book/build/`, other paths under `dist/`, or either
`node_modules/` directory.

## Security reports

Do not open a public issue for a suspected vulnerability. Follow
[SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contribution is licensed under the Apache
License 2.0, as described in [LICENSE](LICENSE).
