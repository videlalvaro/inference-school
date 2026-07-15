import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputPath = path.join(repositoryRoot, "THIRD_PARTY_NOTICES.md");
const licenseFilePattern = /^(license|copying|notice)([-.]|$)/i;
const checkMode = process.argv.slice(2).includes("--check");

function normalizedText(text) {
  return text
    .replaceAll("\r\n", "\n")
    .split("\n")
    .map((line) => line.trimEnd())
    .join("\n")
    .trim();
}

async function licenseFiles(directory) {
  const names = (await readdir(directory))
    .filter((name) => licenseFilePattern.test(name))
    .sort((left, right) => left.localeCompare(right));
  return Promise.all(names.map(async (name) => ({
    name,
    text: normalizedText(await readFile(path.join(directory, name), "utf8"))
  })));
}

async function npmComponents(projectPath) {
  const projectDirectory = path.join(repositoryRoot, projectPath);
  const lock = JSON.parse(await readFile(path.join(projectDirectory, "package-lock.json"), "utf8"));
  const components = [];

  for (const [packagePath, lockPackage] of Object.entries(lock.packages)) {
    if (!packagePath.startsWith("node_modules/") || lockPackage.dev) continue;
    const packageDirectory = path.join(projectDirectory, packagePath);
    const packageMetadata = JSON.parse(
      await readFile(path.join(packageDirectory, "package.json"), "utf8")
    );
    const notices = await licenseFiles(packageDirectory);
    if (notices.length === 0) {
      throw new Error(`${projectPath}/${packagePath} has no license or notice file`);
    }
    components.push({
      ecosystem: "npm",
      name: packageMetadata.name,
      notices,
      declaredLicense: packageMetadata.license ?? lockPackage.license ?? "not declared",
      version: packageMetadata.version
    });
  }

  return components;
}

async function swiftComponents() {
  const resolved = JSON.parse(
    await readFile(path.join(repositoryRoot, "Package.resolved"), "utf8")
  );
  const checkoutsDirectory = path.join(repositoryRoot, ".build", "checkouts");
  const checkoutNames = await readdir(checkoutsDirectory);

  return Promise.all(resolved.pins.map(async (pin) => {
    const checkoutName = checkoutNames.find(
      (name) => name.toLowerCase() === pin.identity.toLowerCase()
    );
    if (!checkoutName) {
      throw new Error(`No SwiftPM checkout found for ${pin.identity}`);
    }
    const notices = await licenseFiles(path.join(checkoutsDirectory, checkoutName));
    if (notices.length === 0) {
      throw new Error(`${pin.identity} has no license or notice file`);
    }
    return {
      ecosystem: "SwiftPM",
      name: pin.identity,
      notices,
      declaredLicense: "see included license text",
      version: pin.state.version ?? pin.state.revision
    };
  }));
}

function componentID(component) {
  return `${component.ecosystem}: ${component.name} ${component.version}`;
}

function render(components) {
  const uniqueComponents = new Map();
  for (const component of components) {
    uniqueComponents.set(componentID(component), component);
  }
  const sortedComponents = [...uniqueComponents.values()].sort((left, right) =>
    componentID(left).localeCompare(componentID(right))
  );

  const noticeGroups = new Map();
  for (const component of sortedComponents) {
    for (const notice of component.notices) {
      const digest = createHash("sha256").update(notice.text).digest("hex");
      const group = noticeGroups.get(digest) ?? {
        components: new Set(),
        fileNames: new Set(),
        text: notice.text
      };
      group.components.add(componentID(component));
      group.fileNames.add(notice.name);
      noticeGroups.set(digest, group);
    }
  }

  const lines = [
    "# Third-Party Notices",
    "",
    "Inference School Studio incorporates the following third-party packages. This file is",
    "generated from the committed lockfiles and installed package license files by",
    "`scripts/generate-third-party-notices.mjs`.",
    "",
    "## Components",
    ""
  ];
  for (const component of sortedComponents) {
    lines.push(
      `- ${componentID(component)} (${component.declaredLicense})`
    );
  }

  lines.push("", "## License And Notice Texts", "");
  const sortedGroups = [...noticeGroups.entries()].sort(([left], [right]) =>
    left.localeCompare(right)
  );
  for (const [digest, group] of sortedGroups) {
    lines.push(
      `### ${digest.slice(0, 12)}`,
      "",
      `Applies to: ${[...group.components].sort().join(", ")}`,
      "",
      `Source files: ${[...group.fileNames].sort().join(", ")}`,
      "",
      ...group.text.split("\n").map((line) => line ? `    ${line}` : ""),
      ""
    );
  }

  return `${lines.join("\n").trim()}\n`;
}

const components = [
  ...(await npmComponents("Web/Editor")),
  ...(await npmComponents("Web/Diagram")),
  ...(await swiftComponents())
];
const renderedNotices = render(components);
if (checkMode) {
  const committedNotices = await readFile(outputPath, "utf8");
  if (committedNotices !== renderedNotices) {
    throw new Error(
      `${path.relative(repositoryRoot, outputPath)} is stale; regenerate it without --check`
    );
  }
  console.log(
    `Verified ${path.relative(repositoryRoot, outputPath)} for ${components.length} package entries.`
  );
} else {
  await writeFile(outputPath, renderedNotices, "utf8");
  console.log(
    `Wrote ${path.relative(repositoryRoot, outputPath)} for ${components.length} package entries.`
  );
}