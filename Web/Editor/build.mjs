import { build } from "esbuild";
import { cp, mkdir, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const projectDirectory = fileURLToPath(new URL(".", import.meta.url));
const sourceDirectory = new URL("src/", import.meta.url);
const outputDirectory = new URL("../../Sources/InferenceSchoolStudio/Resources/Editor/", import.meta.url);

await rm(outputDirectory, { force: true, recursive: true });
await mkdir(outputDirectory, { recursive: true });
await build({
  entryPoints: [fileURLToPath(new URL("editor.js", sourceDirectory))],
  outfile: fileURLToPath(new URL("editor.js", outputDirectory)),
  bundle: true,
  format: "iife",
  minify: true,
  platform: "browser",
  absWorkingDir: projectDirectory
});
await Promise.all([
  cp(new URL("index.html", sourceDirectory), new URL("index.html", outputDirectory)),
  cp(new URL("editor.css", sourceDirectory), new URL("editor.css", outputDirectory))
]);