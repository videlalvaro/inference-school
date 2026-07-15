import { build } from "esbuild";
import { cp, mkdir, rm } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const projectDirectory = fileURLToPath(new URL(".", import.meta.url));
const sourceDirectory = new URL("src/", import.meta.url);
const outputDirectory = new URL("../../Sources/InferenceSchoolStudio/Resources/Diagram/", import.meta.url);

await rm(outputDirectory, { force: true, recursive: true });
await mkdir(outputDirectory, { recursive: true });
await build({
  entryPoints: [fileURLToPath(new URL("diagram.js", sourceDirectory))],
  outfile: fileURLToPath(new URL("diagram.js", outputDirectory)),
  bundle: true,
  format: "iife",
  minify: true,
  platform: "browser",
  absWorkingDir: projectDirectory
});
await Promise.all([
  cp(new URL("index.html", sourceDirectory), new URL("index.html", outputDirectory)),
  cp(new URL("diagram.css", sourceDirectory), new URL("diagram.css", outputDirectory))
]);