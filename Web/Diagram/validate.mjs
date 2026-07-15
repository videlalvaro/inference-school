import { createServer } from "node:http";
import { access, readdir, readFile, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { JSDOM } from "jsdom";
import { chromium } from "playwright-core";
import { PNG } from "pngjs";

const dom = new JSDOM("<!doctype html><html><body></body></html>");
globalThis.window = dom.window;
globalThis.document = dom.window.document;
Object.defineProperty(globalThis, "navigator", {
  configurable: true,
  value: dom.window.navigator
});

const { default: mermaid } = await import("mermaid");

const problemsDirectory = new URL("../../Problems/", import.meta.url);
const repositoryDirectory = new URL("../../", import.meta.url);
const markdownFiles = await collectMarkdownFiles(problemsDirectory);
const diagrams = [];

for (const fileURL of markdownFiles) {
  const markdown = await readFile(fileURL, "utf8");
  const matches = markdown.matchAll(/```mermaid(?:\s+\{#([^}\s]+)[^}]*\})?[^\n]*\n([\s\S]*?)\n```/g);
  for (const [index, match] of Array.from(matches).entries()) {
    const relativePath = fileURL.pathname.slice(problemsDirectory.pathname.length);
    const id = match[1] || `${relativePath.replace(/[^a-zA-Z0-9_-]/g, "-")}-${index + 1}`;
    try {
      await mermaid.parse(match[2]);
    } catch (error) {
      throw new Error(
        `${fileURL.pathname}: invalid Mermaid diagram\n${error instanceof Error ? error.message : String(error)}`
      );
    }
    diagrams.push({ id, source: match[2], lesson: relativePath });
  }
}

if (diagrams.length === 0) {
  throw new Error("No Mermaid diagrams were found in the curriculum.");
}

const problem003Diagram = diagrams.find(({ id }) => id === "p003-tiled-transpose-copy");
if (!problem003Diagram) {
  throw new Error("Missing regression diagram p003-tiled-transpose-copy.");
}

const browserExecutable = await locateBrowserExecutable();
const server = await startStaticServer(repositoryDirectory);
const browser = await chromium.launch({ executablePath: browserExecutable, headless: true });

try {
  const page = await browser.newPage({ viewport: { width: 860, height: 900 } });
  const rendererURL = new URL(
    "Sources/InferenceSchoolStudio/Resources/Diagram/index.html",
    server.origin
  );
  await page.goto(rendererURL.href);
  await page.waitForFunction(() => typeof window.InferenceSchoolDiagram?.render === "function");

  for (const diagram of diagrams) {
    for (const theme of ["light", "dark"]) {
      const rendering = { ...diagram, theme };
      const result = await page.evaluate(async (payload) => {
        await window.InferenceSchoolDiagram.render({
          id: payload.id,
          source: payload.source,
          title: `${payload.lesson} diagram`,
          theme: payload.theme,
          zoom: 1
        });
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

        const root = document.querySelector("#diagram");
        const canvas = root?.querySelector(".diagram-canvas");
        const image = canvas?.querySelector("svg");
        const bounds = image?.getBoundingClientRect();
        const viewBox = image?.viewBox?.baseVal;
        return {
          hasError: Boolean(root?.querySelector(".diagram-error")),
          error: root?.querySelector(".diagram-error")?.textContent ?? "",
          svgCount: root?.querySelectorAll("svg").length ?? 0,
          graphicsCount: image?.querySelectorAll("path, rect, circle, ellipse, line, polyline, polygon, text, foreignObject").length ?? 0,
          text: image?.textContent?.trim() ?? "",
          bounds: bounds ? { width: bounds.width, height: bounds.height } : null,
          viewBox: viewBox ? { width: viewBox.width, height: viewBox.height } : null
        };
      }, rendering);

      validateRenderedDiagram(rendering, result);
      const screenshot = await page.locator("#diagram svg").screenshot({
        animations: "disabled",
        type: "png"
      });
      validateVisiblePixels(rendering, PNG.sync.read(screenshot));
    }
  }
} finally {
  await browser.close();
  await server.close();
}

console.log(
  `Rendered and validated ${diagrams.length} Mermaid diagrams in ${markdownFiles.length} lessons in light and dark themes.`
);

function validateRenderedDiagram(diagram, result) {
  const failure = (reason) => {
    throw new Error(`${diagram.lesson}#${diagram.id} [${diagram.theme}]: ${reason}`);
  };
  if (result.hasError) failure(result.error || "renderer displayed an error");
  if (result.svgCount !== 1) failure(`expected one SVG, found ${result.svgCount}`);
  if (!result.viewBox || result.viewBox.width <= 0 || result.viewBox.height <= 0) {
    failure(`invalid SVG viewBox ${JSON.stringify(result.viewBox)}`);
  }
  if (!result.bounds || result.bounds.width < 2 || result.bounds.height < 2) {
    failure(`SVG is not visible: ${JSON.stringify(result.bounds)}`);
  }
  if (result.graphicsCount === 0 || result.text.length === 0) {
    failure("SVG has no visible diagram content");
  }
  const aspectRatio = result.viewBox.width / result.viewBox.height;
  if (!Number.isFinite(aspectRatio) || aspectRatio > 6 || aspectRatio < 1 / 6) {
    failure(`unreadable SVG aspect ratio ${aspectRatio.toFixed(2)}`);
  }
}

function validateVisiblePixels(diagram, image) {
  const failure = (reason) => {
    throw new Error(`${diagram.lesson}#${diagram.id} [${diagram.theme}]: ${reason}`);
  };
  if (image.width < 2 || image.height < 2) {
    failure(`invalid screenshot dimensions ${image.width}x${image.height}`);
  }

  const background = image.data.subarray(0, 4);
  let visiblePixels = 0;
  for (let offset = 0; offset < image.data.length; offset += 4) {
    const alphaDifference = Math.abs(image.data[offset + 3] - background[3]);
    const colorDifference = Math.max(
      Math.abs(image.data[offset] - background[0]),
      Math.abs(image.data[offset + 1] - background[1]),
      Math.abs(image.data[offset + 2] - background[2])
    );
    if (alphaDifference > 12 || colorDifference > 12) visiblePixels += 1;
  }

  const minimumVisiblePixels = Math.max(64, image.width * image.height * 0.001);
  if (visiblePixels < minimumVisiblePixels) {
    failure(
      `screenshot is visually blank: ${visiblePixels} non-background pixels in ${image.width}x${image.height}`
    );
  }
}

async function locateBrowserExecutable() {
  const configured = process.env.INFERENCE_SCHOOL_BROWSER_PATH;
  const candidates = [
    configured,
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/usr/bin/microsoft-edge",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser"
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      await access(candidate);
      return candidate;
    } catch {}
  }
  throw new Error(
    "A Chromium browser is required for diagram rendering tests. Set INFERENCE_SCHOOL_BROWSER_PATH to its executable."
  );
}

async function startStaticServer(rootURL) {
  const rootPath = resolve(fileURLToPath(rootURL));
  const contentTypes = new Map([
    [".css", "text/css; charset=utf-8"],
    [".html", "text/html; charset=utf-8"],
    [".js", "text/javascript; charset=utf-8"]
  ]);
  const server = createServer(async (request, response) => {
    try {
      const pathname = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
      const filePath = resolve(rootPath, `.${pathname}`);
      if (filePath !== rootPath && !filePath.startsWith(`${rootPath}${sep}`)) {
        response.writeHead(403).end();
        return;
      }
      const metadata = await stat(filePath);
      if (!metadata.isFile()) throw new Error("Not a file");
      response.writeHead(200, {
        "Content-Type": contentTypes.get(extname(filePath)) || "application/octet-stream"
      });
      response.end(await readFile(filePath));
    } catch {
      response.writeHead(404).end();
    }
  });
  await new Promise((resolveReady, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolveReady);
  });
  const address = server.address();
  return {
    origin: `http://127.0.0.1:${address.port}/`,
    close: () => new Promise((resolveClose, reject) => {
      server.close(error => error ? reject(error) : resolveClose());
    })
  };
}

async function collectMarkdownFiles(directoryURL) {
  const files = [];
  for (const entry of await readdir(directoryURL, { withFileTypes: true })) {
    const entryURL = new URL(entry.name + (entry.isDirectory() ? "/" : ""), directoryURL);
    if (entry.isDirectory()) {
      files.push(...await collectMarkdownFiles(entryURL));
    } else if (entry.name.endsWith(".md")) {
      files.push(entryURL);
    }
  }
  return files;
}