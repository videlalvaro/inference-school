import mermaid from "mermaid";

const root = document.querySelector("#diagram");
let renderSequence = 0;
let currentZoom = 1;
let canvas = null;
let canvasAspectRatio = null;
let canvasObserver = null;
let dragState = null;
const maximumReadableAspectRatio = 6;
const maximumFitHeight = 680;

function postMessage(message) {
  window.webkit?.messageHandlers?.diagram?.postMessage(message);
}

function reportSize() {
  const canvasHeight = canvas?.getBoundingClientRect().height ?? root.getBoundingClientRect().height;
  postMessage({
    type: "size",
    height: Math.ceil(canvasHeight + 24)
  });
}

function afterNextPaint() {
  return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
}

function normalizedZoom(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(3, Math.max(0.5, number)) : 1;
}

function applyZoom(value) {
  currentZoom = normalizedZoom(value);
  if (!canvas) return;

  const previousWidth = Math.max(root.scrollWidth, 1);
  const horizontalCenter = (root.scrollLeft + root.clientWidth / 2) / previousWidth;
  const heightLimitedWidth = canvasAspectRatio == null
    ? Number.POSITIVE_INFINITY
    : maximumFitHeight * canvasAspectRatio * currentZoom;
  canvas.style.width = `min(${currentZoom * 100}%, ${heightLimitedWidth}px)`;
  root.classList.toggle("can-pan", currentZoom > 1);

  requestAnimationFrame(() => {
    root.scrollLeft = horizontalCenter * root.scrollWidth - root.clientWidth / 2;
    reportSize();
  });
}

function configuration(theme) {
  const dark = theme === "dark";
  return {
    startOnLoad: false,
    securityLevel: "strict",
    theme: "base",
    fontFamily: "Avenir Next, Avenir, sans-serif",
    flowchart: { curve: "basis", htmlLabels: true, useMaxWidth: true },
    themeVariables: dark
      ? {
          background: "#1b1e22",
          primaryColor: "#24364a",
          primaryTextColor: "#f3f5f7",
          primaryBorderColor: "#72a7d8",
          secondaryColor: "#3a3021",
          tertiaryColor: "#1f3a31",
          lineColor: "#aeb8c2"
        }
      : {
          background: "#ffffff",
          primaryColor: "#e5f0fa",
          primaryTextColor: "#17212b",
          primaryBorderColor: "#477aa8",
          secondaryColor: "#f4ecdb",
          tertiaryColor: "#e3f1e9",
          lineColor: "#586674"
        }
  };
}

function svgAspectRatio(image) {
  const values = image?.getAttribute("viewBox")?.trim().split(/\s+/).map(Number);
  if (values?.length !== 4 || values.some(value => !Number.isFinite(value)) || values[3] <= 0) {
    return null;
  }
  return values[2] / values[3];
}

function verticalFallback(source, image) {
  const aspectRatio = svgAspectRatio(image);
  if (aspectRatio == null || aspectRatio <= maximumReadableAspectRatio) {
    return null;
  }
  return source.replace(/^(\s*flowchart\s+)(LR|RL)\b/, "$1TD");
}

async function renderedCanvas(renderID, source) {
  const result = await mermaid.render(renderID, source);
  const nextCanvas = document.createElement("div");
  nextCanvas.className = "diagram-canvas";
  nextCanvas.innerHTML = result.svg;
  return {
    bindFunctions: result.bindFunctions,
    canvas: nextCanvas,
    image: nextCanvas.querySelector("svg")
  };
}

window.InferenceSchoolDiagram = {
  async render(payload) {
    const sequence = ++renderSequence;
    currentZoom = normalizedZoom(payload.zoom);
    root.setAttribute("aria-label", payload.title || "Lesson diagram");
    try {
      mermaid.initialize(configuration(payload.theme));
      const renderID = `leet-diagram-${String(payload.id).replace(/[^a-zA-Z0-9_-]/g, "-")}-${sequence}`;
      let rendered = await renderedCanvas(`${renderID}-requested`, payload.source);
      const fallbackSource = verticalFallback(payload.source, rendered.image);
      if (fallbackSource && fallbackSource !== payload.source) {
        rendered = await renderedCanvas(`${renderID}-vertical`, fallbackSource);
      }
      if (sequence !== renderSequence) return;
      canvas = rendered.canvas;
      canvasAspectRatio = svgAspectRatio(rendered.image);
      root.replaceChildren(canvas);
      const image = rendered.image;
      image?.setAttribute("role", "img");
      image?.setAttribute("aria-label", payload.title || "Lesson diagram");
      rendered.bindFunctions?.(root);
      canvasObserver?.disconnect();
      canvasObserver = new ResizeObserver(reportSize);
      canvasObserver.observe(canvas);
      applyZoom(currentZoom);
      await afterNextPaint();
      if (sequence !== renderSequence) return;
      const bounds = image?.getBoundingClientRect();
      postMessage({
        type: "rendered",
        svgCount: root.querySelectorAll("svg").length,
        graphicsCount: image?.querySelectorAll(
          "path, rect, circle, ellipse, line, polyline, polygon, text, foreignObject"
        ).length ?? 0,
        text: image?.textContent?.trim() ?? "",
        width: bounds?.width ?? 0,
        height: bounds?.height ?? 0
      });
    } catch (error) {
      if (sequence !== renderSequence) return;
      root.replaceChildren();
      const message = document.createElement("pre");
      message.className = "diagram-error";
      message.textContent = `Diagram could not render\n${error instanceof Error ? error.message : String(error)}`;
      root.append(message);
      reportSize();
      postMessage({ type: "error", message: message.textContent });
    }
  },

  setZoom(value) {
    applyZoom(value);
  }
};

root.addEventListener("pointerdown", (event) => {
  if (currentZoom <= 1 || event.button !== 0) return;
  dragState = {
    pointerID: event.pointerId,
    x: event.clientX,
    y: event.clientY,
    scrollLeft: root.scrollLeft,
    scrollTop: root.scrollTop
  };
  root.setPointerCapture(event.pointerId);
  root.classList.add("is-panning");
  event.preventDefault();
});

root.addEventListener("pointermove", (event) => {
  if (!dragState || event.pointerId !== dragState.pointerID) return;
  root.scrollLeft = dragState.scrollLeft - (event.clientX - dragState.x);
  root.scrollTop = dragState.scrollTop - (event.clientY - dragState.y);
  event.preventDefault();
});

function stopPanning(event) {
  if (!dragState || event.pointerId !== dragState.pointerID) return;
  dragState = null;
  root.classList.remove("is-panning");
}

root.addEventListener("pointerup", stopPanning);
root.addEventListener("pointercancel", stopPanning);