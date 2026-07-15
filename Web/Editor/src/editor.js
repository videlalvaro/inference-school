import { indentWithTab } from "@codemirror/commands";
import { cpp } from "@codemirror/lang-cpp";
import { StreamLanguage } from "@codemirror/language";
import { swift } from "@codemirror/legacy-modes/mode/swift";
import { setDiagnostics } from "@codemirror/lint";
import { Compartment, EditorState, Prec } from "@codemirror/state";
import { EditorView, keymap } from "@codemirror/view";
import { basicSetup } from "codemirror";

const language = new Compartment();
const editable = new Compartment();
const theme = new Compartment();
const textSize = new Compartment();
let documentID = "untitled";
let languageID = "swift";
const darkMode = window.matchMedia("(prefers-color-scheme: dark)");

function post(type, payload = {}) {
  window.webkit?.messageHandlers?.editor?.postMessage({
    type,
    documentID,
    ...payload
  });
}

function languageExtension(identifier) {
  switch (identifier) {
    case "metal":
    case "cpp":
      return cpp();
    case "swift":
      return StreamLanguage.define(swift);
    default:
      return [];
  }
}

const lightTheme = EditorView.theme({
  "&": { color: "#24292f", backgroundColor: "#f7f8fa" },
  ".cm-content": { caretColor: "#0969da", padding: "0" },
  ".cm-cursor, .cm-dropCursor": { borderLeftColor: "#0969da" },
  ".cm-gutters": {
    backgroundColor: "#eef1f4",
    color: "#6e7781",
    borderRight: "1px solid #d8dee4"
  },
  ".cm-activeLine": { backgroundColor: "#ddf4ff80" },
  ".cm-activeLineGutter": { backgroundColor: "#b6e3ff" },
  ".cm-selectionBackground, ::selection": { backgroundColor: "#54aeff55 !important" },
  ".cm-focused": { outline: "none" }
});

const darkTheme = EditorView.theme({
  "&": { color: "#e6edf3", backgroundColor: "#17191c" },
  ".cm-content": { caretColor: "#58a6ff", padding: "0" },
  ".cm-cursor, .cm-dropCursor": { borderLeftColor: "#58a6ff" },
  ".cm-gutters": {
    backgroundColor: "#202328",
    color: "#8b949e",
    borderRight: "1px solid #30363d"
  },
  ".cm-activeLine": { backgroundColor: "#1f6feb22" },
  ".cm-activeLineGutter": { backgroundColor: "#1f6feb44" },
  ".cm-selectionBackground, ::selection": { backgroundColor: "#2f81f755 !important" },
  ".cm-focused": { outline: "none" }
}, { dark: true });

const runAndSaveKeys = Prec.highest(keymap.of([
  {
    key: "Mod-Enter",
    run() {
      post("run");
      return true;
    }
  },
  {
    key: "Mod-s",
    run(view) {
      post("save", { text: view.state.doc.toString() });
      return true;
    }
  },
  indentWithTab
]));

const view = new EditorView({
  parent: document.querySelector("#editor"),
  state: EditorState.create({
    doc: "",
    extensions: [
      basicSetup,
      runAndSaveKeys,
      language.of(languageExtension(languageID)),
      editable.of(EditorView.editable.of(true)),
      theme.of(darkMode.matches ? darkTheme : lightTheme),
      textSize.of(EditorView.theme({
        "&": { fontSize: "13px" }
      })),
      EditorState.allowMultipleSelections.of(true),
      EditorView.contentAttributes.of({
        "aria-label": "Source editor",
        autocapitalize: "off",
        autocomplete: "off",
        autocorrect: "off",
        spellcheck: "false"
      }),
      EditorView.updateListener.of((update) => {
        if (update.docChanged) {
          post("change", { text: update.state.doc.toString() });
        }
      })
    ]
  })
});

darkMode.addEventListener("change", (event) => {
  view.dispatch({ effects: theme.reconfigure(event.matches ? darkTheme : lightTheme) });
});

window.InferenceSchoolEditor = {
  setDocument(document) {
    const effects = [];
    const nextLanguage = document.language || "text";
    if (nextLanguage !== languageID) {
      languageID = nextLanguage;
      effects.push(language.reconfigure(languageExtension(languageID)));
    }
    effects.push(editable.reconfigure(EditorView.editable.of(document.editable !== false)));
    const scale = Math.min(2, Math.max(0.8, Number(document.textScale) || 1));
    const lineHeight = Math.round(13 * scale * 1.4);
    effects.push(textSize.reconfigure(EditorView.theme({
      "&": { fontSize: `${13 * scale}px` },
      ".cm-scroller": { lineHeight: `${lineHeight}px` },
      ".cm-lineNumbers .cm-gutterElement:not(:first-child)": {
        height: `${lineHeight}px !important`
      }
    })));
    documentID = document.id || "untitled";
    const currentText = view.state.doc.toString();
    const nextText = document.text || "";
    view.dispatch({
      changes: currentText === nextText
        ? undefined
        : { from: 0, to: currentText.length, insert: nextText },
      effects
    });
    view.requestMeasure();
  },
  setDiagnostics(diagnostics) {
    const mapped = diagnostics.map((diagnostic) => ({
      from: diagnostic.from,
      to: Math.max(diagnostic.from, diagnostic.to),
      severity: diagnostic.severity,
      message: diagnostic.message
    }));
    view.dispatch(setDiagnostics(view.state, mapped));
  },
  focus() {
    view.focus();
  },
  getText() {
    return view.state.doc.toString();
  }
};

post("ready");