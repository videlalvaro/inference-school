import SwiftUI
import WebKit

struct CodeEditorView: NSViewRepresentable {
    @Binding var text: String
    let documentID: String
    let language: String
    let textScale: Double
    var isEditable = true
    var onRun: () -> Void = {}
    var onSave: () -> Void = {}

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.userContentController.add(context.coordinator, name: "editor")

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.setValue(false, forKey: "drawsBackground")
#if DEBUG
        webView.isInspectable = true
#endif

        guard let resourceURL = Bundle.module.url(
            forResource: "index",
            withExtension: "html",
            subdirectory: "Resources/Editor"
        ) else {
            assertionFailure("The bundled code editor is missing.")
            return webView
        }
        webView.loadFileURL(
            resourceURL,
            allowingReadAccessTo: resourceURL.deletingLastPathComponent()
        )
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.synchronize(webView)
    }

    static func dismantleNSView(_ webView: WKWebView, coordinator: Coordinator) {
        webView.configuration.userContentController.removeScriptMessageHandler(forName: "editor")
        webView.navigationDelegate = nil
    }

    @MainActor
    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        var parent: CodeEditorView
        private var isReady = false

        init(_ parent: CodeEditorView) {
            self.parent = parent
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation?) {
            synchronize(webView)
        }

        func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage
        ) {
            guard message.name == "editor",
                  let body = message.body as? [String: Any],
                  let type = body["type"] as? String
            else { return }

            switch type {
            case "ready":
                isReady = true
                synchronize(message.webView)
            case "change":
                guard body["documentID"] as? String == parent.documentID,
                      let text = body["text"] as? String,
                      text != parent.text
                else { return }
                parent.text = text
            case "save":
                if let text = body["text"] as? String, text != parent.text {
                    parent.text = text
                }
                parent.onSave()
            case "run":
                parent.onRun()
            default:
                break
            }
        }

        func synchronize(_ webView: WKWebView?) {
            guard isReady, let webView else { return }
            let document: [String: Any] = [
                "id": parent.documentID,
                "text": parent.text,
                "language": parent.language,
                "textScale": parent.textScale,
                "editable": parent.isEditable,
            ]
            guard let data = try? JSONSerialization.data(withJSONObject: document),
                  let json = String(data: data, encoding: .utf8)
            else { return }
            webView.evaluateJavaScript("window.InferenceSchoolEditor.setDocument(\(json))")
        }
    }
}