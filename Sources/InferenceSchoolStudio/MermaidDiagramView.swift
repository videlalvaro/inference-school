import SwiftUI
import WebKit

enum DiagramRendererResources {
    static var pageURL: URL? {
        Bundle.module.url(
            forResource: "index",
            withExtension: "html",
            subdirectory: "Resources/Diagram"
        )
    }
}

struct DiagramRenderMetrics: Equatable, Sendable {
    let svgCount: Int
    let graphicsCount: Int
    let text: String
    let width: Double
    let height: Double
    var visiblePixelCount: Int?

    var isVisible: Bool {
        svgCount == 1
            && graphicsCount > 0
            && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && width > 2
            && height > 2
    }
}

enum DiagramRenderEvent: Equatable, Sendable {
    case rendered(DiagramRenderMetrics)
    case failed(String)
}

struct MermaidDiagramView: View {
    let id: String
    let source: String
    let title: String
    var verifySnapshot = false
    var onRenderEvent: (DiagramRenderEvent) -> Void = { _ in }
    @State private var contentHeight: CGFloat = 240
    @State private var zoom = 1.0
    @State private var expandedZoom = 1.0
    @State private var isShowingExpanded = false
    @State private var renderError: String?
    @State private var renderAttempt = 0

    var body: some View {
        ZStack(alignment: .topTrailing) {
            ZStack {
                MermaidWebView(
                    id: id,
                    source: source,
                    title: title,
                    zoom: zoom,
                    contentHeight: $contentHeight,
                    renderError: $renderError,
                    verifySnapshot: verifySnapshot,
                    onRenderEvent: onRenderEvent
                )
                .id(renderAttempt)

                if let renderError {
                    DiagramRenderFailureView(message: renderError) {
                        self.renderError = nil
                        renderAttempt += 1
                    }
                }
            }

            DiagramZoomControls(zoom: $zoom) {
                expandedZoom = zoom
                isShowingExpanded = true
            }
                .padding(8)
        }
        .frame(maxWidth: .infinity)
        .frame(height: min(max(contentHeight, 160), 720))
        .background(.quaternary.opacity(0.18))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay {
            RoundedRectangle(cornerRadius: 6)
                .stroke(.separator.opacity(0.55), lineWidth: 1)
        }
        .sheet(isPresented: $isShowingExpanded) {
            ExpandedMermaidDiagramView(
                id: id,
                source: source,
                title: title,
                zoom: $expandedZoom
            )
        }
    }
}

private struct DiagramRenderFailureView: View {
    let message: String
    let retry: () -> Void

    var body: some View {
        VStack(spacing: 8) {
            Image(systemName: "exclamationmark.triangle")
                .font(.title2)
                .foregroundStyle(.secondary)
            Text("Diagram could not render")
                .font(.headline)
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .lineLimit(3)
            Button("Retry", action: retry)
        }
        .padding(16)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
    }
}

private enum DiagramZoom {
    static let minimum = 0.5
    static let maximum = 3.0
    static let step = 0.25
}

private struct DiagramZoomControls: View {
    @Binding var zoom: Double
    var onExpand: (() -> Void)?

    var body: some View {
        HStack(spacing: 2) {
            Button {
                zoom = max(DiagramZoom.minimum, zoom - DiagramZoom.step)
            } label: {
                Image(systemName: "minus.magnifyingglass")
            }
            .disabled(zoom <= DiagramZoom.minimum)
            .accessibilityLabel("Zoom out")
            .help("Zoom out")

            Text("\(Int((zoom * 100).rounded()))%")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(width: 38)
                .accessibilityLabel("Diagram zoom \(Int((zoom * 100).rounded())) percent")

            Button {
                zoom = 1
            } label: {
                Image(systemName: "arrow.down.right.and.arrow.up.left")
            }
            .disabled(zoom == 1)
            .accessibilityLabel("Fit diagram")
            .help("Fit diagram")

            Button {
                zoom = min(DiagramZoom.maximum, zoom + DiagramZoom.step)
            } label: {
                Image(systemName: "plus.magnifyingglass")
            }
            .disabled(zoom >= DiagramZoom.maximum)
            .accessibilityLabel("Zoom in")
            .help("Zoom in")

            if let onExpand {
                Divider()
                    .frame(height: 16)
                    .padding(.horizontal, 2)

                Button(action: onExpand) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                }
                .accessibilityLabel("Open enlarged diagram")
                .help("Open enlarged diagram")
            }
        }
        .buttonStyle(.borderless)
        .controlSize(.small)
        .padding(4)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 5))
    }
}

private struct ExpandedMermaidDiagramView: View {
    let id: String
    let source: String
    let title: String
    @Binding var zoom: Double
    @State private var contentHeight: CGFloat = 600
    @State private var renderError: String?
    @State private var renderAttempt = 0
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                Text(title)
                    .font(.headline)
                    .lineLimit(1)

                Spacer(minLength: 16)

                DiagramZoomControls(zoom: $zoom)

                Button {
                    dismiss()
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.borderless)
                .accessibilityLabel("Close diagram")
                .help("Close diagram")
            }
            .padding(12)

            Divider()

            ZStack {
                MermaidWebView(
                    id: "\(id)-expanded",
                    source: source,
                    title: title,
                    zoom: zoom,
                    contentHeight: $contentHeight,
                    renderError: $renderError,
                    verifySnapshot: false,
                    onRenderEvent: { _ in }
                )
                .id(renderAttempt)

                if let renderError {
                    DiagramRenderFailureView(message: renderError) {
                        self.renderError = nil
                        renderAttempt += 1
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(.quaternary.opacity(0.18))
        }
        .frame(
            minWidth: 760,
            idealWidth: 1080,
            maxWidth: .infinity,
            minHeight: 520,
            idealHeight: 760,
            maxHeight: .infinity
        )
    }
}

private struct MermaidWebView: NSViewRepresentable {
    let id: String
    let source: String
    let title: String
    let zoom: Double
    @Binding var contentHeight: CGFloat
    @Binding var renderError: String?
    let verifySnapshot: Bool
    let onRenderEvent: (DiagramRenderEvent) -> Void
    @Environment(\.colorScheme) private var colorScheme

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.userContentController.add(context.coordinator, name: "diagram")

        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.navigationDelegate = context.coordinator
        webView.setValue(false, forKey: "drawsBackground")
        context.coordinator.observeScrolling(in: webView)

        guard let resourceURL = DiagramRendererResources.pageURL else {
            assertionFailure("The bundled diagram renderer is missing.")
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
        coordinator.stopObservingScrolling()
        coordinator.cancelRenderTimeout()
        webView.configuration.userContentController.removeScriptMessageHandler(
            forName: "diagram"
        )
        webView.navigationDelegate = nil
    }

    @MainActor
    final class Coordinator: NSObject, WKNavigationDelegate, WKScriptMessageHandler {
        var parent: MermaidWebView
        private var isReady = false
        private var renderedPayload = ""
        private var appliedZoom: Double?
        private weak var webView: WKWebView?
        private var scrollMonitor: Any?
        private var renderTimeoutTask: Task<Void, Never>?

        init(_ parent: MermaidWebView) {
            self.parent = parent
        }

        func observeScrolling(in webView: WKWebView) {
            self.webView = webView
            scrollMonitor = NSEvent.addLocalMonitorForEvents(matching: .scrollWheel) {
                [weak self, weak webView] event in
                guard let self, let webView else { return event }
                return self.routeScrollEvent(event, over: webView)
            }
        }

        func stopObservingScrolling() {
            if let scrollMonitor {
                NSEvent.removeMonitor(scrollMonitor)
            }
            scrollMonitor = nil
            webView = nil
        }

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation?) {
            isReady = true
            synchronize(webView)
        }

        func webView(
            _ webView: WKWebView,
            didFail navigation: WKNavigation?,
            withError error: Error
        ) {
            reportFailure(error.localizedDescription)
        }

        func webView(
            _ webView: WKWebView,
            didFailProvisionalNavigation navigation: WKNavigation?,
            withError error: Error
        ) {
            reportFailure(error.localizedDescription)
        }

        func webViewWebContentProcessDidTerminate(_ webView: WKWebView) {
            isReady = false
            reportFailure("The WebKit content process stopped unexpectedly.")
        }

        func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage
        ) {
            guard message.name == "diagram",
                  let body = message.body as? [String: Any],
                  let type = body["type"] as? String
            else { return }

            if type == "size", let height = body["height"] as? NSNumber {
                parent.contentHeight = CGFloat(truncating: height)
            } else if type == "rendered",
                      let svgCount = body["svgCount"] as? NSNumber,
                      let graphicsCount = body["graphicsCount"] as? NSNumber,
                      let text = body["text"] as? String,
                      let width = body["width"] as? NSNumber,
                      let height = body["height"] as? NSNumber
            {
                let metrics = DiagramRenderMetrics(
                    svgCount: svgCount.intValue,
                    graphicsCount: graphicsCount.intValue,
                    text: text,
                    width: width.doubleValue,
                    height: height.doubleValue,
                    visiblePixelCount: nil
                )
                guard metrics.isVisible else {
                    reportFailure("The renderer returned an empty diagram.")
                    return
                }
                cancelRenderTimeout()
                parent.renderError = nil
                if parent.verifySnapshot {
                    guard let webView else {
                        reportFailure("The WebKit view was released before verification.")
                        return
                    }
                    verifySnapshot(metrics, in: webView)
                } else {
                    parent.onRenderEvent(.rendered(metrics))
                }
            } else if type == "error", let error = body["message"] as? String {
                reportFailure(error)
            }
        }

        func synchronize(_ webView: WKWebView?) {
            guard isReady, let webView else { return }
            let payload: [String: Any] = [
                "id": parent.id,
                "source": parent.source,
                "title": parent.title,
                "theme": parent.colorScheme == .dark ? "dark" : "light",
                "zoom": parent.zoom,
            ]
            guard let data = try? JSONSerialization.data(withJSONObject: payload),
                  let json = String(data: data, encoding: .utf8)
            else { return }
            let contentPayload = "\(parent.id)\u{0}\(parent.source)\u{0}\(parent.title)\u{0}\(parent.colorScheme)"
            if contentPayload != renderedPayload {
                renderedPayload = contentPayload
                appliedZoom = parent.zoom
                startRenderTimeout()
                webView.evaluateJavaScript("window.InferenceSchoolDiagram.render(\(json)); null") {
                    [weak self] _, error in
                    if let error {
                        self?.reportFailure(error.localizedDescription)
                    }
                }
            } else if appliedZoom != parent.zoom {
                appliedZoom = parent.zoom
                webView.evaluateJavaScript("window.InferenceSchoolDiagram.setZoom(\(parent.zoom)); null") {
                    [weak self] _, error in
                    if let error {
                        self?.reportFailure(error.localizedDescription)
                    }
                }
            }
        }

        func cancelRenderTimeout() {
            renderTimeoutTask?.cancel()
            renderTimeoutTask = nil
        }

        private func startRenderTimeout() {
            cancelRenderTimeout()
            renderTimeoutTask = Task { [weak self] in
                try? await Task.sleep(for: .seconds(10))
                guard !Task.isCancelled else { return }
                self?.reportFailure("The renderer did not finish within 10 seconds.")
            }
        }

        private func reportFailure(_ message: String) {
            cancelRenderTimeout()
            parent.renderError = message
            parent.onRenderEvent(.failed(message))
        }

        private func verifySnapshot(_ metrics: DiagramRenderMetrics, in webView: WKWebView) {
            webView.takeSnapshot(with: nil) { [weak self] image, error in
                guard let self else { return }
                if let error {
                    reportFailure("The rendered diagram could not be captured: \(error.localizedDescription)")
                    return
                }
                guard let image, let pixelCount = visiblePixelCount(in: image) else {
                    reportFailure("The rendered diagram snapshot contained no pixels.")
                    return
                }
                guard pixelCount > 500 else {
                    reportFailure("The rendered diagram snapshot was blank.")
                    return
                }
                var verifiedMetrics = metrics
                verifiedMetrics.visiblePixelCount = pixelCount
                parent.onRenderEvent(.rendered(verifiedMetrics))
            }
        }

        private func visiblePixelCount(in image: NSImage) -> Int? {
            guard let representation = image.cgImage(
                forProposedRect: nil,
                context: nil,
                hints: nil
            ), let providerData = representation.dataProvider?.data,
                let bytes = CFDataGetBytePtr(providerData),
                representation.bitsPerPixel >= 24
            else { return nil }

            let background = (bytes[0], bytes[1], bytes[2])
            var visiblePixels = 0
            for row in 0..<representation.height {
                for column in 0..<representation.width {
                    let offset = row * representation.bytesPerRow
                        + column * representation.bitsPerPixel / 8
                    let difference = max(
                        abs(Int(bytes[offset]) - Int(background.0)),
                        abs(Int(bytes[offset + 1]) - Int(background.1)),
                        abs(Int(bytes[offset + 2]) - Int(background.2))
                    )
                    if difference > 12 { visiblePixels += 1 }
                }
            }
            return visiblePixels
        }

        private func routeScrollEvent(_ event: NSEvent, over webView: WKWebView) -> NSEvent? {
            guard event.window === webView.window,
                  webView.bounds.contains(webView.convert(event.locationInWindow, from: nil)),
                  abs(event.scrollingDeltaY) > abs(event.scrollingDeltaX),
                  let lessonScrollView = enclosingScrollView(of: webView)
            else { return event }

            lessonScrollView.scrollWheel(with: event)
            return nil
        }

        private func enclosingScrollView(of webView: WKWebView) -> NSScrollView? {
            var ancestor = webView.superview
            while let current = ancestor {
                if let scrollView = current as? NSScrollView {
                    return scrollView
                }
                ancestor = current.superview
            }
            return nil
        }
    }
}