import AppKit
import Foundation
import InferenceSchoolLessonKit
import SwiftUI
import WebKit
@testable import InferenceSchoolStudio
import XCTest

@MainActor
final class MermaidDiagramRenderingTests: XCTestCase {
    func testProblem003RendersThroughSwiftUIViewLifecycle() async throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let lesson = try XCTUnwrap(
            LessonCatalog.discover(in: repositoryRoot).first { $0.id == "003" }
        )
        let source = try XCTUnwrap(
            lesson.activities.first { $0.id == "p003-tiled-transpose-copy" }?.configuration
        )
        let view = MermaidDiagramView(
            id: "p003-tiled-transpose-copy",
            source: source,
            title: "Transpose and Tiled Copy diagram"
        )
        let hostingView = NSHostingView(rootView: view)
        let window = NSWindow(
            contentRect: CGRect(x: 0, y: 0, width: 860, height: 760),
            styleMask: [.titled],
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        window.contentView = hostingView
        window.orderFront(nil)
        defer { window.close() }

        let webView = try await waitForWebView(in: hostingView)
        let result = try await waitForRenderedDiagram(in: webView)

        XCTAssertEqual(result.svgCount, 1)
        XCTAssertGreaterThan(result.width, 2)
        XCTAssertGreaterThan(result.height, 2)
        XCTAssertGreaterThan(result.textLength, 0)

        let snapshot = try await webView.takeSnapshot(configuration: nil)
        XCTAssertGreaterThan(try visiblePixelCount(in: snapshot), 500)
    }

    func testEveryBuiltInDiagramRendersVisibleSVGInWebKit() async throws {
        let repositoryRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let lessons = try LessonCatalog.discover(in: repositoryRoot)
        let diagrams = lessons.flatMap { lesson in
            LessonMarkdownRendering.blocks(in: lesson).compactMap { block -> Diagram? in
                guard case let .mermaid(id, source) = block else { return nil }
                return Diagram(id: id, source: source, lesson: lesson.relativePath)
            }
        }

        XCTAssertEqual(diagrams.count, 49)
        let transposeDiagram = try XCTUnwrap(
            diagrams.first { $0.id == "p003-tiled-transpose-copy" }
        )
        XCTAssertTrue(transposeDiagram.source.hasPrefix("flowchart TD"))

        let resourceURL = try XCTUnwrap(DiagramRendererResources.pageURL)
        let webView = WKWebView(
            frame: CGRect(x: 0, y: 0, width: 860, height: 900),
            configuration: WKWebViewConfiguration()
        )
        let window = NSWindow(
            contentRect: webView.frame,
            styleMask: [.titled],
            backing: .buffered,
            defer: false
        )
        window.isReleasedWhenClosed = false
        window.contentView = webView
        window.orderFront(nil)
        defer { window.close() }
        let navigationDelegate = NavigationDelegate(
            expectation: expectation(description: "Diagram renderer loads")
        )
        webView.navigationDelegate = navigationDelegate
        webView.loadFileURL(
            resourceURL,
            allowingReadAccessTo: resourceURL.deletingLastPathComponent()
        )
        await fulfillment(of: [navigationDelegate.expectation], timeout: 10)
        if let navigationError = navigationDelegate.error {
            throw navigationError
        }

        for diagram in diagrams {
            for theme in ["light", "dark"] {
                let context = "\(diagram.lesson)#\(diagram.id) [\(theme)]"
                let result: Result
                do {
                    result = try await render(diagram, theme: theme, in: webView)
                } catch {
                    XCTFail("\(context): \(error)")
                    throw error
                }
                XCTAssertFalse(result.hasError, result.error, file: #filePath, line: #line)
                XCTAssertEqual(result.svgCount, 1, context, file: #filePath, line: #line)
                XCTAssertGreaterThan(result.graphicsCount, 0, context, file: #filePath, line: #line)
                XCTAssertGreaterThan(result.textLength, 0, context, file: #filePath, line: #line)
                XCTAssertGreaterThan(result.width, 2, context, file: #filePath, line: #line)
                XCTAssertGreaterThan(result.height, 2, context, file: #filePath, line: #line)
                XCTAssertGreaterThan(result.viewBoxWidth, 0, context, file: #filePath, line: #line)
                XCTAssertGreaterThan(result.viewBoxHeight, 0, context, file: #filePath, line: #line)
                let aspectRatio = result.viewBoxWidth / result.viewBoxHeight
                XCTAssertGreaterThanOrEqual(aspectRatio, 1 / 6, context, file: #filePath, line: #line)
                XCTAssertLessThanOrEqual(aspectRatio, 6, context, file: #filePath, line: #line)
            }
        }
    }

    private func render(
        _ diagram: Diagram,
        theme: String,
        in webView: WKWebView
    ) async throws -> Result {
        let value = try await webView.callAsyncJavaScript(
            """
            const root = document.querySelector("#diagram");
                        root.replaceChildren();
                        window.InferenceSchoolDiagram.render({ id, source, title, theme, zoom: 1 });
                        const deadline = Date.now() + 5000;
                        while (!root.querySelector(".diagram-canvas svg, .diagram-error")) {
                            if (Date.now() >= deadline) throw new Error(`Timed out rendering ${id}`);
                            await new Promise(resolve => setTimeout(resolve, 25));
                        }
            const image = root?.querySelector(".diagram-canvas svg");
            const bounds = image?.getBoundingClientRect();
            const viewBox = image?.viewBox?.baseVal;
            return {
              hasError: Boolean(root?.querySelector(".diagram-error")),
              error: root?.querySelector(".diagram-error")?.textContent ?? "",
              svgCount: root?.querySelectorAll("svg").length ?? 0,
              graphicsCount: image?.querySelectorAll(
                "path, rect, circle, ellipse, line, polyline, polygon, text, foreignObject"
              ).length ?? 0,
              textLength: image?.textContent?.trim().length ?? 0,
              width: bounds?.width ?? 0,
              height: bounds?.height ?? 0,
              viewBoxWidth: viewBox?.width ?? 0,
              viewBoxHeight: viewBox?.height ?? 0
            };
            """,
            arguments: [
                "id": diagram.id,
                "source": diagram.source,
                "title": "\(diagram.lesson) diagram",
                "theme": theme,
            ],
            in: nil,
            contentWorld: .page
        )
        let values = try XCTUnwrap(value as? [String: Any])
        return try Result(values: values)
    }

    private func waitForWebView(in root: NSView) async throws -> WKWebView {
        for _ in 0..<100 {
            if let webView = firstWebView(in: root) {
                return webView
            }
            try await Task.sleep(for: .milliseconds(25))
        }
        throw DiagramTestError.timedOut("waiting for the SwiftUI WKWebView")
    }

    private func firstWebView(in view: NSView) -> WKWebView? {
        if let webView = view as? WKWebView { return webView }
        for subview in view.subviews {
            if let webView = firstWebView(in: subview) { return webView }
        }
        return nil
    }

    private func waitForRenderedDiagram(in webView: WKWebView) async throws -> Result {
        for _ in 0..<200 {
            if let result = try? await renderedDiagramResult(in: webView), result.svgCount == 1 {
                return result
            }
            try await Task.sleep(for: .milliseconds(25))
        }
        throw DiagramTestError.timedOut("waiting for the SwiftUI diagram render")
    }

    private func renderedDiagramResult(in webView: WKWebView) async throws -> Result {
        let value = try await webView.callAsyncJavaScript(
            """
            const root = document.querySelector("#diagram");
            const image = root?.querySelector(".diagram-canvas svg");
            const bounds = image?.getBoundingClientRect();
            const viewBox = image?.viewBox?.baseVal;
            return {
              hasError: Boolean(root?.querySelector(".diagram-error")),
              error: root?.querySelector(".diagram-error")?.textContent ?? "",
              svgCount: root?.querySelectorAll("svg").length ?? 0,
              graphicsCount: image?.querySelectorAll(
                "path, rect, circle, ellipse, line, polyline, polygon, text, foreignObject"
              ).length ?? 0,
              textLength: image?.textContent?.trim().length ?? 0,
              width: bounds?.width ?? 0,
              height: bounds?.height ?? 0,
              viewBoxWidth: viewBox?.width ?? 0,
              viewBoxHeight: viewBox?.height ?? 0
            };
            """,
            arguments: [:],
            in: nil,
            contentWorld: .page
        )
        return try Result(values: XCTUnwrap(value as? [String: Any]))
    }

    private func visiblePixelCount(in image: NSImage) throws -> Int {
        let representation = try XCTUnwrap(
            image.cgImage(forProposedRect: nil, context: nil, hints: nil)
        )
        guard let providerData = representation.dataProvider?.data,
              let bytes = CFDataGetBytePtr(providerData)
        else {
            throw DiagramTestError.missingSnapshotPixels
        }

        let backgroundOffset = 0
        let background = (
            bytes[backgroundOffset],
            bytes[backgroundOffset + 1],
            bytes[backgroundOffset + 2]
        )
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
}

private enum DiagramTestError: Error {
    case missingSnapshotPixels
    case timedOut(String)
}

private struct Diagram {
    let id: String
    let source: String
    let lesson: String
}

private struct Result {
    let hasError: Bool
    let error: String
    let svgCount: Int
    let graphicsCount: Int
    let textLength: Int
    let width: Double
    let height: Double
    let viewBoxWidth: Double
    let viewBoxHeight: Double

    init(values: [String: Any]) throws {
        hasError = try Self.value("hasError", in: values)
        error = try Self.value("error", in: values)
        svgCount = try Self.number("svgCount", in: values).intValue
        graphicsCount = try Self.number("graphicsCount", in: values).intValue
        textLength = try Self.number("textLength", in: values).intValue
        width = try Self.number("width", in: values).doubleValue
        height = try Self.number("height", in: values).doubleValue
        viewBoxWidth = try Self.number("viewBoxWidth", in: values).doubleValue
        viewBoxHeight = try Self.number("viewBoxHeight", in: values).doubleValue
    }

    private static func value<Value>(_ key: String, in values: [String: Any]) throws -> Value {
        try XCTUnwrap(values[key] as? Value, "Missing diagram result value \(key)")
    }

    private static func number(_ key: String, in values: [String: Any]) throws -> NSNumber {
        try value(key, in: values)
    }
}

@MainActor
private final class NavigationDelegate: NSObject, WKNavigationDelegate {
    let expectation: XCTestExpectation
    private(set) var error: Error?

    init(expectation: XCTestExpectation) {
        self.expectation = expectation
    }

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation?) {
        expectation.fulfill()
    }

    func webView(
        _ webView: WKWebView,
        didFail navigation: WKNavigation?,
        withError error: Error
    ) {
        self.error = error
        expectation.fulfill()
    }

    func webView(
        _ webView: WKWebView,
        didFailProvisionalNavigation navigation: WKNavigation?,
        withError error: Error
    ) {
        self.error = error
        expectation.fulfill()
    }
}