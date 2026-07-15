@testable import InferenceSchoolStudio
import XCTest

final class WorkbenchPanelTests: XCTestCase {
    func testUsesBottomPanelBelowWideLayoutThreshold() {
        XCTAssertEqual(
            WorkbenchPanel.placement(
                forWidth: WorkbenchPanel.trailingLayoutMinimumWidth - 1
            ),
            .bottom
        )
    }

    func testUsesTrailingPanelAtWideLayoutThreshold() {
        XCTAssertEqual(
            WorkbenchPanel.placement(
                forWidth: WorkbenchPanel.trailingLayoutMinimumWidth
            ),
            .trailing
        )
    }

    func testPlacementProvidesDirectionalCollapseAndRestoreSymbols() {
        XCTAssertEqual(WorkbenchPanel.Placement.bottom.collapseSymbol, "chevron.down")
        XCTAssertEqual(WorkbenchPanel.Placement.bottom.restoreSymbol, "chevron.up")
        XCTAssertEqual(WorkbenchPanel.Placement.trailing.collapseSymbol, "chevron.right")
        XCTAssertEqual(WorkbenchPanel.Placement.trailing.restoreSymbol, "chevron.left")
    }
}