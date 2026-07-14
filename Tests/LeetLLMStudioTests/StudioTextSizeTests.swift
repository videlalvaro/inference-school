import SwiftUI
@testable import LeetLLMStudio
import XCTest

final class StudioTextSizeTests: XCTestCase {
    func testIncreaseRoundsAndClampsAtMaximum() {
        var value = 1.0

        StudioTextSize.increase(&value)
        XCTAssertEqual(value, 1.1)

        value = StudioTextSize.maximum
        StudioTextSize.increase(&value)
        XCTAssertEqual(value, StudioTextSize.maximum)
        XCTAssertFalse(StudioTextSize.canIncrease(value))
    }

    func testDecreaseRoundsAndClampsAtMinimum() {
        var value = 1.0

        StudioTextSize.decrease(&value)
        XCTAssertEqual(value, 0.9)

        value = StudioTextSize.minimum
        StudioTextSize.decrease(&value)
        XCTAssertEqual(value, StudioTextSize.minimum)
        XCTAssertFalse(StudioTextSize.canDecrease(value))
    }
}
