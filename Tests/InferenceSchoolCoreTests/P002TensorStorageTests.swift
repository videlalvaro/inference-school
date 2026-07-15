import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P002TensorStorageTests: XCTestCase {
    func testCanonicalSolutionPassesJudge() {
        let report = P002TensorStorageJudge.evaluate(P002TensorStorageSolution.gather)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsWrongValuesEvenWhenValidationIsPreserved() {
        let report = P002TensorStorageJudge.evaluate { storage, shape, indices in
            _ = try FloatTensor(storage, shape: shape)
            for index in indices {
                _ = try TensorLayout.rowMajor(shape: shape).offset(for: index)
            }
            return Array(repeating: 0, count: indices.count)
        }
        XCTAssertFalse(report.isPassing)
    }

    func testRowMajorLayoutAndOffsets() throws {
        let layout = try TensorLayout.rowMajor(shape: [2, 3, 4])
        XCTAssertEqual(layout.strides, [12, 4, 1])
        XCTAssertEqual(try layout.offset(for: [1, 2, 3]), 23)
    }

    func testContiguousViewCanReshapeWithoutChangingStorage() throws {
        let tensor = try FloatTensor((0..<6).map(Float.init), shape: [2, 3])
        let reshaped = try tensor.view.reshaped(to: [3, 2])
        XCTAssertEqual(reshaped.shape, [3, 2])
        XCTAssertEqual(reshaped.strides, [2, 1])
        XCTAssertEqual(try reshaped.value(at: [2, 1]), 5)
    }

    func testTransposeViewUsesStridesAndCannotReshape() throws {
        let tensor = try FloatTensor((0..<6).map(Float.init), shape: [2, 3])
        let transpose = try tensor.view.transposed2D()
        XCTAssertEqual(transpose.shape, [3, 2])
        XCTAssertEqual(transpose.strides, [1, 3])
        XCTAssertEqual(try transpose.value(at: [2, 1]), 5)
        XCTAssertThrowsError(try transpose.reshaped(to: [6])) { error in
            XCTAssertEqual(error as? TensorError, .reshapeRequiresContiguousLayout)
        }
    }

    func testTensorRejectsStorageMismatchAndBadIndex() throws {
        XCTAssertThrowsError(try FloatTensor([1, 2], shape: [3]))
        let tensor = try FloatTensor([1, 2], shape: [2])
        XCTAssertThrowsError(try tensor.value(at: [2]))
        XCTAssertThrowsError(try tensor.value(at: [0, 0]))
    }
}