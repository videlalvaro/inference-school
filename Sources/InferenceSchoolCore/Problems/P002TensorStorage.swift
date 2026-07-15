import Foundation

public enum TensorError: Error, Equatable, LocalizedError {
    case invalidDimension(axis: Int, value: Int)
    case invalidStride(axis: Int, value: Int)
    case rankMismatch(expected: Int, actual: Int)
    case elementCountOverflow
    case storageCountMismatch(expected: Int, actual: Int)
    case indexOutOfBounds(axis: Int, index: Int, dimension: Int)
    case offsetOverflow
    case invalidViewOffset(Int)
    case viewExceedsStorage
    case reshapeElementCountMismatch(expected: Int, actual: Int)
    case reshapeRequiresContiguousLayout

    public var errorDescription: String? {
        switch self {
        case let .invalidDimension(axis, value):
            "Shape dimension \(axis) must be nonnegative; received \(value)."
        case let .invalidStride(axis, value):
            "Stride \(axis) must be nonnegative; received \(value)."
        case let .rankMismatch(expected, actual):
            "Index/layout rank must be \(expected); received \(actual)."
        case .elementCountOverflow:
            "The tensor element count exceeds Int.max."
        case let .storageCountMismatch(expected, actual):
            "Tensor storage must contain \(expected) values; received \(actual)."
        case let .indexOutOfBounds(axis, index, dimension):
            "Index \(index) is outside dimension \(axis) with size \(dimension)."
        case .offsetOverflow:
            "The tensor offset exceeds Int.max."
        case let .invalidViewOffset(offset):
            "A tensor view offset must be nonnegative; received \(offset)."
        case .viewExceedsStorage:
            "The tensor view addresses values outside its storage."
        case let .reshapeElementCountMismatch(expected, actual):
            "Reshape must preserve \(expected) elements; requested \(actual)."
        case .reshapeRequiresContiguousLayout:
            "Only a contiguous tensor view can be reshaped without copying."
        }
    }
}

public struct TensorLayout: Sendable, Equatable {
    public let shape: [Int]
    public let strides: [Int]
    public let elementCount: Int

    public var rank: Int { shape.count }

    public var isContiguous: Bool {
        guard let rowMajor = try? Self.rowMajor(shape: shape) else { return false }
        return strides == rowMajor.strides
    }

    public init(shape: [Int], strides: [Int]) throws {
        guard shape.count == strides.count else {
            throw TensorError.rankMismatch(expected: shape.count, actual: strides.count)
        }
        for (axis, dimension) in shape.enumerated() where dimension < 0 {
            throw TensorError.invalidDimension(axis: axis, value: dimension)
        }
        for (axis, stride) in strides.enumerated() where stride < 0 {
            throw TensorError.invalidStride(axis: axis, value: stride)
        }

        self.shape = shape
        self.strides = strides
        self.elementCount = try Self.checkedElementCount(shape)
    }

    public static func rowMajor(shape: [Int]) throws -> TensorLayout {
        var strides = Array(repeating: 0, count: shape.count)
        var runningStride = 1

        for axis in shape.indices.reversed() {
            let dimension = shape[axis]
            guard dimension >= 0 else {
                throw TensorError.invalidDimension(axis: axis, value: dimension)
            }
            strides[axis] = runningStride
            let (nextStride, overflow) = runningStride.multipliedReportingOverflow(by: dimension)
            guard !overflow else { throw TensorError.elementCountOverflow }
            runningStride = nextStride
        }

        return try TensorLayout(shape: shape, strides: strides)
    }

    public func offset(for indices: [Int]) throws -> Int {
        guard indices.count == rank else {
            throw TensorError.rankMismatch(expected: rank, actual: indices.count)
        }

        var offset = 0
        for axis in shape.indices {
            let index = indices[axis]
            guard index >= 0, index < shape[axis] else {
                throw TensorError.indexOutOfBounds(
                    axis: axis,
                    index: index,
                    dimension: shape[axis]
                )
            }
            let (term, multiplyOverflow) = index.multipliedReportingOverflow(by: strides[axis])
            let (nextOffset, addOverflow) = offset.addingReportingOverflow(term)
            guard !multiplyOverflow, !addOverflow else { throw TensorError.offsetOverflow }
            offset = nextOffset
        }
        return offset
    }

    private static func checkedElementCount(_ shape: [Int]) throws -> Int {
        var count = 1
        for (axis, dimension) in shape.enumerated() {
            guard dimension >= 0 else {
                throw TensorError.invalidDimension(axis: axis, value: dimension)
            }
            let (nextCount, overflow) = count.multipliedReportingOverflow(by: dimension)
            guard !overflow else { throw TensorError.elementCountOverflow }
            count = nextCount
        }
        return count
    }
}

public struct FloatTensorView: Sendable, Equatable {
    public let storage: [Float]
    public let offset: Int
    public let layout: TensorLayout

    public var shape: [Int] { layout.shape }
    public var strides: [Int] { layout.strides }
    public var rank: Int { layout.rank }
    public var elementCount: Int { layout.elementCount }
    public var isContiguous: Bool { layout.isContiguous }

    public init(
        storage: [Float],
        offset: Int = 0,
        shape: [Int],
        strides: [Int]
    ) throws {
        guard offset >= 0 else { throw TensorError.invalidViewOffset(offset) }
        let layout = try TensorLayout(shape: shape, strides: strides)

        if layout.elementCount == 0 {
            guard offset <= storage.count else { throw TensorError.viewExceedsStorage }
        } else {
            var maximumOffset = offset
            for axis in shape.indices {
                let (term, multiplyOverflow) = (shape[axis] - 1)
                    .multipliedReportingOverflow(by: strides[axis])
                let (nextOffset, addOverflow) = maximumOffset.addingReportingOverflow(term)
                guard !multiplyOverflow, !addOverflow else { throw TensorError.offsetOverflow }
                maximumOffset = nextOffset
            }
            guard maximumOffset < storage.count else { throw TensorError.viewExceedsStorage }
        }

        self.storage = storage
        self.offset = offset
        self.layout = layout
    }

    public func value(at indices: [Int]) throws -> Float {
        let relativeOffset = try layout.offset(for: indices)
        let (absoluteOffset, overflow) = offset.addingReportingOverflow(relativeOffset)
        guard !overflow else { throw TensorError.offsetOverflow }
        return storage[absoluteOffset]
    }

    public func reshaped(to newShape: [Int]) throws -> FloatTensorView {
        guard isContiguous else { throw TensorError.reshapeRequiresContiguousLayout }
        let newLayout = try TensorLayout.rowMajor(shape: newShape)
        guard newLayout.elementCount == elementCount else {
            throw TensorError.reshapeElementCountMismatch(
                expected: elementCount,
                actual: newLayout.elementCount
            )
        }
        return try FloatTensorView(
            storage: storage,
            offset: offset,
            shape: newShape,
            strides: newLayout.strides
        )
    }

    public func transposed2D() throws -> FloatTensorView {
        guard rank == 2 else { throw TensorError.rankMismatch(expected: 2, actual: rank) }
        return try FloatTensorView(
            storage: storage,
            offset: offset,
            shape: [shape[1], shape[0]],
            strides: [strides[1], strides[0]]
        )
    }
}

public struct FloatTensor: Sendable, Equatable {
    public let storage: [Float]
    public let layout: TensorLayout

    public var shape: [Int] { layout.shape }
    public var strides: [Int] { layout.strides }
    public var rank: Int { layout.rank }
    public var elementCount: Int { layout.elementCount }
    public var view: FloatTensorView {
        try! FloatTensorView(storage: storage, shape: shape, strides: strides)
    }

    public init(_ storage: [Float], shape: [Int]) throws {
        let layout = try TensorLayout.rowMajor(shape: shape)
        guard storage.count == layout.elementCount else {
            throw TensorError.storageCountMismatch(
                expected: layout.elementCount,
                actual: storage.count
            )
        }
        self.storage = storage
        self.layout = layout
    }

    public func value(at indices: [Int]) throws -> Float {
        storage[try layout.offset(for: indices)]
    }
}

public typealias TensorGatherImplementation = (
    _ storage: [Float],
    _ shape: [Int],
    _ indices: [[Int]]
) throws -> [Float]

public enum P002TensorStorageJudge {
    private struct ValueCase {
        let name: String
        let storage: [Float]
        let shape: [Int]
        let indices: [[Int]]
        let expected: [Float]
    }

    public static func evaluate(_ implementation: TensorGatherImplementation) -> JudgeReport {
        let valueCases = [
            ValueCase(
                name: "row-major matrix corners",
                storage: [0, 1, 2, 3, 4, 5],
                shape: [2, 3],
                indices: [[0, 0], [0, 2], [1, 0], [1, 2]],
                expected: [0, 2, 3, 5]
            ),
            ValueCase(
                name: "rank-three offsets",
                storage: (0..<24).map(Float.init),
                shape: [2, 3, 4],
                indices: [[0, 1, 2], [1, 0, 0], [1, 2, 3]],
                expected: [6, 12, 23]
            ),
            ValueCase(
                name: "scalar tensor",
                storage: [42],
                shape: [],
                indices: [[]],
                expected: [42]
            ),
            ValueCase(
                name: "empty gather",
                storage: [],
                shape: [2, 0, 3],
                indices: [],
                expected: []
            ),
        ]

        var failures: [JudgeFailure] = []
        var passed = 0
        for testCase in valueCases {
            do {
                let actual = try implementation(testCase.storage, testCase.shape, testCase.indices)
                if actual == testCase.expected {
                    passed += 1
                } else {
                    failures.append(JudgeFailure(
                        caseName: testCase.name,
                        message: "expected \(testCase.expected), received \(actual)"
                    ))
                }
            } catch {
                failures.append(JudgeFailure(
                    caseName: testCase.name,
                    message: "unexpected error: \(error.localizedDescription)"
                ))
            }
        }

        passed += expectError(
            name: "reject storage mismatch",
            failures: &failures
        ) {
            _ = try implementation([1, 2], [3], [[0]])
        }
        passed += expectError(
            name: "reject out-of-bounds index",
            failures: &failures
        ) {
            _ = try implementation([1, 2], [2], [[2]])
        }

        return JudgeReport(
            passedCaseCount: passed,
            totalCaseCount: valueCases.count + 2,
            failures: failures
        )
    }

    private static func expectError(
        name: String,
        failures: inout [JudgeFailure],
        operation: () throws -> Void
    ) -> Int {
        do {
            try operation()
            failures.append(JudgeFailure(
                caseName: name,
                message: "expected an error, but the implementation returned"
            ))
            return 0
        } catch {
            return 1
        }
    }
}