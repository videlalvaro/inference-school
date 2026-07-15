import Foundation

public enum RooflineError: Error, Equatable, LocalizedError {
    case invalidDimension(name: String, value: Int)
    case negativeFLOPs(Double)
    case nonpositiveBytes(Double)
    case nonpositiveComputePeak(Double)
    case nonpositiveBandwidth(Double)
    case nonpositiveDuration(UInt64)
    case nonpositiveIterations(Int)

    public var errorDescription: String? {
        switch self {
        case let .invalidDimension(name, value): "Dimension \(name) must be nonnegative; received \(value)."
        case let .negativeFLOPs(value): "FLOPs must be nonnegative; received \(value)."
        case let .nonpositiveBytes(value): "Bytes moved must be positive; received \(value)."
        case let .nonpositiveComputePeak(value): "Peak GFLOP/s must be positive; received \(value)."
        case let .nonpositiveBandwidth(value): "Peak GB/s must be positive; received \(value)."
        case let .nonpositiveDuration(value): "Measured duration must be positive; received \(value) ns."
        case let .nonpositiveIterations(value): "Benchmark iterations must be positive; received \(value)."
        }
    }
}

public struct RooflineWorkload: Sendable, Equatable {
    public let name: String
    public let floatingPointOperations: Double
    public let bytesMoved: Double

    public init(name: String, floatingPointOperations: Double, bytesMoved: Double) throws {
        guard floatingPointOperations >= 0 else {
            throw RooflineError.negativeFLOPs(floatingPointOperations)
        }
        guard bytesMoved > 0 else { throw RooflineError.nonpositiveBytes(bytesMoved) }
        self.name = name
        self.floatingPointOperations = floatingPointOperations
        self.bytesMoved = bytesMoved
    }

    public static func gemv(rows: Int, columns: Int) throws -> RooflineWorkload {
        guard rows >= 0 else { throw RooflineError.invalidDimension(name: "rows", value: rows) }
        guard columns >= 0 else {
            throw RooflineError.invalidDimension(name: "columns", value: columns)
        }
        let matrixElements = Double(rows) * Double(columns)
        let flops = 2 * matrixElements
        let elementsMoved = matrixElements + Double(columns) + Double(rows)
        return try RooflineWorkload(
            name: "GEMV [\(rows),\(columns)] x [\(columns)]",
            floatingPointOperations: flops,
            bytesMoved: elementsMoved * Double(MemoryLayout<Float>.stride)
        )
    }

    public static func gemm(m: Int, k: Int, n: Int) throws -> RooflineWorkload {
        guard m >= 0 else { throw RooflineError.invalidDimension(name: "M", value: m) }
        guard k >= 0 else { throw RooflineError.invalidDimension(name: "K", value: k) }
        guard n >= 0 else { throw RooflineError.invalidDimension(name: "N", value: n) }
        let flops = 2 * Double(m) * Double(k) * Double(n)
        let elementsMoved = Double(m) * Double(k)
            + Double(k) * Double(n)
            + Double(m) * Double(n)
        return try RooflineWorkload(
            name: "GEMM [\(m),\(k)] x [\(k),\(n)]",
            floatingPointOperations: flops,
            bytesMoved: elementsMoved * Double(MemoryLayout<Float>.stride)
        )
    }
}

public struct RooflineMachine: Sendable, Equatable {
    public let peakComputeGFLOPS: Double
    public let peakMemoryBandwidthGBps: Double

    public init(peakComputeGFLOPS: Double, peakMemoryBandwidthGBps: Double) throws {
        guard peakComputeGFLOPS > 0 else {
            throw RooflineError.nonpositiveComputePeak(peakComputeGFLOPS)
        }
        guard peakMemoryBandwidthGBps > 0 else {
            throw RooflineError.nonpositiveBandwidth(peakMemoryBandwidthGBps)
        }
        self.peakComputeGFLOPS = peakComputeGFLOPS
        self.peakMemoryBandwidthGBps = peakMemoryBandwidthGBps
    }
}

public enum RooflineBottleneck: String, Sendable, Equatable {
    case memory
    case compute
    case balanced
}

public struct RooflinePrediction: Sendable, Equatable {
    public let arithmeticIntensity: Double
    public let bandwidthCeilingGFLOPS: Double
    public let predictedCeilingGFLOPS: Double
    public let bottleneck: RooflineBottleneck

    public init(
        arithmeticIntensity: Double,
        bandwidthCeilingGFLOPS: Double,
        predictedCeilingGFLOPS: Double,
        bottleneck: RooflineBottleneck
    ) {
        self.arithmeticIntensity = arithmeticIntensity
        self.bandwidthCeilingGFLOPS = bandwidthCeilingGFLOPS
        self.predictedCeilingGFLOPS = predictedCeilingGFLOPS
        self.bottleneck = bottleneck
    }
}

public struct RooflineMeasurement: Sendable, Equatable {
    public let durationNanoseconds: UInt64
    public let achievedGFLOPS: Double
    public let effectiveBandwidthGBps: Double

    public init(workload: RooflineWorkload, durationNanoseconds: UInt64) throws {
        guard durationNanoseconds > 0 else {
            throw RooflineError.nonpositiveDuration(durationNanoseconds)
        }
        let seconds = Double(durationNanoseconds) / 1_000_000_000
        self.durationNanoseconds = durationNanoseconds
        self.achievedGFLOPS = workload.floatingPointOperations / seconds / 1_000_000_000
        self.effectiveBandwidthGBps = workload.bytesMoved / seconds / 1_000_000_000
    }
}

public struct RooflineReport: Sendable, Equatable {
    public let workload: RooflineWorkload
    public let assumedMachine: RooflineMachine
    public let model: RooflinePrediction
    public let measured: RooflineMeasurement?

    public init(
        workload: RooflineWorkload,
        assumedMachine: RooflineMachine,
        model: RooflinePrediction,
        measured: RooflineMeasurement?
    ) {
        self.workload = workload
        self.assumedMachine = assumedMachine
        self.model = model
        self.measured = measured
    }

    public func rendered() -> String {
        var lines = [
            "Workload: \(workload.name)",
            String(format: "MODEL arithmetic intensity: %.4f FLOP/byte", model.arithmeticIntensity),
            String(
                format: "MODEL ceiling: %.3f GFLOP/s (%@-limited; assumes %.3f GFLOP/s compute, %.3f GB/s memory)",
                model.predictedCeilingGFLOPS,
                model.bottleneck.rawValue,
                assumedMachine.peakComputeGFLOPS,
                assumedMachine.peakMemoryBandwidthGBps
            ),
        ]
        if let measured {
            lines.append(String(
                format: "MEASURED result: %.3f ms, %.3f GFLOP/s, %.3f effective GB/s",
                Double(measured.durationNanoseconds) / 1_000_000,
                measured.achievedGFLOPS,
                measured.effectiveBandwidthGBps
            ))
            lines.append("MEASURED throughput is an observation of this implementation; the model ceiling is not a measurement.")
        } else {
            lines.append("MEASURED result: not provided")
        }
        return lines.joined(separator: "\n")
    }
}

public enum RooflineModel {
    public static func predict(
        workload: RooflineWorkload,
        machine: RooflineMachine
    ) -> RooflinePrediction {
        let intensity = workload.floatingPointOperations / workload.bytesMoved
        let bandwidthCeiling = intensity * machine.peakMemoryBandwidthGBps
        let predictedCeiling = min(machine.peakComputeGFLOPS, bandwidthCeiling)
        let bottleneck: RooflineBottleneck
        if bandwidthCeiling < machine.peakComputeGFLOPS {
            bottleneck = .memory
        } else if bandwidthCeiling > machine.peakComputeGFLOPS {
            bottleneck = .compute
        } else {
            bottleneck = .balanced
        }
        return RooflinePrediction(
            arithmeticIntensity: intensity,
            bandwidthCeilingGFLOPS: bandwidthCeiling,
            predictedCeilingGFLOPS: predictedCeiling,
            bottleneck: bottleneck
        )
    }
}

public enum RooflineBenchmark {
    public static func measure(
        iterations: Int,
        workload: RooflineWorkload,
        operation: () throws -> Void
    ) throws -> RooflineMeasurement {
        guard iterations > 0 else { throw RooflineError.nonpositiveIterations(iterations) }
        var samples: [UInt64] = []
        samples.reserveCapacity(iterations)
        for _ in 0..<iterations {
            let start = DispatchTime.now().uptimeNanoseconds
            try operation()
            samples.append(DispatchTime.now().uptimeNanoseconds - start)
        }
        samples.sort()
        return try RooflineMeasurement(
            workload: workload,
            durationNanoseconds: samples[samples.count / 2]
        )
    }
}

public typealias RooflinePredictionImplementation = (
    _ workload: RooflineWorkload,
    _ machine: RooflineMachine
) -> RooflinePrediction

public enum P006RooflineJudge {
    public static func evaluate(_ implementation: RooflinePredictionImplementation) -> JudgeReport {
        let inputs: [(String, RooflineWorkload, RooflineMachine, RooflinePrediction)]
        do {
            let machine = try RooflineMachine(
                peakComputeGFLOPS: 1_000,
                peakMemoryBandwidthGBps: 100
            )
            let memoryWork = try RooflineWorkload(
                name: "streaming",
                floatingPointOperations: 2_000,
                bytesMoved: 8_000
            )
            let computeWork = try RooflineWorkload(
                name: "reuse",
                floatingPointOperations: 20_000,
                bytesMoved: 100
            )
            let balancedWork = try RooflineWorkload(
                name: "ridge",
                floatingPointOperations: 1_000,
                bytesMoved: 100
            )
            inputs = [
                ("memory-bound workload", memoryWork, machine, RooflineModel.predict(workload: memoryWork, machine: machine)),
                ("compute-bound workload", computeWork, machine, RooflineModel.predict(workload: computeWork, machine: machine)),
                ("ridge point", balancedWork, machine, RooflineModel.predict(workload: balancedWork, machine: machine)),
            ]
        } catch {
            return JudgeReport(
                passedCaseCount: 0,
                totalCaseCount: 3,
                failures: [JudgeFailure(caseName: "judge setup", message: error.localizedDescription)]
            )
        }

        var failures: [JudgeFailure] = []
        var passed = 0
        for (name, workload, machine, expected) in inputs {
            let actual = implementation(workload, machine)
            if actual.bottleneck == expected.bottleneck,
               approximatelyEqual(actual.arithmeticIntensity, expected.arithmeticIntensity),
               approximatelyEqual(actual.bandwidthCeilingGFLOPS, expected.bandwidthCeilingGFLOPS),
               approximatelyEqual(actual.predictedCeilingGFLOPS, expected.predictedCeilingGFLOPS) {
                passed += 1
            } else {
                failures.append(JudgeFailure(
                    caseName: name,
                    message: "expected \(expected), received \(actual)"
                ))
            }
        }
        return JudgeReport(passedCaseCount: passed, totalCaseCount: inputs.count, failures: failures)
    }

    private static func approximatelyEqual(_ lhs: Double, _ rhs: Double) -> Bool {
        abs(lhs - rhs) <= 1e-10 * max(1, abs(lhs), abs(rhs))
    }
}