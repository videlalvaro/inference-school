import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P006RooflineTests: XCTestCase {
    func testCanonicalSolutionPassesJudge() {
        let report = P006RooflineJudge.evaluate(P006RooflineSolution.predict)
        XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    }

    func testJudgeRejectsZeroPrediction() {
        let report = P006RooflineJudge.evaluate { _, _ in
            RooflinePrediction(
                arithmeticIntensity: 0,
                bandwidthCeilingGFLOPS: 0,
                predictedCeilingGFLOPS: 0,
                bottleneck: .balanced
            )
        }
        XCTAssertFalse(report.isPassing)
    }

    func testPredictionAndMeasurementRemainDistinct() throws {
        let workload = try RooflineWorkload(
            name: "fixture",
            floatingPointOperations: 2_000_000,
            bytesMoved: 4_000_000
        )
        let machine = try RooflineMachine(
            peakComputeGFLOPS: 100,
            peakMemoryBandwidthGBps: 20
        )
        let model = RooflineModel.predict(workload: workload, machine: machine)
        let measured = try RooflineMeasurement(workload: workload, durationNanoseconds: 2_000_000)
        let report = RooflineReport(
            workload: workload,
            assumedMachine: machine,
            model: model,
            measured: measured
        )

        XCTAssertEqual(model.arithmeticIntensity, 0.5, accuracy: 1e-12)
        XCTAssertEqual(model.predictedCeilingGFLOPS, 10, accuracy: 1e-12)
        XCTAssertEqual(model.bottleneck, .memory)
        XCTAssertEqual(measured.achievedGFLOPS, 1, accuracy: 1e-12)
        XCTAssertEqual(measured.effectiveBandwidthGBps, 2, accuracy: 1e-12)
        XCTAssertTrue(report.rendered().contains("MODEL ceiling"))
        XCTAssertTrue(report.rendered().contains("MEASURED result"))
        XCTAssertTrue(report.rendered().contains("model ceiling is not a measurement"))
    }

    func testInvalidModelInputsAreRejected() {
        XCTAssertThrowsError(try RooflineWorkload(name: "bad", floatingPointOperations: -1, bytesMoved: 4))
        XCTAssertThrowsError(try RooflineWorkload(name: "bad", floatingPointOperations: 1, bytesMoved: 0))
        XCTAssertThrowsError(try RooflineWorkload.gemm(m: -1, k: 2, n: 3))
        XCTAssertThrowsError(try RooflineMachine(peakComputeGFLOPS: 0, peakMemoryBandwidthGBps: 1))
        XCTAssertThrowsError(try RooflineMachine(peakComputeGFLOPS: 1, peakMemoryBandwidthGBps: 0))
    }

    func testGEMMWorkloadCountsMinimumAlgorithmicTraffic() throws {
        let workload = try RooflineWorkload.gemm(m: 2, k: 3, n: 4)
        XCTAssertEqual(workload.floatingPointOperations, 48)
        XCTAssertEqual(workload.bytesMoved, Double((6 + 12 + 8) * 4))
    }
}