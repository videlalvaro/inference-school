import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P036WeightContainerTests: XCTestCase {
  func testCanonicalParserPassesCorruptionJudge() {
    let report = P036WeightContainerJudge.evaluate(P036WeightContainerSolution.parse)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
    XCTAssertEqual(report.totalCaseCount, 13)
  }

  func testJudgeRejectsParserThatIgnoresRequiredNames() {
    let report = P036WeightContainerJudge.evaluate { bytes, _ in
      try P036WeightContainerSolution.parse(bytes: bytes, requiredTensorNames: [])
    }
    XCTAssertFalse(report.isPassing)
  }

  func testFixtureEncoderRoundTripsExplicitLittleEndianFloat32() throws {
    let configuration = try makeConfiguration()
    let tensor = try FloatTensor([1, -2.5, 0.125], shape: [3])
    let bytes = try P036WeightFixtureEncoder.encode(
      configuration: configuration,
      tensors: [WeightFixtureTensor(name: "probe", tensor: tensor)])
    let parsed = try P036WeightContainerSolution.parse(
      bytes: bytes, requiredTensorNames: ["probe"])
    XCTAssertEqual(try parsed.tensor(named: "probe"), tensor)
    XCTAssertEqual(parsed.configuration, configuration)
  }

  func testParserAcceptsArchiveWithNoTensorRanges() throws {
    let configuration = try makeConfiguration()
    let bytes = try P036WeightFixtureEncoder.encode(
      configuration: configuration, tensors: [])
    let parsed = try P036WeightContainerSolution.parse(
      bytes: bytes, requiredTensorNames: [])
    XCTAssertTrue(parsed.tensors.isEmpty)
    XCTAssertEqual(parsed.payloadByteCount, 0)
  }

  func testParserRejectsTruncationMagicVersionAndNonFinitePayload() throws {
    XCTAssertThrowsError(try P036WeightContainerSolution.parse(
      bytes: Array(P036WeightContainerContract.magic.prefix(4)), requiredTensorNames: []))

    let configuration = try makeConfiguration()
    let bytes = try P036WeightFixtureEncoder.encode(
      configuration: configuration,
      tensors: [WeightFixtureTensor(
        name: "probe", tensor: FloatTensor([1], shape: [1]))])
    var badMagic = bytes
    badMagic[0] = 0
    XCTAssertThrowsError(try P036WeightContainerSolution.parse(
      bytes: badMagic, requiredTensorNames: []))
    var badVersion = bytes
    badVersion[8] = 9
    XCTAssertThrowsError(try P036WeightContainerSolution.parse(
      bytes: badVersion, requiredTensorNames: []))

    let preamble = try P036WeightContainerContract.validatePreamble(bytes)
    var nonFinite = bytes
    nonFinite[preamble.payloadStart] = 0
    nonFinite[preamble.payloadStart + 1] = 0
    nonFinite[preamble.payloadStart + 2] = 0x80
    nonFinite[preamble.payloadStart + 3] = 0x7f
    XCTAssertThrowsError(try P036WeightContainerSolution.parse(
      bytes: nonFinite, requiredTensorNames: []))
  }

  func testLoaderRejectsModelWeightShapeMismatch() throws {
    let configuration = try makeConfiguration()
    let tensors = try P036DecoderBlockLoader.requiredTensorNames.enumerated().map {
      index, name -> WeightFixtureTensor in
      let correctShapes = [
        [4], [4, 4], [2, 4], [2, 4], [4, 4], [4], [6, 4], [6, 4], [4, 6],
      ]
      let shape = index == 1 ? [3, 4] : correctShapes[index]
      return WeightFixtureTensor(
        name: name,
        tensor: try FloatTensor(Array(repeating: 0.25, count: shape.reduce(1, *)), shape: shape))
    }
    let bytes = try P036WeightFixtureEncoder.encode(
      configuration: configuration, tensors: tensors)
    let parsed = try P036WeightContainerSolution.parse(
      bytes: bytes,
      requiredTensorNames: P036DecoderBlockLoader.requiredTensorNames)
    XCTAssertThrowsError(try P036DecoderBlockLoader.load(parsed))
  }

  private func makeConfiguration() throws -> DecoderConfiguration {
    try DecoderConfiguration(
      modelDimension: 4,
      hiddenDimension: 6,
      queryHeadCount: 2,
      keyValueHeadCount: 1,
      headDimension: 2,
      rotaryDimension: 2,
      rmsNormEpsilon: 1e-5,
      ropeBase: 100)
  }
}