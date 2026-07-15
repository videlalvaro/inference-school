import InferenceSchoolCore
import InferenceSchoolSolutions
import XCTest

final class P042CheckpointParityTests: XCTestCase {
  func testCanonicalSolutionPassesJudge() {
    let report = P042CheckpointParityJudge.evaluate(P042CheckpointParitySolution.compare)
    XCTAssertTrue(report.isPassing, report.failures.map(\.message).joined(separator: "\n"))
  }

  func testArtifactRejectsWrongFingerprintAndMalformedShape() throws {
    let model = try EducationalMiniModelFixture.make()
    let bytes = try P042EducationalReference.artifactBytes(
      model: model,
      tokenIDs: EducationalMiniModelFixture.defaultPrompt)
    XCTAssertThrowsError(try MiniDecoderReferenceArtifactCodec.decode(
      bytes, expectedModelFingerprint: "stale"))
    let invalid = MiniDecoderReferenceArtifact(
      provenance: P042EducationalReference.provenance,
      modelFingerprint: model.fingerprint,
      tokenIDs: [1],
      positionOffset: 0,
      captures: [ReferenceCaptureTensor(name: "x", shape: [2], values: [1])],
      selectedTokenID: 0)
    XCTAssertThrowsError(try MiniDecoderReferenceArtifactCodec.encode(invalid))
  }

  func testFaultsLocalizeBeforeFinalLogits() throws {
    let model = try EducationalMiniModelFixture.make()
    let tokens = EducationalMiniModelFixture.defaultPrompt
    let bytes = try P042EducationalReference.artifactBytes(
      model: model, tokenIDs: tokens, positionOffset: 2)
    let rope = try P042CheckpointParitySolution.compare(CheckpointParityRequest(
      model: model,
      tokenIDs: tokens,
      positionOffset: 2,
      referenceArtifactBytes: bytes,
      fault: .ropePositionOffset))
    let norm = try P042CheckpointParitySolution.compare(CheckpointParityRequest(
      model: model,
      tokenIDs: tokens,
      positionOffset: 2,
      referenceArtifactBytes: bytes,
      fault: .additiveRMSNormGamma))
    XCTAssertEqual(rope.firstDivergentCapture, "layer.0.rope.query")
    XCTAssertEqual(norm.firstDivergentCapture, "layer.0.attention_norm")
    XCTAssertNotEqual(rope.firstDivergentCapture, "logits")
    XCTAssertNotEqual(norm.firstDivergentCapture, "logits")
  }

  func testArtifactSerializationIsDeterministicAndExplicitlyEducational() throws {
    let model = try EducationalMiniModelFixture.make()
    let artifact = try P042EducationalReference.artifact(
      model: model, tokenIDs: EducationalMiniModelFixture.defaultPrompt)
    let first = try MiniDecoderReferenceArtifactCodec.encode(artifact)
    let second = try MiniDecoderReferenceArtifactCodec.encode(artifact)
    XCTAssertEqual(first, second)
    XCTAssertEqual(
      try MiniDecoderReferenceArtifactCodec.decode(first).provenance,
      P042EducationalReference.provenance)
  }
}