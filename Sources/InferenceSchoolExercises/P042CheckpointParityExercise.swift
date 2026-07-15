import InferenceSchoolCore

public enum P042CheckpointParityExercise {
  public static func compare(
    _ request: CheckpointParityRequest
  ) throws -> CheckpointParityReport {
    let artifact = try P042CheckpointParityContract.validate(request)

    // TODO: execute the complete mini-model with named captures, compare every
    // capture in artifact order, and report the first boundary outside tolerance.
    return CheckpointParityReport(
      modelFingerprint: request.model.fingerprint,
      artifactProvenance: artifact.provenance,
      comparisons: [],
      firstDivergentCapture: nil,
      referenceSelectedTokenID: artifact.selectedTokenID,
      candidateSelectedTokenID: artifact.selectedTokenID,
      selectedTokenMatches: true,
      isPassing: false)
  }
}