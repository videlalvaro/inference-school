import InferenceSchoolCore

public enum P006RooflineExercise {
    public static func predict(
        workload: RooflineWorkload,
        machine: RooflineMachine
    ) -> RooflinePrediction {
        // TODO: Compute intensity, the bandwidth roof, the lower ceiling, and its limiter.
        RooflinePrediction(
            arithmeticIntensity: 0,
            bandwidthCeilingGFLOPS: 0,
            predictedCeilingGFLOPS: 0,
            bottleneck: .balanced
        )
    }
}