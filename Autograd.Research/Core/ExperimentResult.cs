namespace Autograd.Research.Core;

public sealed record ExperimentResult(
    string ExperimentId,
    IReadOnlyList<ModelRunResult> Runs,
    IReadOnlyList<ModelSummary> Summary);

public sealed record ModelRunResult(
    string Model,
    int ModelSeed,
    int ParameterCount,
    float ValidationMse,
    float TestMse,
    long ElapsedMilliseconds,
    IReadOnlyList<LossCheckpoint> History);

public sealed record ModelSummary(
    string Model,
    int ParameterCount,
    double MeanTestMse,
    double TestMseStandardDeviation,
    double MeanElapsedMilliseconds);

public sealed record LossCheckpoint(int Epoch, float ValidationMse);
