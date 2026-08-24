using Autograd.Research.Core;

namespace Autograd.Research.Experiments.KanFunctionApproximation;

public sealed record KanFunctionApproximationOptions(
    int Epochs,
    int BatchSize,
    int ValidationInterval,
    float LearningRate,
    int DatasetSeed,
    int ShuffleSeed,
    DatasetSizes DatasetSizes,
    IReadOnlyList<int> ModelSeeds)
{
    public static KanFunctionApproximationOptions Default { get; } = new(
        Epochs: 360,
        BatchSize: 32,
        ValidationInterval: 20,
        LearningRate: 0.001f,
        DatasetSeed: 1207,
        ShuffleSeed: 404,
        DatasetSizes: new DatasetSizes(Train: 1024, Validation: 256, Test: 1024),
        ModelSeeds: [0, 1, 2, 3, 4]);
}
