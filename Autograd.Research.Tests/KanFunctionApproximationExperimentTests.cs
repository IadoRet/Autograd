using Autograd.Research.Core;
using Autograd.Research.Experiments.KanFunctionApproximation;

namespace Autograd.Research.Tests;

public class KanFunctionApproximationExperimentTests
{
    [Fact]
    public void Run_SmokeCompletesAllModelsAndReportsMatchedParameterCounts()
    {
        KanFunctionApproximationOptions options = SmokeOptions();
        var experiment = new KanFunctionApproximationExperiment(options);

        ExperimentResult result = experiment.Run(new ExperimentContext(new StringWriter()));

        Assert.Equal(5, result.Runs.Count);
        Assert.Equal(5, result.Summary.Count);
        Assert.Equal(321, result.Summary.Single(row => row.Model == "Compound KAN").ParameterCount);
        Assert.Equal(321, result.Summary.Single(row => row.Model == "MLP 2-80-1").ParameterCount);
        Assert.All(result.Runs, run => Assert.True(float.IsFinite(run.TestMse)));
    }

    [Fact]
    public void Run_SameOptionsProducesSameMetrics()
    {
        KanFunctionApproximationOptions options = SmokeOptions();
        var firstExperiment = new KanFunctionApproximationExperiment(options);
        var secondExperiment = new KanFunctionApproximationExperiment(options);

        ExperimentResult first = firstExperiment.Run(new ExperimentContext(new StringWriter()));
        ExperimentResult second = secondExperiment.Run(new ExperimentContext(new StringWriter()));

        Assert.Equal(first.Runs.Select(run => run.ValidationMse), second.Runs.Select(run => run.ValidationMse));
        Assert.Equal(first.Runs.Select(run => run.TestMse), second.Runs.Select(run => run.TestMse));
    }

    private static KanFunctionApproximationOptions SmokeOptions()
    {
        return new KanFunctionApproximationOptions(
            Epochs: 2,
            BatchSize: 8,
            ValidationInterval: 1,
            LearningRate: 0.001f,
            DatasetSeed: 10,
            ShuffleSeed: 20,
            DatasetSizes: new DatasetSizes(16, 8, 8),
            ModelSeeds: [30]);
    }
}
