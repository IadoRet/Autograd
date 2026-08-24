using Autograd.Research.Core;
using Autograd.Research.Experiments.KanFunctionApproximation;

namespace Autograd.Research.Tests;

public class InteractionRegressionProblemTests
{
    [Fact]
    public void CreateDataset_SameSeedProducesSameSplits()
    {
        var problem = new InteractionRegressionProblem();
        var sizes = new DatasetSizes(16, 8, 8);

        DatasetSplit first = problem.CreateDataset(42, sizes);
        DatasetSplit second = problem.CreateDataset(42, sizes);

        Assert.Equal(first.Train.Inputs, second.Train.Inputs);
        Assert.Equal(first.Validation.Inputs, second.Validation.Inputs);
        Assert.Equal(first.Test.Outputs, second.Test.Outputs);
    }

    [Fact]
    public void CreateDataset_DifferentSeedChangesInputs()
    {
        var problem = new InteractionRegressionProblem();
        var sizes = new DatasetSizes(16, 8, 8);

        DatasetSplit first = problem.CreateDataset(1, sizes);
        DatasetSplit second = problem.CreateDataset(2, sizes);

        Assert.NotEqual(first.Train.Inputs, second.Train.Inputs);
    }

    [Fact]
    public void CreateDataset_CreatesRequestedSplitSizes()
    {
        var problem = new InteractionRegressionProblem();

        DatasetSplit dataset = problem.CreateDataset(1, new DatasetSizes(10, 5, 7));

        Assert.Equal(10, dataset.Train.Count);
        Assert.Equal(5, dataset.Validation.Count);
        Assert.Equal(7, dataset.Test.Count);
    }
}
