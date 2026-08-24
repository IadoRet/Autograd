using Autograd.Research.Core;

namespace Autograd.Research.Abstractions;

public interface IRegressionProblem
{
    string Id { get; }

    IReadOnlyList<VariableDescriptor> Inputs { get; }

    IReadOnlyList<VariableDescriptor> Outputs { get; }

    DatasetSplit CreateDataset(int seed, DatasetSizes sizes);
}
