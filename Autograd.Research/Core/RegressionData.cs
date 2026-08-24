using Autograd.Engine.Core;

namespace Autograd.Research.Core;

public sealed record VariableDescriptor(string Name, string? Unit = null);

public sealed record DatasetSizes(int Train, int Validation, int Test)
{
    public int Total => Train + Validation + Test;
}

public sealed record RegressionDataset(float[] Inputs, float[] Outputs, int Count, int InputSize, int OutputSize)
{
    public Tensor InputTensor() => new(Inputs, [Count, InputSize]);

    public Tensor OutputTensor() => new(Outputs, [Count, OutputSize]);
}

public sealed record DatasetSplit(
    RegressionDataset Train,
    RegressionDataset Validation,
    RegressionDataset Test);
