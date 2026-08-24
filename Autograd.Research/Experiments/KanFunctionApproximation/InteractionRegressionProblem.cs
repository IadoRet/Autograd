using Autograd.Research.Abstractions;
using Autograd.Research.Core;

namespace Autograd.Research.Experiments.KanFunctionApproximation;

public sealed class InteractionRegressionProblem : IRegressionProblem
{
    private const float Range = 2f;

    public string Id => "nonlinear-interaction";

    public IReadOnlyList<VariableDescriptor> Inputs { get; } =
    [
        new("x0"),
        new("x1")
    ];

    public IReadOnlyList<VariableDescriptor> Outputs { get; } = [new("y")];

    public DatasetSplit CreateDataset(int seed, DatasetSizes sizes)
    {
        ArgumentNullException.ThrowIfNull(sizes);
        if (sizes.Train <= 0 || sizes.Validation <= 0 || sizes.Test <= 0)
            throw new ArgumentOutOfRangeException(nameof(sizes), "All dataset splits must contain at least one sample.");

        Random random = new(seed);
        RegressionDataset all = Generate(random, sizes.Total);
        int validationStart = sizes.Train;
        int testStart = sizes.Train + sizes.Validation;

        return new DatasetSplit(
            Slice(all, 0, sizes.Train),
            Slice(all, validationStart, sizes.Validation),
            Slice(all, testStart, sizes.Test));
    }

    public static float Evaluate(float x0, float x1)
    {
        return MathF.Tanh(3f * x0 + x1) + 0.5f * MathF.Exp(0.2f * x1 - 0.2f * x0);
    }

    private static RegressionDataset Generate(Random random, int count)
    {
        float[] inputs = new float[count * 2];
        float[] outputs = new float[count];

        for (int i = 0; i < count; i++)
        {
            float x0 = random.NextSingle() * 2f * Range - Range;
            float x1 = random.NextSingle() * 2f * Range - Range;
            inputs[i * 2] = x0;
            inputs[i * 2 + 1] = x1;
            outputs[i] = Evaluate(x0, x1);
        }

        return new RegressionDataset(inputs, outputs, count, InputSize: 2, OutputSize: 1);
    }

    private static RegressionDataset Slice(RegressionDataset source, int start, int count)
    {
        float[] inputs = new float[count * source.InputSize];
        float[] outputs = new float[count * source.OutputSize];
        Array.Copy(source.Inputs, start * source.InputSize, inputs, 0, inputs.Length);
        Array.Copy(source.Outputs, start * source.OutputSize, outputs, 0, outputs.Length);
        return new RegressionDataset(inputs, outputs, count, source.InputSize, source.OutputSize);
    }
}
