using Autograd.Engine.Core;
using Autograd.Engine.Enums;
using KanNetwork = Autograd.KAN.KAN;
using MlpNetwork = Autograd.MLP.MLP;

namespace Autograd.Research.Experiments.KanFunctionApproximation;

internal interface IRegressionModel
{
    int ParameterCount { get; }

    Tensor Forward(Tensor input);

    void Adjust(float learningRate);

    void Zero();
}

internal sealed class KanRegressionModel(KanNetwork model) : IRegressionModel
{
    public int ParameterCount => model.ParameterCount;

    public Tensor Forward(Tensor input) => model.Forward(input);

    public void Adjust(float learningRate) => model.Adjust(learningRate);

    public void Zero() => model.Zero();
}

internal sealed class MlpRegressionModel(MlpNetwork model) : IRegressionModel
{
    public int ParameterCount => model.ParameterCount;

    public Tensor Forward(Tensor input) => model.Forward(input);

    public void Adjust(float learningRate) => model.Adjust(learningRate);

    public void Zero() => model.Zero();
}

internal sealed record ModelSpecification(string Name, Func<int, IRegressionModel> Create)
{
    public static IReadOnlyList<ModelSpecification> All { get; } =
    [
        new(
            "Polynomial 0..2",
            seed => new KanRegressionModel(
                KanNetwork.Create(2, seed).WithPolynomialOutput(1, [0, 1, 2]))),
        new(
            "Chebyshev 0..2",
            seed => new KanRegressionModel(
                KanNetwork.Create(2, seed).WithPolynomialOutput(1, [0, 1, 2], BasisType.Chebyshev))),
        new(
            "Cubic B-spline",
            seed => new KanRegressionModel(
                KanNetwork.Create(2, seed).WithSplineOutput(
                    1,
                    gridSize: 8,
                    splineOrder: 3,
                    gridMin: -2f,
                    gridMax: 2f))),
        new(
            "Compound KAN",
            seed => new KanRegressionModel(
                KanNetwork.Create(2, seed)
                    .WithSplineLayer(8, gridSize: 12, splineOrder: 5, gridMin: -2f, gridMax: 2f)
                    .WithPolynomialOutput(1, [0, 1, 2, 3, 5], BasisType.Chebyshev))),
        new(
            "MLP 2-80-1",
            seed => new MlpRegressionModel(
                MlpNetwork.Create(2, seed)
                    .WithLayer(80, ActivationType.Tanh)
                    .WithOutput(1)))
    ];
}
