using Autograd.Engine.Core;
using Autograd.Engine.Enums;
using Autograd.KAN.Abstractions;
using Autograd.KAN.Layers;

namespace Autograd.KAN;

/// <summary>
/// Kolmogorov-Arnold Network with configurable basis functions.
/// </summary>
// ReSharper disable once InconsistentNaming
public class KAN
{
    private readonly int _inputSize;
    private readonly List<IKANLayer> _layers;
    private readonly Random _random;

    public int ParameterCount => _layers.Sum(layer => layer.ParameterCount);

    private KAN(int inputSize, Random random)
    {
        _inputSize = inputSize;
        _random = random;
        _layers = [];
    }

    public static KAN Create(int inputSize)
    {
        return new KAN(inputSize, new Random());
    }

    public static KAN Create(int inputSize, int seed)
    {
        return new KAN(inputSize, new Random(seed));
    }

    public KAN WithPolynomialLayer(int outputSize, int[] degrees, BasisType basis = BasisType.Polynomial)
    {
        AddPolynomialLayer(outputSize, degrees, basis);

        return this;
    }

    public KAN WithPolynomialOutput(int outputSize, int[] degrees, BasisType basis = BasisType.Polynomial)
    {
        AddPolynomialLayer(outputSize, degrees, basis);

        return this;
    }

    public KAN WithSplineLayer(int outputSize, int gridSize, int splineOrder, float gridMin = -1f, float gridMax = 1f)
    {
        AddSplineLayer(outputSize, gridSize, splineOrder, gridMin, gridMax);

        return this;
    }

    public KAN WithSplineOutput(int outputSize, int gridSize, int splineOrder, float gridMin = -1f, float gridMax = 1f)
    {
        AddSplineLayer(outputSize, gridSize, splineOrder, gridMin, gridMax);

        return this;
    }

    private void AddPolynomialLayer(int outputSize, int[] degrees, BasisType basis)
    {
        int previousOutputSize = _layers.Count != 0 ? _layers.Last().GetOutputSize() : _inputSize;
        _layers.Add(new PolynomialLayer(previousOutputSize, outputSize, basis, degrees, _random));
    }

    private void AddSplineLayer(int outputSize, int gridSize, int splineOrder, float gridMin, float gridMax)
    {
        int previousOutputSize = _layers.Count != 0 ? _layers.Last().GetOutputSize() : _inputSize;
        _layers.Add(new SplineLayer(previousOutputSize, outputSize, gridSize, splineOrder, gridMin, gridMax, _random));
    }

    public Tensor Forward(Tensor input)
    {
        foreach (IKANLayer layer in _layers)
            input = layer.Forward(input);

        return input;
    }

    public void Zero()
    {
        foreach (IKANLayer layer in _layers)
            layer.Zero();
    }

    public void Adjust(float learningRate)
    {
        foreach (IKANLayer layer in _layers)
            layer.Adjust(learningRate);
    }
}
