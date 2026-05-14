using Autograd.Engine.Core;
using Autograd.Engine.Enums;

namespace Autograd.KAN;

/// <summary>
/// Kolmogorov-Arnold Network with configurable basis functions.
/// </summary>
// ReSharper disable once InconsistentNaming
public class KAN
{
    private readonly int _inputSize;
    private readonly LinkedList<Layer> _layers;
    private readonly Random _random = new Random();

    private KAN(int inputSize)
    {
        _inputSize = inputSize;
        _layers = [];
    }

    public static KAN Create(int inputSize)
    {
        return new KAN(inputSize);
    }

    public KAN WithLayer(int outputSize, int degree, BasisType basis = BasisType.Polynomial)
    {
        AddLayer(outputSize, degree, basis);

        return this;
    }

    public KAN WithOutput(int outputSize, int degree, BasisType basis = BasisType.Polynomial)
    {
        AddLayer(outputSize, degree, basis);

        return this;
    }

    private void AddLayer(int outputSize, int degree, BasisType basis)
    {
        int previousOutputSize = _layers.Last == null ? _inputSize : _layers.Last.ValueRef.OutputSize;
        _layers.AddLast(new Layer(previousOutputSize, outputSize, degree, _random, basis));
    }

    public Tensor Forward(Tensor input)
    {
        foreach (Layer layer in _layers)
            input = layer.Forward(input);

        return input;
    }

    public void Zero()
    {
        foreach (Layer layer in _layers)
            layer.Zero();
    }

    public void Adjust(float learningRate)
    {
        foreach (Layer layer in _layers)
            layer.Adjust(learningRate);
    }
}
