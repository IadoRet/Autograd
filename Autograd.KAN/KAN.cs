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
    private readonly Random _random;

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

    public KAN WithLayer(int outputSize, int[] degrees, BasisType basis = BasisType.Polynomial)
    {
        AddLayer(outputSize, degrees, basis);

        return this;
    }

    public KAN WithOutput(int outputSize, int[] degrees, BasisType basis = BasisType.Polynomial)
    {
        AddLayer(outputSize, degrees, basis);

        return this;
    }

    private void AddLayer(int outputSize, int[] degrees, BasisType basis)
    {
        int previousOutputSize = _layers.Last == null ? _inputSize : _layers.Last.ValueRef.OutputSize;
        _layers.AddLast(new Layer(previousOutputSize, outputSize, basis, degrees, _random));
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
