using System.Diagnostics;
using Autograd.Engine.Core;
using Autograd.Engine.Enums;

namespace Autograd.MLP;

/// <summary>
/// Multi-layered perceptron
/// </summary>
// ReSharper disable once InconsistentNaming
public class MLP
{
    private readonly int _inputSize;
    private readonly LinkedList<Layer> _layers;
    private readonly Random _random;

    public int ParameterCount => _layers.Sum(layer => layer.ParameterCount);

    private MLP(int inputSize, Random random)
    {
        _inputSize = inputSize;
        _random = random;
        _layers = [];
    }

    public static MLP Create(int inputSize)
    {
        return new MLP(inputSize, new Random());
    }

    public static MLP Create(int inputSize, int seed)
    {
        return new MLP(inputSize, new Random(seed));
    }

    public MLP WithLayer(int outputSize, ActivationType activation)
    {
        AddLayer(outputSize, activation);
        
        return this;
    }

    public MLP WithOutput(int outputSize)
    {
        AddLayer(outputSize);

        return this;
    }

    private void AddLayer(int outputSize, ActivationType? activation = null)
    {
        int previousOutputSize = _layers.Last == null ? _inputSize : _layers.Last.ValueRef.OutputSize;
        _layers.AddLast(new Layer(previousOutputSize, outputSize, _random, activation));
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
