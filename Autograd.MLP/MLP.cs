using System.Diagnostics;
using Autograd.Engine.Core;

namespace Autograd.MLP;

/// <summary>
/// Multi-layered perceptron
/// </summary>
// ReSharper disable once InconsistentNaming
public class MLP
{
    private readonly int _inputSize;
    private LinkedList<Layer> _layers;
    private Random _random = new Random();

    private MLP(int inputSize)
    {
        _inputSize = inputSize;
        _layers = [];
    }

    public static MLP Create(int inputSize)
    {
        return new MLP(inputSize);
    }

    public MLP WithLayer(int outputSize)
    {
        AddLayer(outputSize, false);
        
        return this;
    }

    public MLP WithOutput(int outputSize)
    {
        AddLayer(outputSize, true);

        return this;
    }

    private void AddLayer(int outputSize, bool final)
    {
        int previousOutputSize = _layers.Last == null ? _inputSize : _layers.Last.ValueRef.OutputSize;
        _layers.AddLast(new Layer(previousOutputSize, outputSize, _random, !final));
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