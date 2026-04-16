using Autograd.Engine.Core;
using Autograd.Engine.Enums;

namespace Autograd.CNN;

/// <summary>
/// Convolutional neural network
/// </summary>
// ReSharper disable once InconsistentNaming
public class CNN
{
    private readonly int _channels;
    private LinkedList<ConvolutionLayer> _layers;
    private Random _random = new Random();

    private CNN(int channels)
    {
        _channels = channels;
        _layers = [];
    }

    public static CNN Create(int channels)
    {
        return new CNN(channels);
    }

    public CNN WithLayer(int kernelSize, ActivationType activation)
    {
        _layers.AddLast(new ConvolutionLayer(_channels, kernelSize, _random, activation));

        return this;
    }

    public CNN WithOutput(int kernelSize)
    {
        _layers.AddLast(new ConvolutionLayer(_channels, kernelSize, _random));

        return this;
    }

    public Tensor Forward(Tensor input)
    {
        foreach (ConvolutionLayer layer in _layers)
            input = layer.Forward(input);

        return input;
    }

    public void Zero()
    {
        foreach (ConvolutionLayer layer in _layers)
            layer.Zero();
    }

    public void Adjust(float learningRate)
    {
        foreach (ConvolutionLayer layer in _layers)
            layer.Adjust(learningRate);
    }
}
