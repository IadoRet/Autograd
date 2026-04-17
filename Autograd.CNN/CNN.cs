using Autograd.Engine.Core;
using Autograd.Engine.Enums;

namespace Autograd.CNN;

/// <summary>
/// Convolutional neural network
/// </summary>
// ReSharper disable once InconsistentNaming
public class CNN
{
    private int _prevChannels;
    private readonly LinkedList<ConvolutionLayer> _layers;
    private readonly Random _random = new Random();

    private CNN(int inChannels)
    {
        _prevChannels = inChannels;
        _layers = [];
    }

    public static CNN Create(int inChannels)
    {
        return new CNN(inChannels);
    }

    public CNN WithLayer(int outChannels, int kernelSize, ActivationType activation)
    {
        _layers.AddLast(new ConvolutionLayer(_prevChannels, outChannels, kernelSize, _random, activation));
        _prevChannels = outChannels;

        return this;
    }

    public CNN WithOutput(int outChannels, int kernelSize)
    {
        _layers.AddLast(new ConvolutionLayer(_prevChannels, outChannels, kernelSize, _random));
        _prevChannels = outChannels;

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
