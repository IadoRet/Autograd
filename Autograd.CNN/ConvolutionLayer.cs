using Autograd.Engine.Core;
using Autograd.Engine.Enums;

namespace Autograd.CNN;

/// <summary>
/// Convolution layer
/// </summary>
public class ConvolutionLayer
{
    /// <summary>
    /// Activation function
    /// </summary>
    private readonly ActivationType? _activation;

    /// <summary>
    /// Kernel (filters)
    /// </summary>
    private readonly Tensor _kernel;

    /// <summary>
    /// Biases
    /// </summary>
    private readonly Tensor _b;

    public ConvolutionLayer(int channels, int kernelSize, Random random, ActivationType? activation = null)
    {
        _activation = activation;

        int size = channels * kernelSize * kernelSize;
        float[] kernelData = new float[size];

        for (int i = 0; i < size; i++)
            kernelData[i] = random.NextSingle() * 2 - 1;

        // kernel shape: [1, channels, kernelSize, kernelSize]
        _kernel = new Tensor(kernelData, [1, channels, kernelSize, kernelSize]);
        _b = new Tensor(new float[channels], [1, channels, 1, 1]);
    }

    /// <summary>
    /// Forward pass
    /// </summary>
    public Tensor Forward(Tensor input)
    {
        Tensor pass = Tensor.Convolution(input, _kernel) + _b;

        return _activation switch
        {
            ActivationType.ReLU => Tensor.ReLU(pass),
            ActivationType.Tanh => Tensor.TanH(pass),
            _ => pass
        };
    }

    /// <summary>
    /// Zero out gradients
    /// </summary>
    public void Zero()
    {
        _kernel.Zero();
        _b.Zero();
    }

    /// <summary>
    /// Adjust weights and biases according to gradients
    /// </summary>
    public void Adjust(float rate)
    {
        _kernel.Adjust(rate);
        _b.Adjust(rate);
    }
}