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
    /// Kernel (filters). Shape: [outChannels, inChannels, kernelSize, kernelSize]
    /// </summary>
    private readonly Tensor _kernel;

    /// <summary>
    /// Biases. Shape: [1, outChannels, 1, 1] (broadcast over batch and spatial axes)
    /// </summary>
    private readonly Tensor _b;

    public ConvolutionLayer(int inChannels, int outChannels, int kernelSize, Random random, ActivationType? activation = null)
    {
        _activation = activation;

        int fanIn = inChannels * kernelSize * kernelSize;
        float stdDev = MathF.Sqrt(2f / fanIn);

        int size = outChannels * inChannels * kernelSize * kernelSize;
        float[] kernelData = new float[size];

        // He initialization
        for (int i = 0; i < size; i++)
        {
            float u1 = 1f - random.NextSingle();
            float u2 = random.NextSingle();
            float z = MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2);
            kernelData[i] = z * stdDev;
        }

        _kernel = new Tensor(kernelData, [outChannels, inChannels, kernelSize, kernelSize]);
        _b = new Tensor(new float[outChannels], [1, outChannels, 1, 1]);
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
