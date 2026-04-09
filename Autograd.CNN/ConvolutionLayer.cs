using Autograd.Engine.Core;

namespace Autograd.CNN;

public class ConvolutionLayer
{
    private readonly Tensor _kernel;
    private readonly Tensor _b;

    public Tensor Forward(Tensor input) => Tensor.Convolution(input, _kernel) + _b;

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