using Autograd.Engine.Core;
using Autograd.Engine.Enums;

namespace Autograd.KAN;

/// <summary>
/// KAN layer with configurable basis functions.
/// </summary>
public class Layer
{
    private readonly int _degree;
    private readonly BasisType _basis;

    /// <summary>
    /// Basis coefficients. Shape: [inputSize * (degree + 1), outputSize]
    /// </summary>
    private readonly Tensor _c;

    /// <summary>
    /// Biases. Shape: [1, outputSize]
    /// </summary>
    private readonly Tensor _b;

    /// <summary>
    /// Output size.
    /// </summary>
    public int OutputSize { get; }

    public Layer(int inputSize, int outputSize, int degree, Random random, BasisType basis)
    {
        if (degree < 0)
            throw new ArgumentOutOfRangeException(nameof(degree), "Basis degree must be non-negative.");

        _degree = degree;
        _basis = basis;
        OutputSize = outputSize;

        int basisSize = inputSize * (degree + 1);
        int size = basisSize * outputSize;
        float[] cData = new float[size];
        float[] bData = new float[outputSize];

        float limit = MathF.Sqrt(6f / (basisSize + outputSize));
        for (int i = 0; i < size; i++)
            cData[i] = (random.NextSingle() * 2f - 1f) * limit;

        _c = new Tensor(cData, [basisSize, outputSize]);
        _b = new Tensor(bData, [1, outputSize]);
    }

    /// <summary>
    /// Forward pass.
    /// </summary>
    public Tensor Forward(Tensor input)
    {
        Tensor basis = _basis switch
        {
            BasisType.Polynomial => Tensor.PolynomialBasis(input, _degree),
            BasisType.Chebyshev => Tensor.ChebyshevBasis(input, _degree),
            _ => throw new ArgumentOutOfRangeException(nameof(_basis), _basis, "Unsupported basis type.")
        };

        return basis * _c + _b;
    }

    /// <summary>
    /// Zero out gradients.
    /// </summary>
    public void Zero()
    {
        _c.Zero();
        _b.Zero();
    }

    /// <summary>
    /// Adjust coefficients and biases according to gradients.
    /// </summary>
    public void Adjust(float rate)
    {
        _c.Adjust(rate);
        _b.Adjust(rate);
    }
}
