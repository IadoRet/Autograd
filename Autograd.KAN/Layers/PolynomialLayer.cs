using Autograd.Engine.Core;
using Autograd.Engine.Enums;
using Autograd.KAN.Abstractions;

namespace Autograd.KAN.Layers;

/// <summary>
/// KAN layer with configurable basis functions.
/// </summary>
public class PolynomialLayer : IKANLayer
{
    private readonly int[] _degrees;
    private readonly BasisType _basis;

    /// <summary>
    /// Basis coefficients. Shape: [inputSize * basisSize, outputSize]
    /// </summary>
    private readonly Tensor _c;

    /// <summary>
    /// Biases. Shape: [1, outputSize]
    /// </summary>
    private readonly Tensor _b;

    /// <summary>
    /// Output size.
    /// </summary>
    private readonly int _outputSize;

    public PolynomialLayer(int inputSize, int outputSize, BasisType basis, int[] degrees, Random random)
    {
        ArgumentNullException.ThrowIfNull(degrees);

        if (degrees.Length == 0)
            throw new ArgumentException("At least one basis degree must be provided.", nameof(degrees));

        if (degrees.Any(t => t < 0))
            throw new ArgumentOutOfRangeException(nameof(degrees), "Basis degrees must be non-negative.");

        _degrees = degrees.ToArray();
        _basis = basis;
        _outputSize = outputSize;

        int basisSize = inputSize * degrees.Length;
        (_c, _b) = CreateParameters(outputSize, random, basisSize);
    }

    private static (Tensor c, Tensor b) CreateParameters(int outputSize, Random random, int basisSize)
    {
        int size = basisSize * outputSize;
        float[] cData = new float[size];
        float[] bData = new float[outputSize];

        float limit = MathF.Sqrt(6f / (basisSize + outputSize));
        for (int i = 0; i < size; i++)
            cData[i] = (random.NextSingle() * 2f - 1f) * limit;

        return (new Tensor(cData, [basisSize, outputSize]), new Tensor(bData, [1, outputSize]));
    }

    public int GetOutputSize() => _outputSize;

    /// <summary>
    /// Forward pass.
    /// </summary>
    public Tensor Forward(Tensor input)
    {
        Tensor basis = _basis switch
        {
            BasisType.Polynomial => Tensor.PolynomialBasis(input, _degrees),
            BasisType.Chebyshev => Tensor.ChebyshevBasis(input, _degrees),
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
