using Autograd.Engine.Core;
using Autograd.Engine.Enums;

namespace Autograd.KAN;

/// <summary>
/// KAN layer with configurable basis functions.
/// </summary>
public class Layer
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
    public int OutputSize { get; }

    public Layer(int inputSize, int outputSize, BasisType basis, int[] degrees, Random random)
    {
        ArgumentNullException.ThrowIfNull(degrees);

        if (degrees.Length == 0)
            throw new ArgumentException("At least one basis degree must be provided.", nameof(degrees));

        for (int i = 0; i < degrees.Length; i++)
        {
            if (degrees[i] < 0)
                throw new ArgumentOutOfRangeException(nameof(degrees), "Basis degrees must be non-negative.");
        }

        _degrees = degrees.ToArray();
        _basis = basis;
        OutputSize = outputSize;

        int basisSize = inputSize * GetBasisSize(basis, _degrees);
        (_c, _b) = CreateParameters(outputSize, random, basisSize);
    }

    // TODO: remove function after supporting degrees array
    private static int GetBasisSize(BasisType basis, int[] degrees)
    {
        return basis switch
        {
            BasisType.Polynomial => degrees.Length,
            // TODO: support selected Chebyshev degrees instead of expanding to all degrees up to Max().
            BasisType.Chebyshev => degrees.Max() + 1,
            _ => throw new ArgumentOutOfRangeException(nameof(basis), basis, "Unsupported basis type.")
        };
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
