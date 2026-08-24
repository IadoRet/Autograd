using Autograd.Engine.Core;
using Autograd.KAN.Abstractions;

namespace Autograd.KAN.Layers;

/// <summary>
/// KAN layer with B-spline basis functions.
/// </summary>
public class SplineLayer : IKANLayer
{
    private readonly int _gridSize;
    private readonly int _splineOrder;
    private readonly float _gridMin;
    private readonly float _gridMax;

    /// <summary>
    /// Basis coefficients. Shape: [inputSize * (gridSize + splineOrder), outputSize]
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

    public int ParameterCount => _c.ElementCount + _b.ElementCount;

    public SplineLayer(int inputSize, int outputSize, int gridSize, int splineOrder, float gridMin, float gridMax, Random random)
    {
        ArgumentNullException.ThrowIfNull(random);

        if (gridSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(gridSize), "B-spline grid size must be positive.");

        if (splineOrder < 0)
            throw new ArgumentOutOfRangeException(nameof(splineOrder), "B-spline order must be non-negative.");

        if (gridMax <= gridMin)
            throw new ArgumentOutOfRangeException(nameof(gridMax), "B-spline grid max must be greater than grid min.");

        _gridSize = gridSize;
        _splineOrder = splineOrder;
        _gridMin = gridMin;
        _gridMax = gridMax;
        _outputSize = outputSize;

        int basisSize = inputSize * (gridSize + splineOrder);
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
        Tensor basis = Tensor.BSplineBasis(input, _gridSize, _splineOrder, _gridMin, _gridMax);

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
