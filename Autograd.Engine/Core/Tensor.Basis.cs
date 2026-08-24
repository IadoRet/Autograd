using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Core;

public partial class Tensor
{
    /// <summary>
    /// Expands every feature into the selected polynomial degrees.
    /// Input shape: [batch, features]. Output shape: [batch, features * degrees.Length].
    /// </summary>
    public static Tensor PolynomialBasis(Tensor input, int[] degrees)
    {
        ArgumentNullException.ThrowIfNull(input);
        ValidateDegrees(degrees, "Polynomial");

        if (input._shape.Length != 2)
            throw new TensorDimensionException("Polynomial basis requires a 2D tensor shaped [batch, features].");

        int batches = input._shape[0];
        int features = input._shape[1];
        int basisSize = degrees.Length;
        int outputFeatures = features * basisSize;
        float[] data = new float[batches * outputFeatures];

        for (int batch = 0; batch < batches; batch++)
        {
            int inputBase = batch * features;
            int outputBase = batch * outputFeatures;

            for (int feature = 0; feature < features; feature++)
            {
                float value = input._data[inputBase + feature];
                int basisBase = outputBase + feature * basisSize;

                for (int i = 0; i < basisSize; i++)
                    data[basisBase + i] = MathF.Pow(value, degrees[i]);
            }
        }

        Tensor output = CreateOperation(data, [batches, outputFeatures], input);
        output._backward = () =>
        {
            for (int batch = 0; batch < batches; batch++)
            {
                int inputBase = batch * features;
                int outputBase = batch * outputFeatures;

                for (int feature = 0; feature < features; feature++)
                {
                    float value = input._data[inputBase + feature];
                    float gradient = 0f;
                    int basisBase = outputBase + feature * basisSize;

                    for (int i = 0; i < basisSize; i++)
                    {
                        int degree = degrees[i];
                        if (degree != 0)
                        {
                            gradient += output._gradients[basisBase + i]
                                        * degree
                                        * MathF.Pow(value, degree - 1);
                        }
                    }

                    input._gradients[inputBase + feature] += gradient;
                }
            }
        };

        return output;
    }

    /// <summary>
    /// Expands every feature into the selected Chebyshev polynomial degrees.
    /// Input shape: [batch, features]. Output shape: [batch, features * degrees.Length].
    /// </summary>
    public static Tensor ChebyshevBasis(Tensor input, int[] degrees)
    {
        ArgumentNullException.ThrowIfNull(input);
        ValidateDegrees(degrees, "Chebyshev");

        if (input._shape.Length != 2)
            throw new TensorDimensionException("Chebyshev basis requires a 2D tensor shaped [batch, features].");

        int maxDegree = degrees.Max();
        int batches = input._shape[0];
        int features = input._shape[1];
        int basisSize = degrees.Length;
        int outputFeatures = features * basisSize;
        float[] data = new float[batches * outputFeatures];
        int[] degreeOrder = Enumerable.Range(0, basisSize).OrderBy(index => degrees[index]).ToArray();

        for (int batch = 0; batch < batches; batch++)
        {
            int inputBase = batch * features;
            int outputBase = batch * outputFeatures;

            for (int feature = 0; feature < features; feature++)
            {
                float x = input._data[inputBase + feature];
                int basisBase = outputBase + feature * basisSize;
                float previous = 1f;
                float current = x;
                int nextDegreeIndex = 0;

                for (int degree = 0; degree <= maxDegree; degree++)
                {
                    float value;
                    switch (degree)
                    {
                        case 0:
                            value = 1f;
                            break;
                        case 1:
                            value = current;
                            break;
                        default:
                            value = 2f * x * current - previous;
                            previous = current;
                            current = value;
                            break;
                    }

                    while (nextDegreeIndex < basisSize && degrees[degreeOrder[nextDegreeIndex]] == degree)
                    {
                        data[basisBase + degreeOrder[nextDegreeIndex]] = value;
                        nextDegreeIndex++;
                    }
                }
            }
        }

        Tensor output = CreateOperation(data, [batches, outputFeatures], input);
        output._backward = () =>
        {
            for (int batch = 0; batch < batches; batch++)
            {
                int inputBase = batch * features;
                int outputBase = batch * outputFeatures;

                for (int feature = 0; feature < features; feature++)
                {
                    float x = input._data[inputBase + feature];
                    float gradient = 0f;
                    int basisBase = outputBase + feature * basisSize;
                    float previous = 1f;
                    float current = x;
                    float previousDerivative = 0f;
                    float currentDerivative = 1f;
                    int nextDegreeIndex = 0;

                    for (int degree = 0; degree <= maxDegree; degree++)
                    {
                        float derivative;
                        switch (degree)
                        {
                            case 0:
                                derivative = 0f;
                                break;
                            case 1:
                                derivative = currentDerivative;
                                break;
                            default:
                            {
                                float next = 2f * x * current - previous;
                                derivative = 2f * current + 2f * x * currentDerivative - previousDerivative;
                                previous = current;
                                current = next;
                                previousDerivative = currentDerivative;
                                currentDerivative = derivative;
                                break;
                            }
                        }

                        while (nextDegreeIndex < basisSize && degrees[degreeOrder[nextDegreeIndex]] == degree)
                        {
                            gradient += output._gradients[basisBase + degreeOrder[nextDegreeIndex]] * derivative;
                            nextDegreeIndex++;
                        }
                    }

                    input._gradients[inputBase + feature] += gradient;
                }
            }
        };

        return output;
    }

    /// <summary>
    /// Expands every feature into a B-spline basis on a uniform extended knot grid.
    /// </summary>
    public static Tensor BSplineBasis(
        Tensor input,
        int gridSize,
        int splineOrder,
        float gridMin,
        float gridMax)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (gridSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(gridSize), "B-spline grid size must be positive.");
        if (splineOrder < 0)
            throw new ArgumentOutOfRangeException(nameof(splineOrder), "B-spline order must be non-negative.");
        if (gridMax <= gridMin)
            throw new ArgumentOutOfRangeException(nameof(gridMax), "B-spline grid max must be greater than grid min.");
        if (input._shape.Length != 2)
            throw new TensorDimensionException("B-spline basis requires a 2D tensor shaped [batch, features].");

        int batches = input._shape[0];
        int features = input._shape[1];
        int basisSize = gridSize + splineOrder;
        int outputFeatures = features * basisSize;
        float[] data = new float[batches * outputFeatures];
        float[] knots = CreateUniformBSplineKnots(gridSize, splineOrder, gridMin, gridMax);
        float[] basis = new float[knots.Length - 1];

        for (int batch = 0; batch < batches; batch++)
        {
            int inputBase = batch * features;
            int outputBase = batch * outputFeatures;

            for (int feature = 0; feature < features; feature++)
            {
                int basisBase = outputBase + feature * basisSize;
                FillBSplineBasis(input._data[inputBase + feature], knots, splineOrder, basis);

                for (int i = 0; i < basisSize; i++)
                    data[basisBase + i] = basis[i];
            }
        }

        Tensor output = CreateOperation(data, [batches, outputFeatures], input);
        output._backward = () =>
        {
            if (splineOrder == 0)
                return;

            float[] lowerBasis = new float[knots.Length - 1];
            for (int batch = 0; batch < batches; batch++)
            {
                int inputBase = batch * features;
                int outputBase = batch * outputFeatures;

                for (int feature = 0; feature < features; feature++)
                {
                    float x = input._data[inputBase + feature];
                    float gradient = 0f;
                    int basisBase = outputBase + feature * basisSize;
                    FillBSplineBasis(x, knots, splineOrder - 1, lowerBasis);

                    for (int i = 0; i < basisSize; i++)
                    {
                        float derivative = BSplineDerivative(knots, lowerBasis, splineOrder, i);
                        gradient += output._gradients[basisBase + i] * derivative;
                    }

                    input._gradients[inputBase + feature] += gradient;
                }
            }
        };

        return output;
    }

    private static void ValidateDegrees(int[] degrees, string basisName)
    {
        ArgumentNullException.ThrowIfNull(degrees);
        if (degrees.Length == 0)
            throw new ArgumentException($"At least one {basisName} degree must be provided.", nameof(degrees));
        if (degrees.Any(degree => degree < 0))
            throw new ArgumentOutOfRangeException(nameof(degrees), $"{basisName} degrees must be non-negative.");
    }

    private static float[] CreateUniformBSplineKnots(
        int gridSize,
        int splineOrder,
        float gridMin,
        float gridMax)
    {
        int knotCount = gridSize + 2 * splineOrder + 1;
        float step = (gridMax - gridMin) / gridSize;
        float[] knots = new float[knotCount];

        for (int i = 0; i < knotCount; i++)
            knots[i] = gridMin + (i - splineOrder) * step;

        return knots;
    }

    private static void FillBSplineBasis(float x, float[] knots, int splineDegree, float[] basis)
    {
        int intervalCount = knots.Length - 1;
        Array.Clear(basis, 0, intervalCount);

        for (int i = 0; i < intervalCount; i++)
        {
            if (IsInsideKnotInterval(x, knots[i], knots[i + 1], i, intervalCount))
                basis[i] = 1f;
        }

        for (int degree = 1; degree <= splineDegree; degree++)
        {
            int basisCount = intervalCount - degree;
            for (int i = 0; i < basisCount; i++)
            {
                float leftDenominator = knots[i + degree] - knots[i];
                float rightDenominator = knots[i + degree + 1] - knots[i + 1];
                float left = leftDenominator == 0f ? 0f : (x - knots[i]) / leftDenominator * basis[i];
                float right = rightDenominator == 0f ? 0f : (knots[i + degree + 1] - x) / rightDenominator * basis[i + 1];
                basis[i] = left + right;
            }
        }
    }

    private static bool IsInsideKnotInterval(
        float x,
        float left,
        float right,
        int intervalIndex,
        int intervalCount)
    {
        return x >= left && x < right || intervalIndex == intervalCount - 1 && x == right;
    }

    private static float BSplineDerivative(
        float[] knots,
        float[] lowerBasis,
        int splineOrder,
        int basisIndex)
    {
        float leftDenominator = knots[basisIndex + splineOrder] - knots[basisIndex];
        float rightDenominator = knots[basisIndex + splineOrder + 1] - knots[basisIndex + 1];
        float left = leftDenominator == 0f ? 0f : splineOrder / leftDenominator * lowerBasis[basisIndex];
        float right = rightDenominator == 0f ? 0f : splineOrder / rightDenominator * lowerBasis[basisIndex + 1];
        return left - right;
    }
}
