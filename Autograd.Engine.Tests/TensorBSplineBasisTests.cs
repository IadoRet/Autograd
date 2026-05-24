using Autograd.Engine.Core;
using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Tests;

public class TensorBSplineBasisTests
{
    private const float Delta = 1e-5f;

    [Fact]
    public void BSplineBasis_LinearSpline_CorrectResult()
    {
        var input = new Tensor([0f, 0.25f, 0.5f, 0.75f, 1f], [5, 1]);

        Tensor basis = Tensor.BSplineBasis(input, gridSize: 2, splineOrder: 1, gridMin: 0f, gridMax: 1f);

        float[] expected =
        [
            1f, 0f, 0f,
            0.5f, 0.5f, 0f,
            0f, 1f, 0f,
            0f, 0.5f, 0.5f,
            0f, 0f, 1f
        ];

        float[] result = basis.GetData();

        Assert.Equal(expected.Length, result.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], Delta);
    }

    [Fact]
    public void BSplineBasis_CubicSpline_CorrectShape()
    {
        var input = new Tensor(new float[6], [2, 3]);

        Tensor basis = Tensor.BSplineBasis(input, gridSize: 4, splineOrder: 3, gridMin: -1f, gridMax: 1f);

        Assert.Equal([2, 21], basis.GetShape());
    }

    [Fact]
    public void BSplineBasis_CubicSpline_PartitionOfUnityInsideCoreGrid()
    {
        var input = new Tensor([0f, 0.125f, 0.5f, 0.875f, 1f], [5, 1]);

        Tensor basis = Tensor.BSplineBasis(input, gridSize: 4, splineOrder: 3, gridMin: 0f, gridMax: 1f);

        float[] result = basis.GetData();
        int basisSize = 7;

        for (int b = 0; b < 5; b++)
        {
            float sum = 0f;
            for (int i = 0; i < basisSize; i++)
            {
                float value = result[b * basisSize + i];
                Assert.True(value >= -Delta);
                sum += value;
            }

            Assert.Equal(1f, sum, Delta);
        }
    }

    [Fact]
    public void BSplineBasis_LinearSpline_BackwardPropagatesGradientsToInput()
    {
        var input = new Tensor([0.25f, 0.75f], [2, 1]);
        var weights = new Tensor([0f, 1f, 2f], [3, 1]);

        Tensor basis = Tensor.BSplineBasis(input, gridSize: 2, splineOrder: 1, gridMin: 0f, gridMax: 1f);
        Tensor output = basis * weights;

        output.Backward();

        float[] gradients = input.GetGradients();

        Assert.Equal(2f, gradients[0], Delta);
        Assert.Equal(2f, gradients[1], Delta);
    }

    [Fact]
    public void BSplineBasis_CubicSpline_BackwardMatchesFiniteDifference()
    {
        var input = new Tensor([0.37f], [1, 1]);
        var weights = new Tensor([0f, 1f, -2f, 3f, 1.5f, -0.5f, 2f], [7, 1]);

        Tensor basis = Tensor.BSplineBasis(input, gridSize: 4, splineOrder: 3, gridMin: 0f, gridMax: 1f);
        Tensor output = basis * weights;

        output.Backward();

        float actual = input.GetGradients()[0];
        float expected = FiniteDifference(0.37f);

        Assert.Equal(expected, actual, 1e-2f);
    }

    [Fact]
    public void BSplineBasis_Order0Backward_DoesNotPropagateGradientsToInput()
    {
        var input = new Tensor([0.25f, 0.75f], [2, 1]);

        Tensor basis = Tensor.BSplineBasis(input, gridSize: 2, splineOrder: 0, gridMin: 0f, gridMax: 1f);

        basis.Backward();

        Assert.All(input.GetGradients(), g => Assert.Equal(0f, g, Delta));
    }

    [Fact]
    public void BSplineBasis_InvalidGridSize_ThrowsArgumentOutOfRangeException()
    {
        var input = new Tensor([0f], [1, 1]);

        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor.BSplineBasis(input, 0, 3, 0f, 1f));
    }

    [Fact]
    public void BSplineBasis_InvalidSplineOrder_ThrowsArgumentOutOfRangeException()
    {
        var input = new Tensor([0f], [1, 1]);

        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor.BSplineBasis(input, 4, -1, 0f, 1f));
    }

    [Fact]
    public void BSplineBasis_InvalidGridRange_ThrowsArgumentOutOfRangeException()
    {
        var input = new Tensor([0f], [1, 1]);

        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor.BSplineBasis(input, 4, 3, 1f, 1f));
    }

    [Fact]
    public void BSplineBasis_Non2DInput_ThrowsTensorDimensionException()
    {
        var input = new Tensor([1f, 2f, 3f], [3]);

        Assert.Throws<TensorDimensionException>(() => Tensor.BSplineBasis(input, 4, 3, 0f, 1f));
    }

    private static float FiniteDifference(float x)
    {
        const float h = 1e-3f;

        return (WeightedBSpline(x + h) - WeightedBSpline(x - h)) / (2f * h);
    }

    private static float WeightedBSpline(float x)
    {
        float[] weights = [0f, 1f, -2f, 3f, 1.5f, -0.5f, 2f];
        Tensor basis = Tensor.BSplineBasis(new Tensor([x], [1, 1]), gridSize: 4, splineOrder: 3, gridMin: 0f, gridMax: 1f);
        float[] data = basis.GetData();
        float sum = 0f;

        for (int i = 0; i < weights.Length; i++)
            sum += data[i] * weights[i];

        return sum;
    }
}
