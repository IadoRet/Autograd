using Autograd.Engine.Core;
using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Tests;

public class TensorReductionTests
{
    private const float Delta = 1e-5f;

    [Fact]
    public void Sum_AllElements_ReturnsScalarAndPropagatesGradient()
    {
        var input = new Tensor([1f, 2f, 3f, 4f], [2, 2]);

        Tensor output = Tensor.Sum(input);
        output.Backward();

        Assert.Equal([10f], output.GetData());
        Assert.Equal([1], output.GetShape());
        Assert.All(input.GetGradients(), gradient => Assert.Equal(1f, gradient, Delta));
    }

    [Fact]
    public void Sum_Axis_ReturnsExpectedShapeAndValues()
    {
        var input = new Tensor([1f, 2f, 3f, 4f, 5f, 6f], [2, 3]);

        Tensor output = Tensor.Sum(input, axis: 1);
        output.Backward();

        Assert.Equal([6f, 15f], output.GetData());
        Assert.Equal([2], output.GetShape());
        Assert.All(input.GetGradients(), gradient => Assert.Equal(1f, gradient, Delta));
    }

    [Fact]
    public void Sum_NegativeAxisCanKeepDimension()
    {
        var input = new Tensor([1f, 2f, 3f, 4f], [2, 2]);

        Tensor output = Tensor.Sum(input, axis: -1, keepDimension: true);

        Assert.Equal([3f, 7f], output.GetData());
        Assert.Equal([2, 1], output.GetShape());
    }

    [Fact]
    public void Mean_AllElements_PropagatesScaledGradient()
    {
        var input = new Tensor([1f, 2f, 3f, 4f], [2, 2]);

        Tensor output = Tensor.Mean(input);
        output.Backward();

        Assert.Equal(2.5f, output.GetData()[0], Delta);
        Assert.All(input.GetGradients(), gradient => Assert.Equal(0.25f, gradient, Delta));
    }

    [Fact]
    public void Mean_EmptyTensor_ThrowsTensorDimensionException()
    {
        Assert.Throws<TensorDimensionException>(() => Tensor.Mean(Tensor.Empty));
    }

    [Fact]
    public void Reshape_PreservesOrderAndGradientMapping()
    {
        var input = new Tensor([1f, 2f, 3f, 4f], [4]);

        Tensor output = Tensor.Reshape(input, 2, 2);
        output.Backward();

        Assert.Equal([2, 2], output.GetShape());
        Assert.Equal([1f, 2f, 3f, 4f], output.GetData());
        Assert.All(input.GetGradients(), gradient => Assert.Equal(1f, gradient, Delta));
    }

    [Fact]
    public void Sum_InvalidAxis_ThrowsArgumentOutOfRangeException()
    {
        var input = new Tensor([1f, 2f], [2]);

        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor.Sum(input, 1));
    }
}
