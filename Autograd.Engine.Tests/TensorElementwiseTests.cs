using Autograd.Engine.Core;

namespace Autograd.Engine.Tests;

public class TensorElementwiseTests
{
    private const float Delta = 1e-5f;

    [Fact]
    public void MultiplyElementwise_BroadcastsAndComputesExactGradients()
    {
        var left = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        var right = new Tensor([10f, 20f], [2]);

        Tensor output = Tensor.MultiplyElementwise(left, right);
        output.Backward();

        Assert.Equal([10f, 40f, 30f, 80f], output.GetData());
        Assert.Equal([10f, 20f, 10f, 20f], left.GetGradients());
        Assert.Equal([4f, 6f], right.GetGradients());
    }

    [Fact]
    public void Subtract_BroadcastsAndAccumulatesGradients()
    {
        var left = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        var right = new Tensor([1f, 2f], [2]);

        Tensor output = left - right;
        output.Backward();

        Assert.Equal([0f, 0f, 2f, 2f], output.GetData());
        Assert.All(left.GetGradients(), gradient => Assert.Equal(1f, gradient, Delta));
        Assert.Equal([-2f, -2f], right.GetGradients());
    }

    [Fact]
    public void ScalarMultiplyAndDivide_PropagateGradient()
    {
        var input = new Tensor([2f, 4f], [2]);

        Tensor output = input * 3f / 2f;
        output.Backward();

        Assert.Equal([3f, 6f], output.GetData());
        Assert.All(input.GetGradients(), gradient => Assert.Equal(1.5f, gradient, Delta));
    }

    [Fact]
    public void Abs_BackwardUsesZeroDerivativeAtZero()
    {
        var input = new Tensor([-2f, 0f, 3f], [3]);

        Tensor.Abs(input).Backward();

        Assert.Equal([-1f, 0f, 1f], input.GetGradients());
    }

    [Fact]
    public void Square_BackwardMatchesDerivative()
    {
        var input = new Tensor([2f, -3f], [2]);

        Tensor.Square(input).Backward();

        Assert.Equal([4f, -6f], input.GetGradients());
    }

    [Fact]
    public void Log_BackwardMatchesDerivative()
    {
        var input = new Tensor([1f, 2f, 4f], [3]);

        Tensor.Log(input).Backward();

        Assert.Equal(1f, input.GetGradients()[0], Delta);
        Assert.Equal(0.5f, input.GetGradients()[1], Delta);
        Assert.Equal(0.25f, input.GetGradients()[2], Delta);
    }
}
