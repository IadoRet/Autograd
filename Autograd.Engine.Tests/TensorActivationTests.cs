using Autograd.Engine.Core;

namespace Autograd.Engine.Tests;

public class TensorActivationTests
{
    private const float Delta = 1e-5f;

    [Fact]
    public void ReLU_NegativeValues_BecomeZero()
    {
        var t = new Tensor([-3f, -1f, -0.001f], [3]);

        float[] result = Tensor.ReLU(t).GetData();

        Assert.All(result, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void ReLU_PositiveValues_Unchanged()
    {
        var t = new Tensor([0.5f, 1f, 100f], [3]);

        float[] result = Tensor.ReLU(t).GetData();

        Assert.Equal(0.5f,  result[0], Delta);
        Assert.Equal(1f,    result[1], Delta);
        Assert.Equal(100f,  result[2], Delta);
    }

    [Fact]
    public void ReLU_Zero_RemainsZero()
    {
        var t = new Tensor([0f], [1]);

        float[] result = Tensor.ReLU(t).GetData();

        Assert.Equal(0f, result[0]);
    }

    [Fact]
    public void TanH_Zero_ReturnsZero()
    {
        var t = new Tensor([0f], [1]);

        float[] result = Tensor.TanH(t).GetData();

        Assert.Equal(0f, result[0], Delta);
    }

    [Fact]
    public void TanH_AppliesCorrectValues()
    {
        float[] input = [-2f, -1f, 0f, 1f, 2f];
        var t = new Tensor(input, [5]);

        float[] result = Tensor.TanH(t).GetData();

        for (int i = 0; i < input.Length; i++)
            Assert.Equal(MathF.Tanh(input[i]), result[i], Delta);
    }

    [Fact]
    public void TanH_OutputBoundedBetweenMinusOneAndOne()
    {
        var t = new Tensor([-100f, 0f, 100f], [3]);

        float[] result = Tensor.TanH(t).GetData();

        Assert.All(result, v =>
        {
            Assert.True(v >= -1f);
            Assert.True(v <= 1f);
        });
    }
}
