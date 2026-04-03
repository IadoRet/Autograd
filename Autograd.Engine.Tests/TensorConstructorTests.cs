using Autograd.Engine.Core;

namespace Autograd.Engine.Tests;

public class TensorConstructorTests
{
    [Fact]
    public void GetData_ReturnsInitialData()
    {
        var t = new Tensor([1f, 2f, 3f], [3]);

        Assert.Equal([1f, 2f, 3f], t.GetData());
    }

    [Fact]
    public void GetGradients_InitiallyAllZeros()
    {
        var t = new Tensor([1f, 2f, 3f], [3]);

        Assert.All(t.GetGradients(), g => Assert.Equal(0f, g));
    }

    [Fact]
    public void GetData_ReturnsCopy_NotReference()
    {
        var t = new Tensor([1f, 2f, 3f], [3]);

        float[] copy = t.GetData();
        copy[0] = 999f;

        Assert.Equal(1f, t.GetData()[0]);
    }

    [Fact]
    public void GetGradients_ReturnsCopy_NotReference()
    {
        var t = new Tensor([1f, 2f, 3f], [3]);

        float[] copy = t.GetGradients();
        copy[0] = 999f;

        Assert.Equal(0f, t.GetGradients()[0]);
    }
}
