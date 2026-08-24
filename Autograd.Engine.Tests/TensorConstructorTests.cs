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

    [Fact]
    public void Constructor_CopiesInputArrays()
    {
        float[] data = [1f, 2f];
        int[] shape = [2];
        var tensor = new Tensor(data, shape);

        data[0] = 99f;
        shape[0] = 1;

        Assert.Equal([1f, 2f], tensor.GetData());
        Assert.Equal([2], tensor.GetShape());
    }

    [Fact]
    public void Constructor_DataLengthDoesNotMatchShape_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new Tensor([1f], [2]));
    }

    [Fact]
    public void Constructor_NegativeDimension_ThrowsArgumentOutOfRangeException()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Tensor([], [-1]));
    }

    [Fact]
    public void Constructor_ShapeProductOverflow_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() => new Tensor([], [int.MaxValue, int.MaxValue, 3]));
    }

    [Fact]
    public void Constructor_EmptyShapeCreatesScalar()
    {
        var tensor = new Tensor([3f], []);

        Assert.Empty(tensor.GetShape());
        Assert.Equal([3f], tensor.GetData());
    }

    [Fact]
    public void Empty_ReturnsConsistentZeroLengthTensor()
    {
        Tensor tensor = Tensor.Empty;

        Assert.Equal([0], tensor.GetShape());
        Assert.Empty(tensor.GetData());
    }
}
