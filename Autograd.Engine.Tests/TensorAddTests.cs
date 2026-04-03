using Autograd.Engine.Core;
using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Tests;

public class TensorAddTests
{
    private const float Delta = 1e-5f;

    [Fact]
    public void Add_SameShape_CorrectResult()
    {
        var a = new Tensor([1f, 2f, 3f], [3]);
        var b = new Tensor([4f, 5f, 6f], [3]);

        float[] result = (a + b).GetData();

        Assert.Equal([5f, 7f, 9f], result);
    }

    // [1 2; 3 4; 5 6] + [10 20] = [11 22; 13 24; 15 26]
    [Fact]
    public void Add_Broadcasting_2D_Plus_1D_CorrectResult()
    {
        var a = new Tensor([1f, 2f, 3f, 4f, 5f, 6f], [3, 2]);
        var b = new Tensor([10f, 20f], [2]);

        float[] result = (a + b).GetData();

        Assert.Equal(6, result.Length);
        Assert.Equal(11f, result[0], Delta);
        Assert.Equal(22f, result[1], Delta);
        Assert.Equal(13f, result[2], Delta);
        Assert.Equal(24f, result[3], Delta);
        Assert.Equal(15f, result[4], Delta);
        Assert.Equal(26f, result[5], Delta);
    }

    // [1; 2; 3] (shape [3,1]) + [10 20] (shape [1,2]) = [11 21; 12 22; 13 23]
    [Fact]
    public void Add_Broadcasting_ColumnPlusRow_CorrectResult()
    {
        var a = new Tensor([1f, 2f, 3f], [3, 1]);
        var b = new Tensor([10f, 20f], [1, 2]);

        float[] result = (a + b).GetData();

        Assert.Equal(6, result.Length);
        Assert.Equal(11f, result[0], Delta);
        Assert.Equal(21f, result[1], Delta);
        Assert.Equal(12f, result[2], Delta);
        Assert.Equal(22f, result[3], Delta);
        Assert.Equal(13f, result[4], Delta);
        Assert.Equal(23f, result[5], Delta);
    }

    // [5,4] + [1,5] — несовместимые размеры по dim 1: 4 ≠ 5
    [Fact]
    public void Add_IncompatibleDimensions_ThrowsTensorDimensionException()
    {
        var a = new Tensor(new float[20], [5, 4]);
        var b = new Tensor(new float[5],  [1, 5]);

        Assert.Throws<TensorDimensionException>(() => _ = a + b);
    }
}
