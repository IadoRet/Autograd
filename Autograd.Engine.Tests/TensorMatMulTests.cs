using Autograd.Engine.Core;
using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Tests;

public class TensorMatMulTests
{
    private const float Delta = 1e-5f;

    // [1 2]   [5 6]   [19 22]
    // [3 4] * [7 8] = [43 50]
    [Fact]
    public void MatMul_2x2_Times_2x2_CorrectResult()
    {
        var a = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        var b = new Tensor([5f, 6f, 7f, 8f], [2, 2]);

        float[] result = (a * b).GetData();

        Assert.Equal(4, result.Length);
        Assert.Equal(19f, result[0], Delta);
        Assert.Equal(22f, result[1], Delta);
        Assert.Equal(43f, result[2], Delta);
        Assert.Equal(50f, result[3], Delta);
    }

    // [1 2 3]   [7  8 ]   [58  64 ]
    // [4 5 6] * [9  10] = [139 154]
    //           [11 12]
    [Fact]
    public void MatMul_2x3_Times_3x2_CorrectResult()
    {
        var a = new Tensor([1f, 2f, 3f, 4f, 5f, 6f], [2, 3]);
        var b = new Tensor([7f, 8f, 9f, 10f, 11f, 12f], [3, 2]);

        float[] result = (a * b).GetData();

        Assert.Equal(4, result.Length);
        Assert.Equal(58f,  result[0], Delta);
        Assert.Equal(64f,  result[1], Delta);
        Assert.Equal(139f, result[2], Delta);
        Assert.Equal(154f, result[3], Delta);
    }

    // Batch [2, 2, 2]: two independent 2x2 matrix multiplications
    [Fact]
    public void MatMul_Batched_2x2x2_CorrectResult()
    {
        // batch 0: [1 0; 0 1] * [5 6; 7 8] = [5 6; 7 8]
        // batch 1: [2 0; 0 2] * [1 0; 0 1] = [2 0; 0 2]
        var a = new Tensor([1f, 0f, 0f, 1f,  2f, 0f, 0f, 2f], [2, 2, 2]);
        var b = new Tensor([5f, 6f, 7f, 8f,  1f, 0f, 0f, 1f], [2, 2, 2]);

        float[] result = (a * b).GetData();

        Assert.Equal(8, result.Length);
        // batch 0
        Assert.Equal(5f, result[0], Delta);
        Assert.Equal(6f, result[1], Delta);
        Assert.Equal(7f, result[2], Delta);
        Assert.Equal(8f, result[3], Delta);
        // batch 1
        Assert.Equal(2f, result[4], Delta);
        Assert.Equal(0f, result[5], Delta);
        Assert.Equal(0f, result[6], Delta);
        Assert.Equal(2f, result[7], Delta);
    }

    [Fact]
    public void MatMul_IncompatibleInnerDimensions_ThrowsTensorDimensionException()
    {
        // [2,3] * [2,3] — inner dims 3 ≠ 2
        var a = new Tensor([1f, 2f, 3f, 4f, 5f, 6f], [2, 3]);
        var b = new Tensor([1f, 2f, 3f, 4f, 5f, 6f], [2, 3]);

        Assert.Throws<TensorDimensionException>(() => _ = a * b);
    }

    [Fact]
    public void MatMul_BatchDimensionMismatch_ThrowsTensorDimensionException()
    {
        var a = new Tensor(new float[8], [2, 2, 2]);
        var b = new Tensor(new float[8], [3, 2, 2]);

        Assert.Throws<TensorDimensionException>(() => _ = a * b);
    }
}
