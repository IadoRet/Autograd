using Autograd.Engine.Core;
using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Tests;

public class TensorMseTests
{
    private const float Delta = 1e-5f;

    [Fact]
    public void MSE_IdenticalTensors_ReturnsZero()
    {
        var p  = new Tensor([1f, 2f, 3f], [3]);
        var gt = new Tensor([1f, 2f, 3f], [3]);

        float loss = Tensor.MSE(p, gt).GetData()[0];

        Assert.Equal(0f, loss, Delta);
    }

    // p=[2,4], gt=[1,3]: ((2-1)^2 + (4-3)^2) / 2 = 1
    [Fact]
    public void MSE_KnownValues_CorrectLoss()
    {
        var p  = new Tensor([2f, 4f], [2]);
        var gt = new Tensor([1f, 3f], [2]);

        float loss = Tensor.MSE(p, gt).GetData()[0];

        Assert.Equal(1f, loss, Delta);
    }

    [Fact]
    public void MSE_OutputIsScalar_ShapeOne()
    {
        var p  = new Tensor([1f, 2f, 3f], [3]);
        var gt = new Tensor([0f, 0f, 0f], [3]);

        float[] data = Tensor.MSE(p, gt).GetData();

        Assert.Single(data);
    }

    [Fact]
    public void MSE_SizeMismatch_ThrowsTensorDimensionException()
    {
        var p  = new Tensor([1f, 2f], [2]);
        var gt = new Tensor([1f, 2f, 3f], [3]);

        Assert.Throws<TensorDimensionException>(() => Tensor.MSE(p, gt));
    }
}
