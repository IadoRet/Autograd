using Autograd.Engine.Core;
using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Tests;

public class TensorChebyshevBasisTests
{
    private const float Delta = 1e-5f;

    [Fact]
    public void ChebyshevBasis_Degree4_CorrectResult()
    {
        var input = new Tensor([0.5f, -1f, 0f, 1f], [2, 2]);

        Tensor basis = Tensor.ChebyshevBasis(input, degree: 4);

        float[] result = basis.GetData();

        float[] expected =
        [
            1f, 0.5f, -0.5f, -1f, -0.5f,
            1f, -1f, 1f, -1f, 1f,
            1f, 0f, -1f, 0f, 1f,
            1f, 1f, 1f, 1f, 1f
        ];

        Assert.Equal(expected.Length, result.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], Delta);
    }

    [Fact]
    public void ChebyshevBasis_Degree4_CorrectShape()
    {
        var input = new Tensor(new float[6], [2, 3]);

        Tensor basis = Tensor.ChebyshevBasis(input, degree: 4);

        Assert.Equal([2, 15], basis.GetShape());
    }

    [Fact]
    public void ChebyshevBasis_Backward_PropagatesGradientsToInput()
    {
        var input = new Tensor([0.5f, -0.25f], [1, 2]);

        Tensor basis = Tensor.ChebyshevBasis(input, degree: 4);
        basis.Backward();

        float[] gradients = input.GetGradients();

        Assert.Equal(-1f, gradients[0], Delta);
        Assert.Equal(1.25f, gradients[1], Delta);
    }

    [Fact]
    public void ChebyshevBasis_Degree0_Backward_DoesNotPropagateGradientsToInput()
    {
        var input = new Tensor([0.5f, -0.25f], [1, 2]);

        Tensor basis = Tensor.ChebyshevBasis(input, degree: 0);
        basis.Backward();

        Assert.All(input.GetGradients(), g => Assert.Equal(0f, g, Delta));
    }

    [Fact]
    public void ChebyshevBasis_Degree0_ReturnsConstantTerms()
    {
        var input = new Tensor([0.5f, -0.25f, 1f], [1, 3]);

        Tensor basis = Tensor.ChebyshevBasis(input, degree: 0);

        Assert.Equal([1, 3], basis.GetShape());
        Assert.Equal([1f, 1f, 1f], basis.GetData());
    }

    [Fact]
    public void ChebyshevBasis_NegativeDegree_ThrowsArgumentOutOfRangeException()
    {
        var input = new Tensor([1f], [1, 1]);

        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor.ChebyshevBasis(input, degree: -1));
    }

    [Fact]
    public void ChebyshevBasis_Non2DInput_ThrowsTensorDimensionException()
    {
        var input = new Tensor([1f, 2f, 3f], [3]);

        Assert.Throws<TensorDimensionException>(() => Tensor.ChebyshevBasis(input, degree: 2));
    }
}
