using Autograd.Engine.Core;
using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Tests;

public class TensorPolynomialBasisTests
{
    private const float Delta = 1e-5f;

    [Fact]
    public void PolynomialBasis_Degree3_CorrectResult()
    {
        var input = new Tensor([2f, -3f, 0.5f, 4f], [2, 2]);

        Tensor basis = Tensor.PolynomialBasis(input, [0, 1, 2, 3]);

        float[] result = basis.GetData();

        float[] expected =
        [
            1f, 2f, 4f, 8f,
            1f, -3f, 9f, -27f,
            1f, 0.5f, 0.25f, 0.125f,
            1f, 4f, 16f, 64f
        ];

        Assert.Equal(expected.Length, result.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], Delta);
    }

    [Fact]
    public void PolynomialBasis_Degree3_CorrectShape()
    {
        var input = new Tensor(new float[6], [2, 3]);

        Tensor basis = Tensor.PolynomialBasis(input, [0, 1, 2, 3]);

        Assert.Equal([2, 12], basis.GetShape());
    }

    [Fact]
    public void PolynomialBasis_SelectedDegrees_CorrectResult()
    {
        var input = new Tensor([2f, -3f, 0.5f, 4f], [2, 2]);

        Tensor basis = Tensor.PolynomialBasis(input, [0, 2, 4]);

        float[] result = basis.GetData();

        float[] expected =
        [
            1f, 4f, 16f,
            1f, 9f, 81f,
            1f, 0.25f, 0.0625f,
            1f, 16f, 256f
        ];

        Assert.Equal(expected.Length, result.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], result[i], Delta);
    }

    [Fact]
    public void PolynomialBasis_SelectedDegrees_CorrectShape()
    {
        var input = new Tensor(new float[6], [2, 3]);

        Tensor basis = Tensor.PolynomialBasis(input, [1, 3]);

        Assert.Equal([2, 6], basis.GetShape());
    }

    [Fact]
    public void PolynomialBasis_SelectedDegrees_PreservesDegreeOrder()
    {
        var input = new Tensor([2f], [1, 1]);

        Tensor basis = Tensor.PolynomialBasis(input, [4, 0, 2]);

        Assert.Equal([16f, 1f, 4f], basis.GetData());
    }

    [Fact]
    public void PolynomialBasis_Backward_PropagatesGradientsToInput()
    {
        var input = new Tensor([2f, -3f], [1, 2]);

        Tensor basis = Tensor.PolynomialBasis(input, [0, 1, 2, 3]);
        basis.Backward();

        float[] gradients = input.GetGradients();

        Assert.Equal(17f, gradients[0], Delta);
        Assert.Equal(22f, gradients[1], Delta);
    }

    [Fact]
    public void PolynomialBasis_Degree0_Backward_DoesNotPropagateGradientsToInput()
    {
        var input = new Tensor([2f, -3f], [1, 2]);

        Tensor basis = Tensor.PolynomialBasis(input, [0]);
        basis.Backward();

        Assert.All(input.GetGradients(), g => Assert.Equal(0f, g, Delta));
    }

    [Fact]
    public void PolynomialBasis_SelectedDegrees_Backward_PropagatesSelectedGradientsToInput()
    {
        var input = new Tensor([2f, -3f], [1, 2]);

        Tensor basis = Tensor.PolynomialBasis(input, [0, 2, 4]);
        basis.Backward();

        float[] gradients = input.GetGradients();

        Assert.Equal(36f, gradients[0], Delta);
        Assert.Equal(-114f, gradients[1], Delta);
    }

    [Fact]
    public void PolynomialBasis_Degree0_ReturnsConstantTerms()
    {
        var input = new Tensor([2f, -3f, 4f], [1, 3]);

        Tensor basis = Tensor.PolynomialBasis(input, [0]);

        Assert.Equal([1, 3], basis.GetShape());
        Assert.Equal([1f, 1f, 1f], basis.GetData());
    }

    [Fact]
    public void PolynomialBasis_NegativeDegree_ThrowsArgumentOutOfRangeException()
    {
        var input = new Tensor([1f], [1, 1]);

        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor.PolynomialBasis(input, [-1]));
    }

    [Fact]
    public void PolynomialBasis_SelectedNegativeDegree_ThrowsArgumentOutOfRangeException()
    {
        var input = new Tensor([1f], [1, 1]);

        Assert.Throws<ArgumentOutOfRangeException>(() => Tensor.PolynomialBasis(input, [0, -1, 2]));
    }

    [Fact]
    public void PolynomialBasis_EmptySelectedDegrees_ThrowsArgumentException()
    {
        var input = new Tensor([1f], [1, 1]);

        Assert.Throws<ArgumentException>(() => Tensor.PolynomialBasis(input, []));
    }

    [Fact]
    public void PolynomialBasis_Non2DInput_ThrowsTensorDimensionException()
    {
        var input = new Tensor([1f, 2f, 3f], [3]);

        Assert.Throws<TensorDimensionException>(() => Tensor.PolynomialBasis(input, [0, 1, 2]));
    }
}
