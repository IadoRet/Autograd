using Autograd.Engine.Core;

namespace Autograd.Engine.Tests;

public class TensorBackpropTests
{
    private const float Delta = 1e-5f;

    // c = a + b; c.Backward() → grad_a = grad_b = 1
    [Fact]
    public void Backward_Add_PropagatesGradientsToLeaves()
    {
        var a = new Tensor([1f, 2f, 3f], [3]);
        var b = new Tensor([4f, 5f, 6f], [3]);
        var c = a + b;

        c.Backward();

        Assert.All(a.GetGradients(), g => Assert.Equal(1f, g, Delta));
        Assert.All(b.GetGradients(), g => Assert.Equal(1f, g, Delta));
    }

    // b shape [1], a shape [3] → grad_b accumulates 3 times
    [Fact]
    public void Backward_Add_Broadcasting_AccumulatesGradients()
    {
        var a = new Tensor([1f, 2f, 3f], [3]);
        var b = new Tensor([10f], [1]);
        var c = a + b;

        c.Backward();

        // b was broadcast across 3 elements → accumulated gradient = 3
        Assert.Equal(3f, b.GetGradients()[0], Delta);
    }

    // C = A * I; C.Backward() → grad_A = I^T * grad_C = I (since grad_C = all ones after topo init)
    // We check that gradient flows into A at all (non-zero)
    [Fact]
    public void Backward_MatMul_PropagatesGradientsToLeaves()
    {
        var a = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        var b = new Tensor([1f, 0f, 0f, 1f], [2, 2]); // identity
        var c = a * b;

        c.Backward();

        // grad_A = grad_C * B^T = ones * I = ones (because Backward initialises root grad to 1 per element)
        // But Backward() sets _gradients[i]=1 only for the root scalar — here c has 4 elements,
        // so all 4 are set to 1. grad_A = [[1,1],[1,1]] * I^T = [[1,1],[1,1]]
        float[] gradA = a.GetGradients();
        Assert.All(gradA, g => Assert.NotEqual(0f, g));
    }

    [Fact]
    public void Backward_MatMul_Identity_GradAEqualsGradC()
    {
        // A * I = A, so grad_A should equal grad_C (grad_C = all-ones from root init)
        var a = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        var b = new Tensor([1f, 0f, 0f, 1f], [2, 2]);
        var c = a * b;

        c.Backward();

        // grad_A = grad_C * I^T = grad_C. Since root sets all c._gradients=1: grad_A = [[1,1],[1,1]]
        float[] gradA = a.GetGradients();
        Assert.All(gradA, g => Assert.Equal(1f, g, Delta));
    }

    // ReLU: positive input → gradient passes through
    [Fact]
    public void Backward_ReLU_PositiveInput_PassesGradient()
    {
        var t = new Tensor([1f, 2f, 3f], [3]);
        var o = Tensor.ReLU(t);

        o.Backward();

        Assert.All(t.GetGradients(), g => Assert.Equal(1f, g, Delta));
    }

    // ReLU: negative input → gradient is blocked (= 0)
    [Fact]
    public void Backward_ReLU_NegativeInput_BlocksGradient()
    {
        var t = new Tensor([-1f, -2f, -3f], [3]);
        var o = Tensor.ReLU(t);

        o.Backward();

        Assert.All(t.GetGradients(), g => Assert.Equal(0f, g, Delta));
    }

    // TanH: grad = (1 - tanh²(x)) * upstream_grad
    // For a root tensor upstream_grad = 1. At x=0: (1 - 0) * 1 = 1
    [Fact]
    public void Backward_TanH_AtZero_GradIsOne()
    {
        var t = new Tensor([0f], [1]);
        var o = Tensor.TanH(t);

        o.Backward();

        Assert.Equal(1f, t.GetGradients()[0], Delta);
    }

    [Fact]
    public void Backward_TanH_GradMatchesFormula()
    {
        float[] input = [-1f, 0f, 1f];
        var t = new Tensor(input, [3]);
        var o = Tensor.TanH(t);

        o.Backward();

        float[] grads = t.GetGradients();
        for (int i = 0; i < input.Length; i++)
        {
            float tanhVal = MathF.Tanh(input[i]);
            float expected = 1f - tanhVal * tanhVal; // upstream = 1
            Assert.Equal(expected, grads[i], Delta);
        }
    }

    // MSE backward: grad_p = (2/n) * (p - gt)
    [Fact]
    public void Backward_MSE_GradMatchesFormula()
    {
        var p  = new Tensor([2f, 4f], [2]);
        var gt = new Tensor([1f, 3f], [2]);
        var loss = Tensor.MSE(p, gt);

        loss.Backward();

        // scale = 1 (root grad) * 2 / 2 = 1; diff = [1, 1]
        float[] gradP  = p.GetGradients();
        float[] gradGt = gt.GetGradients();

        Assert.Equal(1f, gradP[0], Delta);
        Assert.Equal(1f, gradP[1], Delta);
        Assert.Equal(-1f, gradGt[0], Delta);
        Assert.Equal(-1f, gradGt[1], Delta);
    }

    [Fact]
    public void Adjust_UpdatesDataByLearningRate()
    {
        var t = new Tensor([2f, 4f], [2]);
        var gt = new Tensor([1f, 3f], [2]);
        Tensor.MSE(t, gt).Backward();

        float[] before = t.GetData();
        float[] grads  = t.GetGradients();
        float lr = 0.1f;

        t.Adjust(lr);

        float[] after = t.GetData();
        for (int i = 0; i < before.Length; i++)
            Assert.Equal(before[i] - lr * grads[i], after[i], Delta);
    }

    [Fact]
    public void Zero_ClearsAllGradients()
    {
        var t = new Tensor([2f, 4f], [2]);
        var gt = new Tensor([1f, 3f], [2]);
        Tensor.MSE(t, gt).Backward();

        t.Zero();

        Assert.All(t.GetGradients(), g => Assert.Equal(0f, g));
    }

    // End-to-end: linear → relu → mse. Gradients must flow all the way to weights.
    [Fact]
    public void Backward_ComputationChain_EndToEnd_GradientsAreNonZero()
    {
        // linear = [1,2] * [[1],[1]] = [3] > 0, so ReLU passes the gradient through
        var input   = new Tensor([1f, 2f], [1, 2]);
        var weights = new Tensor([1f, 1f], [2, 1]);

        var linear    = input * weights;        // [[3]]
        var activated = Tensor.ReLU(linear);    // [[3]] — positive, gradient flows
        var target    = new Tensor([0f], [1, 1]);
        var loss      = Tensor.MSE(activated, target);

        loss.Backward();

        float[] wGrads = weights.GetGradients();
        Assert.True(wGrads[0] != 0f || wGrads[1] != 0f,
            "Expected non-zero gradients in weights after end-to-end backward pass.");
    }
}
