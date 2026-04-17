using Autograd.Engine.Core;
using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Tests;

public class TensorConvolutionTests
{
    private const float Delta = 1e-5f;

    // ---- Validation ----

    [Fact]
    public void Convolution_LessThan3Dims_Throws()
    {
        var t = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        var k = new Tensor([1f], [1, 1]);

        Assert.Throws<TensorDimensionException>(() => Tensor.Convolution(t, k));
    }

    [Fact]
    public void Convolution_DimMismatch_Throws()
    {
        var t = new Tensor(new float[12], [1, 1, 3, 4]);
        var k = new Tensor(new float[6], [1, 1, 6]);

        Assert.Throws<TensorDimensionException>(() => Tensor.Convolution(t, k));
    }

    // ---- 1D convolution (shape: [N, C, W]) ----

    // input: [1, 1, 5] = [1, 2, 3, 4, 5]
    // kernel: [1, 1, 3] = [1, 0, -1]
    // output: [1, 1, 3] = [1*1+2*0+3*(-1), 2*1+3*0+4*(-1), 3*1+4*0+5*(-1)] = [-2, -2, -2]
    [Fact]
    public void Convolution1D_SingleBatchSingleChannel()
    {
        var t = new Tensor([1f, 2f, 3f, 4f, 5f], [1, 1, 5]);
        var k = new Tensor([1f, 0f, -1f], [1, 1, 3]);

        float[] result = Tensor.Convolution(t, k).GetData();

        Assert.Equal(3, result.Length);
        Assert.Equal(-2f, result[0], Delta);
        Assert.Equal(-2f, result[1], Delta);
        Assert.Equal(-2f, result[2], Delta);
    }

    // kernel size == input size => single output element
    // input: [1, 1, 3] = [2, 3, 4], kernel: [1, 1, 3] = [1, 1, 1]
    // output: [1, 1, 1] = [9]
    [Fact]
    public void Convolution1D_KernelSameAsInput_SingleOutput()
    {
        var t = new Tensor([2f, 3f, 4f], [1, 1, 3]);
        var k = new Tensor([1f, 1f, 1f], [1, 1, 3]);

        float[] result = Tensor.Convolution(t, k).GetData();

        Assert.Single(result);
        Assert.Equal(9f, result[0], Delta);
    }

    // ---- 2D convolution (shape: [N, C, H, W]) ----

    // input [1,1,3,3]:        kernel [1,1,2,2]:
    //  1 2 3                   1 0
    //  4 5 6                   0 1
    //  7 8 9
    // output [1,1,2,2]:
    //  (1*1+2*0+4*0+5*1)=6   (2*1+3*0+5*0+6*1)=8
    //  (4*1+5*0+7*0+8*1)=12  (5*1+6*0+8*0+9*1)=14
    [Fact]
    public void Convolution2D_3x3_Kernel2x2_Identity()
    {
        var t = new Tensor([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], [1, 1, 3, 3]);
        var k = new Tensor([1f, 0f, 0f, 1f], [1, 1, 2, 2]);

        float[] result = Tensor.Convolution(t, k).GetData();

        Assert.Equal(4, result.Length);
        Assert.Equal(6f, result[0], Delta);
        Assert.Equal(8f, result[1], Delta);
        Assert.Equal(12f, result[2], Delta);
        Assert.Equal(14f, result[3], Delta);
    }

    // input [1,1,4,4], kernel [1,1,3,3] all ones => each output = sum of 3x3 window
    // output [1,1,2,2]
    [Fact]
    public void Convolution2D_4x4_Kernel3x3_AllOnes()
    {
        float[] data =
        [
            1f, 2f, 3f, 4f,
            5f, 6f, 7f, 8f,
            9f, 10f, 11f, 12f,
            13f, 14f, 15f, 16f
        ];
        float[] kData = Enumerable.Repeat(1f, 9).ToArray();

        var t = new Tensor(data, [1, 1, 4, 4]);
        var k = new Tensor(kData, [1, 1, 3, 3]);

        float[] result = Tensor.Convolution(t, k).GetData();

        Assert.Equal(4, result.Length);
        // top-left 3x3: 1+2+3+5+6+7+9+10+11 = 54
        Assert.Equal(54f, result[0], Delta);
        // top-right 3x3: 2+3+4+6+7+8+10+11+12 = 63
        Assert.Equal(63f, result[1], Delta);
        // bottom-left 3x3: 5+6+7+9+10+11+13+14+15 = 90
        Assert.Equal(90f, result[2], Delta);
        // bottom-right 3x3: 6+7+8+10+11+12+14+15+16 = 99
        Assert.Equal(99f, result[3], Delta);
    }

    // ---- Multiple batches / channels ----

    // Shared 1x1 kernel convolved over two batches independently.
    [Fact]
    public void Convolution2D_TwoBatches()
    {
        // batch 0: all ones, batch 1: all twos. Kernel: single 2x2 all ones.
        float[] data = Enumerable.Repeat(1f, 9).Concat(Enumerable.Repeat(2f, 9)).ToArray();
        float[] kData = Enumerable.Repeat(1f, 4).ToArray();

        var t = new Tensor(data, [2, 1, 3, 3]);
        var k = new Tensor(kData, [1, 1, 2, 2]);

        float[] result = Tensor.Convolution(t, k).GetData();

        // output [2, 1, 2, 2]: batch 0 => 4*1=4, batch 1 => 4*2=8
        Assert.Equal(8, result.Length);
        for (int i = 0; i < 4; i++)
            Assert.Equal(4f, result[i], Delta);
        for (int i = 4; i < 8; i++)
            Assert.Equal(8f, result[i], Delta);
    }

    // Two input channels are summed into a single output channel.
    [Fact]
    public void Convolution2D_TwoInputChannels()
    {
        // ch0: 1s, ch1: 3s. Kernel [OutC=1, InC=2, 2, 2] all ones.
        float[] data = Enumerable.Repeat(1f, 9).Concat(Enumerable.Repeat(3f, 9)).ToArray();
        float[] kData = Enumerable.Repeat(1f, 8).ToArray();

        var t = new Tensor(data, [1, 2, 3, 3]);
        var k = new Tensor(kData, [1, 2, 2, 2]);

        float[] result = Tensor.Convolution(t, k).GetData();

        // output [1, 1, 2, 2]: each = (4*1) + (4*3) = 16
        Assert.Equal(4, result.Length);
        for (int i = 0; i < 4; i++)
            Assert.Equal(16f, result[i], Delta);
    }

    // Single input channel produces two output channels via two distinct filters.
    [Fact]
    public void Convolution2D_TwoOutputChannels()
    {
        // input [1, 1, 3, 3] = 1..9
        // kernel [OutC=2, InC=1, 2, 2]:
        //   filter 0 = [[1,0],[0,1]]  (diagonal identity)
        //   filter 1 = [[1,1],[1,1]]  (sum)
        float[] data = [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f];
        float[] kData = [1f, 0f, 0f, 1f,
                         1f, 1f, 1f, 1f];

        var t = new Tensor(data, [1, 1, 3, 3]);
        var k = new Tensor(kData, [2, 1, 2, 2]);

        float[] result = Tensor.Convolution(t, k).GetData();

        // output [1, 2, 2, 2]:
        //   out ch0 (diagonal): 1+5=6, 2+6=8, 4+8=12, 5+9=14
        //   out ch1 (sum):      1+2+4+5=12, 2+3+5+6=16, 4+5+7+8=24, 5+6+8+9=28
        Assert.Equal(8, result.Length);
        Assert.Equal(6f, result[0], Delta);
        Assert.Equal(8f, result[1], Delta);
        Assert.Equal(12f, result[2], Delta);
        Assert.Equal(14f, result[3], Delta);
        Assert.Equal(12f, result[4], Delta);
        Assert.Equal(16f, result[5], Delta);
        Assert.Equal(24f, result[6], Delta);
        Assert.Equal(28f, result[7], Delta);
    }

    [Fact]
    public void Convolution_InputChannelMismatch_Throws()
    {
        // input has InC=2, kernel expects InC=3
        var t = new Tensor(new float[18], [1, 2, 3, 3]);
        var k = new Tensor(new float[12], [1, 3, 2, 2]);

        Assert.Throws<TensorDimensionException>(() => Tensor.Convolution(t, k));
    }

    // ---- 3D convolution (shape: [N, C, D, H, W]) ----

    // input [1,1,3,3,3] all ones, kernel [1,1,2,2,2] all ones
    // output [1,1,2,2,2], each element = 8
    [Fact]
    public void Convolution3D_AllOnes()
    {
        float[] data = Enumerable.Repeat(1f, 27).ToArray();
        float[] kData = Enumerable.Repeat(1f, 8).ToArray();

        var t = new Tensor(data, [1, 1, 3, 3, 3]);
        var k = new Tensor(kData, [1, 1, 2, 2, 2]);

        float[] result = Tensor.Convolution(t, k).GetData();

        Assert.Equal(8, result.Length);
        for (int i = 0; i < 8; i++)
            Assert.Equal(8f, result[i], Delta);
    }

    // ---- Backward (gradients) ----

    [Fact]
    public void Convolution2D_Backward_InputGradients()
    {
        // input [1,1,3,3], kernel [1,1,2,2] all ones
        // output [1,1,2,2], each = sum of 2x2 window
        // After backward with all-ones output grad:
        // input grad at (r,c) = number of output windows covering (r,c)
        var t = new Tensor([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], [1, 1, 3, 3]);
        var k = new Tensor([1f, 1f, 1f, 1f], [1, 1, 2, 2]);

        Tensor o = Tensor.Convolution(t, k);
        o.Backward();

        float[] tGrad = t.GetGradients();

        // corners touched by 1 window
        Assert.Equal(1f, tGrad[0], Delta); // (0,0)
        Assert.Equal(1f, tGrad[2], Delta); // (0,2)
        Assert.Equal(1f, tGrad[6], Delta); // (2,0)
        Assert.Equal(1f, tGrad[8], Delta); // (2,2)

        // edges touched by 2 windows
        Assert.Equal(2f, tGrad[1], Delta); // (0,1)
        Assert.Equal(2f, tGrad[3], Delta); // (1,0)
        Assert.Equal(2f, tGrad[5], Delta); // (1,2)
        Assert.Equal(2f, tGrad[7], Delta); // (2,1)

        // center touched by 4 windows
        Assert.Equal(4f, tGrad[4], Delta); // (1,1)
    }

    [Fact]
    public void Convolution2D_Backward_KernelGradients()
    {
        // input [1,1,3,3] = 1..9, kernel [1,1,2,2] all ones
        // kernel grad[i,j] = sum of input patches at (i,j) across all output positions
        var t = new Tensor([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], [1, 1, 3, 3]);
        var k = new Tensor([1f, 1f, 1f, 1f], [1, 1, 2, 2]);

        Tensor o = Tensor.Convolution(t, k);
        o.Backward();

        float[] kGrad = k.GetGradients();

        // kGrad[0,0] = sum of top-left of each 2x2 window = 1+2+4+5 = 12
        Assert.Equal(12f, kGrad[0], Delta);
        // kGrad[0,1] = 2+3+5+6 = 16
        Assert.Equal(16f, kGrad[1], Delta);
        // kGrad[1,0] = 4+5+7+8 = 24
        Assert.Equal(24f, kGrad[2], Delta);
        // kGrad[1,1] = 5+6+8+9 = 28
        Assert.Equal(28f, kGrad[3], Delta);
    }

    // Multi output-channel backward: input grad must accumulate across all OutC.
    [Fact]
    public void Convolution2D_MultiOutputChannel_Backward()
    {
        // input [1, 1, 3, 3] = 1..9
        // kernel [OutC=2, InC=1, 2, 2]:
        //   filter 0 = [[1,0],[0,0]]  (top-left only)
        //   filter 1 = [[0,0],[0,1]]  (bottom-right only)
        var t = new Tensor([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], [1, 1, 3, 3]);
        var k = new Tensor([1f, 0f, 0f, 0f,
                            0f, 0f, 0f, 1f], [2, 1, 2, 2]);

        Tensor o = Tensor.Convolution(t, k);
        o.Backward();

        float[] tGrad = t.GetGradients();
        float[] kGrad = k.GetGradients();

        // grad_input = Σ_oc grad_out * kernel_sum. With grad_out all ones,
        // grad_input[y,x] = count of (oc, outPos, kPos) triples that land at (y,x).
        // filter 0 only contributes at offset (0,0); filter 1 only at offset (1,1).
        float[] expectedTGrad =
        [
            1f, 1f, 0f, // filter 0 covers (0,0),(0,1); filter 1 doesn't reach here
            1f, 2f, 1f, // (1,1): filter 0 at out(1,1) + filter 1 at out(0,0)
            0f, 1f, 1f  // filter 1 covers (2,1),(2,2)
        ];
        for (int i = 0; i < 9; i++)
            Assert.Equal(expectedTGrad[i], tGrad[i], Delta);

        // grad_kernel[oc, 0, ky, kx] = sum of input[y+ky, x+kx] across all output positions.
        // Same values for both oc since grad_out is uniform 1s.
        float[] expectedKGrad = [12f, 16f, 24f, 28f, 12f, 16f, 24f, 28f];
        for (int i = 0; i < 8; i++)
            Assert.Equal(expectedKGrad[i], kGrad[i], Delta);
    }

    [Fact]
    public void Convolution1D_Backward_Gradients()
    {
        var t = new Tensor([1f, 2f, 3f, 4f], [1, 1, 4]);
        var k = new Tensor([1f, -1f], [1, 1, 2]);

        Tensor o = Tensor.Convolution(t, k);
        o.Backward();

        float[] tGrad = t.GetGradients();
        float[] kGrad = k.GetGradients();

        // output = [1-2, 2-3, 3-4] = [-1, -1, -1]
        // tGrad: pos0 covered by window0 with k[0]=1 => 1
        //        pos1 covered by window0 k[1]=-1 and window1 k[0]=1 => 0
        //        pos2 covered by window1 k[1]=-1 and window2 k[0]=1 => 0
        //        pos3 covered by window2 k[1]=-1 => -1
        Assert.Equal(1f, tGrad[0], Delta);
        Assert.Equal(0f, tGrad[1], Delta);
        Assert.Equal(0f, tGrad[2], Delta);
        Assert.Equal(-1f, tGrad[3], Delta);

        // kGrad[0] = sum of input at k[0] position across windows = 1+2+3 = 6
        Assert.Equal(6f, kGrad[0], Delta);
        // kGrad[1] = 2+3+4 = 9
        Assert.Equal(9f, kGrad[1], Delta);
    }
}
