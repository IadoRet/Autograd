using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Core;

public partial class Tensor
{
    /// <summary>
    /// Valid-mode N-dimensional convolution.
    /// Input shape: [batch, input channels, spatial...].
    /// Kernel shape: [output channels, input channels, kernel spatial...].
    /// </summary>
    public static Tensor Convolution(Tensor input, Tensor kernel)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(kernel);

        if (input._shape.Length < 3)
            throw new TensorDimensionException("Convolution requires tensors with rank 3 or greater.");
        if (input._shape.Length != kernel._shape.Length)
            throw new TensorDimensionException("Tensor and kernel ranks do not match.");
        if (input._shape[1] != kernel._shape[1])
        {
            throw new TensorDimensionException(
                $"Input channels do not match. Input: [{input._shape[1]}], kernel: [{kernel._shape[1]}].");
        }

        int rank = input._shape.Length;
        int batches = input._shape[0];
        int inputChannels = input._shape[1];
        int outputChannels = kernel._shape[0];
        int spatialRank = rank - 2;
        int[] shape = new int[rank];
        shape[0] = batches;
        shape[1] = outputChannels;

        int outputSpatialSize = 1;
        int kernelSpatialSize = 1;
        int inputSpatialSize = 1;
        int[] inputStrides = new int[spatialRank];

        for (int dimension = 2; dimension < rank; dimension++)
        {
            if (input._shape[dimension] == 0 || kernel._shape[dimension] == 0)
                throw new TensorDimensionException("Convolution spatial dimensions must be positive.");
            if (kernel._shape[dimension] > input._shape[dimension])
            {
                throw new TensorDimensionException(
                    $"Kernel dimension [{kernel._shape[dimension]}] exceeds input dimension [{input._shape[dimension]}] at axis [{dimension}].");
            }

            shape[dimension] = input._shape[dimension] - kernel._shape[dimension] + 1;
            outputSpatialSize *= shape[dimension];
            kernelSpatialSize *= kernel._shape[dimension];
            inputSpatialSize *= input._shape[dimension];
        }

        inputStrides[^1] = 1;
        for (int i = spatialRank - 2; i >= 0; i--)
            inputStrides[i] = inputStrides[i + 1] * input._shape[i + 3];

        float[] data = new float[batches * outputChannels * outputSpatialSize];

        for (int batch = 0; batch < batches; batch++)
        {
            for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++)
            {
                int outputBase = (batch * outputChannels + outputChannel) * outputSpatialSize;

                for (int outputPosition = 0; outputPosition < outputSpatialSize; outputPosition++)
                {
                    float sum = 0f;
                    for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
                    {
                        int inputBase = (batch * inputChannels + inputChannel) * inputSpatialSize;
                        int kernelBase = (outputChannel * inputChannels + inputChannel) * kernelSpatialSize;

                        for (int kernelPosition = 0; kernelPosition < kernelSpatialSize; kernelPosition++)
                        {
                            int inputOffset = GetConvolutionInputOffset(
                                outputPosition,
                                kernelPosition,
                                shape,
                                kernel._shape,
                                inputStrides,
                                spatialRank);
                            sum += input._data[inputBase + inputOffset] * kernel._data[kernelBase + kernelPosition];
                        }
                    }

                    data[outputBase + outputPosition] = sum;
                }
            }
        }

        Tensor output = CreateOperation(data, shape, input, kernel);
        output._backward = () =>
        {
            for (int batch = 0; batch < batches; batch++)
            {
                for (int outputChannel = 0; outputChannel < outputChannels; outputChannel++)
                {
                    int outputBase = (batch * outputChannels + outputChannel) * outputSpatialSize;

                    for (int outputPosition = 0; outputPosition < outputSpatialSize; outputPosition++)
                    {
                        float upstream = output._gradients[outputBase + outputPosition];

                        for (int inputChannel = 0; inputChannel < inputChannels; inputChannel++)
                        {
                            int inputBase = (batch * inputChannels + inputChannel) * inputSpatialSize;
                            int kernelBase = (outputChannel * inputChannels + inputChannel) * kernelSpatialSize;

                            for (int kernelPosition = 0; kernelPosition < kernelSpatialSize; kernelPosition++)
                            {
                                int inputOffset = GetConvolutionInputOffset(
                                    outputPosition,
                                    kernelPosition,
                                    shape,
                                    kernel._shape,
                                    inputStrides,
                                    spatialRank);
                                int inputIndex = inputBase + inputOffset;
                                int kernelIndex = kernelBase + kernelPosition;
                                input._gradients[inputIndex] += upstream * kernel._data[kernelIndex];
                                kernel._gradients[kernelIndex] += upstream * input._data[inputIndex];
                            }
                        }
                    }
                }
            }
        };

        return output;
    }

    private static int GetConvolutionInputOffset(
        int outputPosition,
        int kernelPosition,
        int[] outputShape,
        int[] kernelShape,
        int[] inputStrides,
        int spatialRank)
    {
        int inputOffset = 0;
        int remainingOutput = outputPosition;
        int remainingKernel = kernelPosition;

        for (int dimension = spatialRank - 1; dimension >= 0; dimension--)
        {
            int outputCoordinate = remainingOutput % outputShape[dimension + 2];
            remainingOutput /= outputShape[dimension + 2];
            int kernelCoordinate = remainingKernel % kernelShape[dimension + 2];
            remainingKernel /= kernelShape[dimension + 2];
            inputOffset += (outputCoordinate + kernelCoordinate) * inputStrides[dimension];
        }

        return inputOffset;
    }
}
