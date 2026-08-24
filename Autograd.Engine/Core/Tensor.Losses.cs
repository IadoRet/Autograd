using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Core;

public partial class Tensor
{
    // ReSharper disable once InconsistentNaming
    public static Tensor MSE(Tensor prediction, Tensor groundTruth)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(groundTruth);

        if (!HaveSameShape(prediction, groundTruth))
            throw new TensorDimensionException("Mean square error requires tensors with identical shapes.");

        if (prediction._data.Length == 0)
            throw new TensorDimensionException("Mean square error is undefined for empty tensors.");

        float sum = 0f;
        for (int i = 0; i < prediction._data.Length; i++)
        {
            float difference = prediction._data[i] - groundTruth._data[i];
            sum += difference * difference;
        }

        int elementCount = prediction._data.Length;
        Tensor output = CreateOperation([sum / elementCount], [1], prediction, groundTruth);
        output._backward = () =>
        {
            float scale = output._gradients[0] * 2f / elementCount;
            for (int i = 0; i < elementCount; i++)
            {
                float difference = prediction._data[i] - groundTruth._data[i];
                prediction._gradients[i] += scale * difference;
                groundTruth._gradients[i] -= scale * difference;
            }
        };

        return output;
    }
}
