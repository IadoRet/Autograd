using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Core;

public partial class Tensor
{
    /// <summary>
    /// Matrix multiplication over the final two dimensions. Leading batch dimensions must match.
    /// </summary>
    public static Tensor operator *(Tensor left, Tensor right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (left._shape.Length < 2 || right._shape.Length < 2)
            throw new TensorDimensionException("Matrix multiplication requires tensors with rank 2 or greater.");

        if (left._shape.Length != right._shape.Length)
            throw new TensorDimensionException("Matrix multiplication requires tensors with the same rank.");

        int m = left._shape[^2];
        int n = right._shape[^1];
        int k = left._shape[^1];

        if (k != right._shape[^2])
        {
            throw new TensorDimensionException(
                $"Dimensions do not match. Left inner dimension [{k}] != right outer dimension [{right._shape[^2]}].");
        }

        int[] shape = new int[left._shape.Length];
        for (int i = 0; i < shape.Length - 2; i++)
        {
            if (left._shape[i] != right._shape[i])
                throw new TensorDimensionException($"Batch dimensions do not match at dimension [{i}].");

            shape[i] = left._shape[i];
        }

        shape[^2] = m;
        shape[^1] = n;

        float[] data = MatrixMultiply(left._data, right._data, shape, m, n, k);
        Tensor output = CreateOperation(data, shape, left, right);
        output._backward = Backward;
        return output;

        void Backward()
        {
            float[] leftGradients = MatrixMultiply(
                output._gradients, right._data, left._shape, m, k, n, transposeRight: true);
            float[] rightGradients = MatrixMultiply(
                left._data, output._gradients, right._shape, k, n, m, transposeLeft: true);

            for (int i = 0; i < leftGradients.Length; i++)
                left._gradients[i] += leftGradients[i];

            for (int i = 0; i < rightGradients.Length; i++)
                right._gradients[i] += rightGradients[i];
        }
    }

    private static float[] MatrixMultiply(
        float[] left,
        float[] right,
        int[] shape,
        int m,
        int n,
        int k,
        bool transposeLeft = false,
        bool transposeRight = false)
    {
        int elementCount = shape.Aggregate(1, (product, dimension) => product * dimension);
        float[] result = new float[elementCount];
        int batches = 1;

        for (int i = 0; i < shape.Length - 2; i++)
            batches *= shape[i];

        for (int batch = 0; batch < batches; batch++)
        {
            int leftOffset = batch * m * k;
            int rightOffset = batch * k * n;
            int resultOffset = batch * m * n;

            for (int row = 0; row < m; row++)
            {
                for (int column = 0; column < n; column++)
                {
                    for (int inner = 0; inner < k; inner++)
                    {
                        int leftIndex = transposeLeft
                            ? leftOffset + inner * m + row
                            : leftOffset + row * k + inner;
                        int rightIndex = transposeRight
                            ? rightOffset + column * k + inner
                            : rightOffset + inner * n + column;

                        result[resultOffset + row * n + column] += left[leftIndex] * right[rightIndex];
                    }
                }
            }
        }

        return result;
    }
}
