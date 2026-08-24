using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Core;

public partial class Tensor
{
    public static Tensor Sum(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        float sum = 0f;
        foreach (float value in input._data)
            sum += value;

        Tensor output = CreateOperation([sum], [1], input);
        output._backward = () =>
        {
            float upstream = output._gradients[0];
            for (int i = 0; i < input._gradients.Length; i++)
                input._gradients[i] += upstream;
        };

        return output;
    }

    public static Tensor Sum(Tensor input, int axis, bool keepDimension = false)
    {
        ArgumentNullException.ThrowIfNull(input);

        int normalizedAxis = NormalizeAxis(axis, input._shape.Length);
        int axisSize = input._shape[normalizedAxis];
        int outerSize = Product(input._shape, 0, normalizedAxis);
        int innerSize = Product(input._shape, normalizedAxis + 1, input._shape.Length);

        int[] outputShape = keepDimension
            ? input._shape.Select((dimension, index) => index == normalizedAxis ? 1 : dimension).ToArray()
            : input._shape.Where((_, index) => index != normalizedAxis).ToArray();

        float[] data = new float[outerSize * innerSize];
        for (int outer = 0; outer < outerSize; outer++)
        {
            for (int inner = 0; inner < innerSize; inner++)
            {
                int outputIndex = outer * innerSize + inner;
                for (int coordinate = 0; coordinate < axisSize; coordinate++)
                {
                    int inputIndex = (outer * axisSize + coordinate) * innerSize + inner;
                    data[outputIndex] += input._data[inputIndex];
                }
            }
        }

        Tensor output = CreateOperation(data, outputShape, input);
        output._backward = () =>
        {
            for (int outer = 0; outer < outerSize; outer++)
            {
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int outputIndex = outer * innerSize + inner;
                    for (int coordinate = 0; coordinate < axisSize; coordinate++)
                    {
                        int inputIndex = (outer * axisSize + coordinate) * innerSize + inner;
                        input._gradients[inputIndex] += output._gradients[outputIndex];
                    }
                }
            }
        };

        return output;
    }

    public static Tensor Mean(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        if (input._data.Length == 0)
            throw new TensorDimensionException("Mean is undefined for an empty tensor.");

        return Sum(input) / input._data.Length;
    }

    public static Tensor Mean(Tensor input, int axis, bool keepDimension = false)
    {
        ArgumentNullException.ThrowIfNull(input);

        int normalizedAxis = NormalizeAxis(axis, input._shape.Length);
        int axisSize = input._shape[normalizedAxis];
        if (axisSize == 0)
            throw new TensorDimensionException("Mean is undefined over an empty dimension.");

        return Sum(input, normalizedAxis, keepDimension) / axisSize;
    }

    public static Tensor Reshape(Tensor input, params int[] shape)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(shape);
        ValidateElementCount(input._data.Length, shape);

        Tensor output = CreateOperation(input._data.ToArray(), shape.ToArray(), input);
        output._backward = () =>
        {
            for (int i = 0; i < input._gradients.Length; i++)
                input._gradients[i] += output._gradients[i];
        };

        return output;
    }

    private static int NormalizeAxis(int axis, int rank)
    {
        int normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank)
            throw new ArgumentOutOfRangeException(nameof(axis), axis, $"Axis must be in the range [{-rank}, {rank - 1}].");

        return normalized;
    }

    private static int Product(int[] values, int start, int end)
    {
        int product = 1;
        for (int i = start; i < end; i++)
            product *= values[i];

        return product;
    }
}
