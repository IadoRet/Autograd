using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Core;

public partial class Tensor
{
    public static Tensor operator +(Tensor left, Tensor right)
    {
        return BroadcastBinary(
            left,
            right,
            static (a, b) => a + b,
            static (_, _, upstream) => upstream,
            static (_, _, upstream) => upstream);
    }

    public static Tensor operator -(Tensor left, Tensor right)
    {
        return Subtract(left, right);
    }

    public static Tensor operator *(Tensor tensor, float scalar)
    {
        return Scale(tensor, scalar);
    }

    public static Tensor operator *(float scalar, Tensor tensor)
    {
        return Scale(tensor, scalar);
    }

    public static Tensor operator /(Tensor tensor, float scalar)
    {
        if (scalar == 0f)
            throw new DivideByZeroException("Cannot divide a tensor by zero.");

        return Scale(tensor, 1f / scalar);
    }

    public static Tensor Subtract(Tensor left, Tensor right)
    {
        return BroadcastBinary(
            left,
            right,
            static (a, b) => a - b,
            static (_, _, upstream) => upstream,
            static (_, _, upstream) => -upstream);
    }

    public static Tensor MultiplyElementwise(Tensor left, Tensor right)
    {
        return BroadcastBinary(
            left,
            right,
            static (a, b) => a * b,
            static (_, b, upstream) => b * upstream,
            static (a, _, upstream) => a * upstream);
    }

    public static Tensor Abs(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        float[] data = new float[input._data.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = MathF.Abs(input._data[i]);

        Tensor output = CreateOperation(data, input._shape.ToArray(), input);
        output._backward = () =>
        {
            for (int i = 0; i < data.Length; i++)
            {
                float derivative = input._data[i] > 0f ? 1f : input._data[i] < 0f ? -1f : 0f;
                input._gradients[i] += derivative * output._gradients[i];
            }
        };

        return output;
    }

    public static Tensor Square(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        float[] data = new float[input._data.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = input._data[i] * input._data[i];

        Tensor output = CreateOperation(data, input._shape.ToArray(), input);
        output._backward = () =>
        {
            for (int i = 0; i < data.Length; i++)
                input._gradients[i] += 2f * input._data[i] * output._gradients[i];
        };

        return output;
    }

    public static Tensor Log(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        float[] data = new float[input._data.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = MathF.Log(input._data[i]);

        Tensor output = CreateOperation(data, input._shape.ToArray(), input);
        output._backward = () =>
        {
            for (int i = 0; i < data.Length; i++)
                input._gradients[i] += output._gradients[i] / input._data[i];
        };

        return output;
    }

    private static Tensor Scale(Tensor input, float scalar)
    {
        ArgumentNullException.ThrowIfNull(input);

        float[] data = new float[input._data.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = input._data[i] * scalar;

        Tensor output = CreateOperation(data, input._shape.ToArray(), input);
        output._backward = () =>
        {
            for (int i = 0; i < data.Length; i++)
                input._gradients[i] += scalar * output._gradients[i];
        };

        return output;
    }

    private static Tensor BroadcastBinary(
        Tensor left,
        Tensor right,
        Func<float, float, float> operation,
        Func<float, float, float, float> leftDerivative,
        Func<float, float, float, float> rightDerivative)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        BroadcastPlan plan = CreateBroadcastPlan(left._shape, right._shape);
        float[] data = new float[plan.ElementCount];

        for (int i = 0; i < data.Length; i++)
        {
            (int leftIndex, int rightIndex) = plan.GetSourceIndices(i);
            data[i] = operation(left._data[leftIndex], right._data[rightIndex]);
        }

        Tensor output = CreateOperation(data, plan.Shape, left, right);
        output._backward = () =>
        {
            for (int i = 0; i < data.Length; i++)
            {
                (int leftIndex, int rightIndex) = plan.GetSourceIndices(i);
                float upstream = output._gradients[i];
                float leftValue = left._data[leftIndex];
                float rightValue = right._data[rightIndex];

                left._gradients[leftIndex] += leftDerivative(leftValue, rightValue, upstream);
                right._gradients[rightIndex] += rightDerivative(leftValue, rightValue, upstream);
            }
        };

        return output;
    }

    private static BroadcastPlan CreateBroadcastPlan(int[] leftShape, int[] rightShape)
    {
        int rank = Math.Max(leftShape.Length, rightShape.Length);
        int[] expandedLeft = new int[rank];
        int[] expandedRight = new int[rank];
        int[] leftStrides = new int[rank];
        int[] rightStrides = new int[rank];
        int[] shape = new int[rank];
        int elementCount = 1;

        for (int i = rank - 1; i >= 0; i--)
        {
            int leftIndex = i - (rank - leftShape.Length);
            int rightIndex = i - (rank - rightShape.Length);
            int leftDimension = leftIndex < 0 ? 1 : leftShape[leftIndex];
            int rightDimension = rightIndex < 0 ? 1 : rightShape[rightIndex];

            if (leftDimension != rightDimension && leftDimension != 1 && rightDimension != 1)
            {
                throw new TensorDimensionException(
                    $"Dimensions do not match at dimension [{i}]: [{leftDimension}] <=> [{rightDimension}].");
            }

            expandedLeft[i] = leftDimension;
            expandedRight[i] = rightDimension;
            shape[i] = leftDimension == 1 ? rightDimension : leftDimension;
            elementCount *= shape[i];
        }

        int leftStride = 1;
        int rightStride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            leftStrides[i] = expandedLeft[i] == 1 ? 0 : leftStride;
            rightStrides[i] = expandedRight[i] == 1 ? 0 : rightStride;
            leftStride *= expandedLeft[i];
            rightStride *= expandedRight[i];
        }

        return new BroadcastPlan(shape, leftStrides, rightStrides, elementCount);
    }

    private sealed record BroadcastPlan(int[] Shape, int[] LeftStrides, int[] RightStrides, int ElementCount)
    {
        public (int Left, int Right) GetSourceIndices(int flatIndex)
        {
            int remaining = flatIndex;
            int left = 0;
            int right = 0;

            for (int dimension = Shape.Length - 1; dimension >= 0; dimension--)
            {
                int coordinate = remaining % Shape[dimension];
                remaining /= Shape[dimension];
                left += coordinate * LeftStrides[dimension];
                right += coordinate * RightStrides[dimension];
            }

            return (left, right);
        }
    }
}
