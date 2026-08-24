namespace Autograd.Engine.Core;

public partial class Tensor
{
    public static Tensor ReLU(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        float[] data = new float[input._data.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = MathF.Max(input._data[i], 0f);

        Tensor output = CreateOperation(data, input._shape.ToArray(), input);
        output._backward = () =>
        {
            for (int i = 0; i < input._gradients.Length; i++)
                input._gradients[i] += input._data[i] > 0f ? output._gradients[i] : 0f;
        };

        return output;
    }

    public static Tensor TanH(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);

        float[] data = new float[input._data.Length];
        for (int i = 0; i < data.Length; i++)
            data[i] = MathF.Tanh(input._data[i]);

        Tensor output = CreateOperation(data, input._shape.ToArray(), input);
        output._backward = () =>
        {
            for (int i = 0; i < input._gradients.Length; i++)
            {
                float value = output._data[i];
                input._gradients[i] += (1f - value * value) * output._gradients[i];
            }
        };

        return output;
    }
}
