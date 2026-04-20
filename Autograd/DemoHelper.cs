using System.Globalization;
using System.Text;

namespace Autograd;

public static class DemoHelper
{
    public static void Dump(string path, int grid, float range, Func<float, float, float> eval)
    {
        float step = 2f * range / (grid - 1);
        var sb = new StringBuilder();
        sb.Append('[');
        for (int i = 0; i < grid; i++)
        {
            sb.Append('[');
            for (int j = 0; j < grid; j++)
            {
                float a = -range + i * step;
                float b = -range + j * step;
                sb.Append(eval(a, b).ToString(CultureInfo.InvariantCulture));
                if (j < grid - 1) sb.Append(',');
            }
            sb.Append(']');
            if (i < grid - 1) sb.Append(',');
        }
        sb.Append(']');
        File.WriteAllText(path, sb.ToString());
    }

    public static void DumpCnnData(string path, float[][] input, float[][] prediction, float[][] groundTruth, float[] lossHistory)
    {
        var sb = new StringBuilder();
        sb.Append('{');
        sb.Append("\"input\":"); Append2D(sb, input);
        sb.Append(",\"groundTruth\":"); Append2D(sb, groundTruth);
        sb.Append(",\"prediction\":"); Append2D(sb, prediction);
        sb.Append(",\"lossHistory\":"); Append1D(sb, lossHistory);
        sb.Append('}');
        File.WriteAllText(path, sb.ToString());
    }

    public static float[][] ReshapeTo2D(float[] flat, int rows, int cols)
    {
        float[][] result = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            result[i] = new float[cols];
            Array.Copy(flat, i * cols, result[i], 0, cols);
        }
        return result;
    }

    private static void Append2D(StringBuilder sb, float[][] data)
    {
        sb.Append('[');
        for (int i = 0; i < data.Length; i++)
        {
            Append1D(sb, data[i]);
            if (i < data.Length - 1) sb.Append(',');
        }
        sb.Append(']');
    }

    private static void Append1D(StringBuilder sb, float[] data)
    {
        sb.Append('[');
        for (int i = 0; i < data.Length; i++)
        {
            sb.Append(data[i].ToString(CultureInfo.InvariantCulture));
            if (i < data.Length - 1) sb.Append(',');
        }
        sb.Append(']');
    }
}
