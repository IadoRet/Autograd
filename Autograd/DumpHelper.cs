using System.Globalization;
using System.Text;

namespace Autograd.Demos;

public static class DumpHelper
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
}
