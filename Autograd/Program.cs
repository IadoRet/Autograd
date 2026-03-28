// See https://aka.ms/new-console-template for more information

using System.Globalization;
using System.Text;
using Autograd.Engine.Core;
using Autograd.Engine.Enums;
using Autograd.MLP;

const int epochs = 1000;
const int dataSize = 50;

MLP mlp = MLP.Create(2)
             .WithLayer(64, ActivationType.ReLU)
             .WithLayer(64, ActivationType.ReLU)
             .WithOutput(1);

Random random = new (328);

float loss = 0;
for (int i = 0; i < epochs; i++)
{
    loss = 0;
    (Tensor input, Tensor gt)[] trainingData = Enumerable.Range(0, dataSize)
                                                         .Select(_ => CreateData(random))
                                                         .ToArray();
    
    foreach ((Tensor input, Tensor gt) in trainingData)
    {
        Tensor o = mlp.Forward(input);
        Tensor mse = Tensor.MSE(o, gt);
        loss += mse.GetData().Single();
        
        //todo: batch learning
        
        mse.Backward();
        mlp.Adjust(0.0005f);
        mlp.Zero();
    }

    loss /= dataSize;
    Console.WriteLine($"EPOCH {i + 1}, LOSS: {loss}");
}

Console.WriteLine($"TRAINING FINISHED. LOSS: {loss}");

var example = CreateData(random);
var exampleOutput = mlp.Forward(example.input);
Console.ForegroundColor = ConsoleColor.Red;

Console.WriteLine($"EXAMPLE. INPUTS: {string.Join(',', example.input.GetData())}, PREDICTED: {exampleOutput.GetData().Single()}, GROUND TRUTH: {example.gt.GetData().Single()}");
Console.ResetColor();

// Dump for visualization

const int grid = 50;
const float range = 3f;

Dump("ground_truth.json", grid, range, Fn);
Dump("pred.json", grid, range, (a, b) =>
{
    Tensor input = new Tensor([a, b], [1, 2]);
    return mlp.Forward(input).GetData().Single();
});

return;

static void Dump(string path, int grid, float range, Func<float, float, float> eval)
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

(Tensor input, Tensor gt) CreateData(Random r)
{
    float a = r.NextSingle() * 4f - 2;
    float b = r.NextSingle() * 4f - 2;

    float output = Fn(a, b);
    
    Tensor input = new Tensor([a, b], [1, 2]);
    Tensor gt = new Tensor([output], [1, 1]);

    return (input, gt);
}

float Fn(float a, float b)
{
    return 3 * MathF.Pow(a, 2) + MathF.Pow(b, 2) - 5;
}