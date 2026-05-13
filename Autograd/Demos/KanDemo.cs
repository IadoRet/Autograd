using Autograd.Engine.Core;

namespace Autograd.Demos;

public class KanDemo : IDemo
{
    private const int Epochs = 500;
    private const int DataSize = 50;
    private const int Grid = 50;
    private const float Range = 3f;

    public string Name => "Kolmogorov-Arnold Network";

    public void Run()
    {
        KAN.KAN kan = KAN.KAN.Create(2)
                             .WithOutput(1, degree: 2);

        Random random = new(328);

        float loss = 0;
        for (int i = 0; i < Epochs; i++)
        {
            loss = 0;
            (Tensor input, Tensor gt)[] trainingData = Enumerable.Range(0, DataSize)
                                                                 .Select(_ => CreateData(random))
                                                                 .ToArray();

            foreach ((Tensor input, Tensor gt) in trainingData)
            {
                Tensor o = kan.Forward(input);
                Tensor mse = Tensor.MSE(o, gt);
                loss += mse.GetData().Single();

                mse.Backward();
                kan.Adjust(0.001f);
                kan.Zero();
            }

            loss /= DataSize;
            Console.WriteLine($"EPOCH {i + 1}, LOSS: {loss}");
        }

        Console.WriteLine($"TRAINING FINISHED. LOSS: {loss}");

        var example = CreateData(random);
        var exampleOutput = kan.Forward(example.input);
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"EXAMPLE. INPUTS: {string.Join(',', example.input.GetData())}, PREDICTED: {exampleOutput.GetData().Single()}, GROUND TRUTH: {example.gt.GetData().Single()}");
        Console.ResetColor();

        DemoHelper.Dump("kan_ground_truth.json", Grid, Range, Fn);
        DemoHelper.Dump("kan_pred.json", Grid, Range, (a, b) =>
        {
            Tensor input = new Tensor([a, b], [1, 2]);
            return kan.Forward(input).GetData().Single();
        });
    }

    private static (Tensor input, Tensor gt) CreateData(Random r)
    {
        float a = r.NextSingle() * 4f - 2;
        float b = r.NextSingle() * 4f - 2;

        float output = Fn(a, b);

        Tensor input = new Tensor([a, b], [1, 2]);
        Tensor gt = new Tensor([output], [1, 1]);

        return (input, gt);
    }

    private static float Fn(float a, float b)
    {
        return 3 * MathF.Pow(a, 2) + MathF.Pow(b, 2) - 5;
    }
}
