using Autograd.Engine.Core;
using Autograd.Engine.Enums;

namespace Autograd.Demos;

public class MlpDemo : IDemo
{
    private const int Epochs = 1000;
    private const int DataSize = 50;
    private const int Grid = 50;
    private const float Range = 3f;

    public string Name => "Multi-Layer Perceptron";

    public void Run()
    {
        MLP.MLP mlp = MLP.MLP.Create(2)
                             .WithLayer(64, ActivationType.ReLU)
                             .WithLayer(64, ActivationType.ReLU)
                             .WithOutput(1);

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
                Tensor o = mlp.Forward(input);
                Tensor mse = Tensor.MSE(o, gt);
                loss += mse.GetData().Single();
                
                mse.Backward();
                mlp.Adjust(0.0005f);
                mlp.Zero();
            }

            loss /= DataSize;
            Console.WriteLine($"EPOCH {i + 1}, LOSS: {loss}");
        }

        Console.WriteLine($"TRAINING FINISHED. LOSS: {loss}");

        var example = CreateData(random);
        var exampleOutput = mlp.Forward(example.input);
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"EXAMPLE. INPUTS: {string.Join(',', example.input.GetData())}, PREDICTED: {exampleOutput.GetData().Single()}, GROUND TRUTH: {example.gt.GetData().Single()}");
        Console.ResetColor();

        DumpHelper.Dump("ground_truth.json", Grid, Range, Fn);
        DumpHelper.Dump("pred.json", Grid, Range, (a, b) =>
        {
            Tensor input = new Tensor([a, b], [1, 2]);
            return mlp.Forward(input).GetData().Single();
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
