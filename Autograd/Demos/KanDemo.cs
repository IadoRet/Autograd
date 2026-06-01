using Autograd.Engine.Core;
using Autograd.Engine.Enums;
using System.Diagnostics;

namespace Autograd.Demos;

public class KanDemo : IDemo
{
    private const int Epochs = 360;
    private const int DataSize = 32;
    private const int ValidationSize = 200;
    private const int CheckpointInterval = 20;
    private const int Grid = 50;
    private const float Range = 3f;
    private const float LearningRate = 0.001f;
    private const int ModelSeed = 328;
    private const int TrainingSeed = 1207;
    private const int ValidationSeed = 404;

    public string Name => "Kolmogorov-Arnold Network";

    public void Run()
    {
        Experiment[] experiments =
        [
            new("Polynomial 0..2", "poly_full_2", () => KAN.KAN.Create(2, ModelSeed)
                                                               .WithPolynomialOutput(1, [0, 1, 2])),
            new("Polynomial [2]", "poly_2", () => KAN.KAN.Create(2, ModelSeed)
                                                         .WithPolynomialOutput(1, [2])),
            new("Polynomial [0,2]", "poly_0_2", () => KAN.KAN.Create(2, ModelSeed)
                                                             .WithPolynomialOutput(1, [0, 2])),
            new("Polynomial [0,1]", "poly_0_1", () => KAN.KAN.Create(2, ModelSeed)
                                                             .WithPolynomialOutput(1, [0, 1])),
            new("Chebyshev 0..2", "cheb_full_2", () => KAN.KAN.Create(2, ModelSeed)
                                                              .WithPolynomialOutput(1, [0, 1, 2], BasisType.Chebyshev)),
            new("Cubic B-spline", "spline_cubic", () => KAN.KAN.Create(2, ModelSeed)
                                                               .WithSplineOutput(1, gridSize: 8, splineOrder: 3, gridMin: -3f, gridMax: 3f))
        ];

        List<ExperimentResult> results = [];

        foreach (Experiment experiment in experiments)
        {
            Console.WriteLine($"-- {experiment.Name} --");
            ExperimentResult result = Train(experiment);
            results.Add(result);
            Console.WriteLine();
        }

        PrintSummary(results);

        ExperimentResult best = results.MinBy(r => r.FinalLoss)!;
        Console.WriteLine();
        Console.WriteLine($"BEST: {best.Name}. VALIDATION LOSS: {best.FinalLoss:0.000000}");

        var example = CreateData(new Random(ValidationSeed + 1));
        var exampleOutput = best.Model.Forward(example.input);
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"EXAMPLE. INPUTS: {string.Join(',', example.input.GetData())}, PREDICTED: {exampleOutput.GetData().Single()}, GROUND TRUTH: {example.gt.GetData().Single()}");
        Console.ResetColor();

        DemoHelper.Dump("kan_ground_truth.json", Grid, Range, Fn);

        foreach (ExperimentResult result in results)
        {
            DemoHelper.Dump($"kan_pred_{result.FileName}.json", Grid, Range, (a, b) =>
            {
                Tensor input = new Tensor([a, b], [1, 2]);
                return result.Model.Forward(input).GetData().Single();
            });
        }

        DemoHelper.Dump("kan_pred.json", Grid, Range, (a, b) =>
        {
            Tensor input = new Tensor([a, b], [1, 2]);
            return best.Model.Forward(input).GetData().Single();
        });
    }

    private static ExperimentResult Train(Experiment experiment)
    {
        KAN.KAN kan = experiment.CreateModel();
        Random trainingRandom = new(TrainingSeed);
        List<LossCheckpoint> history = [];
        Stopwatch stopwatch = Stopwatch.StartNew();

        for (int epoch = 1; epoch <= Epochs; epoch++)
        {
            for (int i = 0; i < DataSize; i++)
            {
                (Tensor input, Tensor gt) = CreateData(trainingRandom);

                Tensor o = kan.Forward(input);
                Tensor mse = Tensor.MSE(o, gt);

                mse.Backward();
                kan.Adjust(LearningRate);
                kan.Zero();
            }

            if (epoch == 1 || epoch % CheckpointInterval == 0 || epoch == Epochs)
            {
                float validationLoss = Evaluate(kan);
                history.Add(new LossCheckpoint(epoch, validationLoss));
                Console.WriteLine($"epoch {epoch,3}: validation loss {validationLoss:0.000000}");
            }
        }

        stopwatch.Stop();
        return new ExperimentResult(experiment.Name, experiment.FileName, kan, history[^1].Loss, stopwatch.ElapsedMilliseconds, history);
    }

    private static float Evaluate(KAN.KAN kan)
    {
        Random validationRandom = new(ValidationSeed);
        float loss = 0;

        for (int i = 0; i < ValidationSize; i++)
        {
            (Tensor input, Tensor gt) = CreateData(validationRandom);
            Tensor o = kan.Forward(input);
            Tensor mse = Tensor.MSE(o, gt);
            loss += mse.GetData().Single();
        }

        return loss / ValidationSize;
    }

    private static void PrintSummary(List<ExperimentResult> results)
    {
        Console.WriteLine("SUMMARY");
        Console.WriteLine("Basis              Epoch 1    Epoch 20   Epoch 60   Epoch 360  Time ms");

        foreach (ExperimentResult result in results.OrderBy(r => r.FinalLoss))
        {
            float epoch1 = FindLoss(result, 1);
            float epoch20 = FindLoss(result, 20);
            float epoch60 = FindLoss(result, 60);
            float epoch360 = FindLoss(result, 360);

            Console.WriteLine($"{result.Name,-18}{epoch1,10:0.0000}{epoch20,11:0.0000}{epoch60,11:0.0000}{epoch360,11:0.0000}{result.ElapsedMilliseconds,9}");
        }
    }

    private static float FindLoss(ExperimentResult result, int epoch)
    {
        return result.History.Single(h => h.Epoch == epoch).Loss;
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
        return MathF.Sin(3f * a) + 0.5f * MathF.Cos(5f * b) + MathF.Pow(a, 2f); 
    }

    private sealed record Experiment(string Name, string FileName, Func<KAN.KAN> CreateModel);

    private sealed record LossCheckpoint(int Epoch, float Loss);

    private sealed record ExperimentResult(
        string Name,
        string FileName,
        KAN.KAN Model,
        float FinalLoss,
        long ElapsedMilliseconds,
        List<LossCheckpoint> History);
}
