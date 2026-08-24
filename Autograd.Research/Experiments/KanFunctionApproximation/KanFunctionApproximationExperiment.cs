using System.Diagnostics;
using System.Globalization;
using Autograd.Engine.Core;
using Autograd.Research.Abstractions;
using Autograd.Research.Core;

namespace Autograd.Research.Experiments.KanFunctionApproximation;

public sealed class KanFunctionApproximationExperiment : IExperiment
{
    private readonly KanFunctionApproximationOptions _options;
    private readonly IRegressionProblem _problem;

    public KanFunctionApproximationExperiment(KanFunctionApproximationOptions options)
        : this(options, new InteractionRegressionProblem())
    {
    }

    public KanFunctionApproximationExperiment(
        KanFunctionApproximationOptions options,
        IRegressionProblem problem)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(problem);
        ValidateOptions(options);
        _options = options;
        _problem = problem;
    }

    public string Id => "kan-function-approximation";

    public string Description => "Reproducible comparison of KAN bases and a parameter-matched MLP.";

    public ExperimentResult Run(ExperimentContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        DatasetSplit dataset = _problem.CreateDataset(_options.DatasetSeed, _options.DatasetSizes);
        PrintConfiguration(context.Output);

        List<ModelRunResult> runs = [];
        foreach (ModelSpecification specification in ModelSpecification.All)
        {
            foreach (int modelSeed in _options.ModelSeeds)
            {
                context.Output.WriteLine($"Training {specification.Name}, model seed {modelSeed}...");
                ModelRunResult run = Train(specification, modelSeed, dataset);
                runs.Add(run);
                context.Output.WriteLine(string.Create(
                    CultureInfo.InvariantCulture,
                    $"  validation MSE={run.ValidationMse:0.000000}, test MSE={run.TestMse:0.000000}, parameters={run.ParameterCount}, time={run.ElapsedMilliseconds} ms"));
            }
        }

        IReadOnlyList<ModelSummary> summary = Summarize(runs);
        PrintSummary(context.Output, summary);
        return new ExperimentResult(Id, runs, summary);
    }

    private ModelRunResult Train(
        ModelSpecification specification,
        int modelSeed,
        DatasetSplit dataset)
    {
        IRegressionModel model = specification.Create(modelSeed);
        int[] order = Enumerable.Range(0, dataset.Train.Count).ToArray();
        Random shuffle = new(_options.ShuffleSeed);
        List<LossCheckpoint> history = [];
        Stopwatch stopwatch = Stopwatch.StartNew();

        for (int epoch = 1; epoch <= _options.Epochs; epoch++)
        {
            Shuffle(order, shuffle);

            for (int start = 0; start < order.Length; start += _options.BatchSize)
            {
                int count = Math.Min(_options.BatchSize, order.Length - start);
                (Tensor inputs, Tensor outputs) = CreateBatch(dataset.Train, order, start, count);
                Tensor prediction = model.Forward(inputs);
                Tensor loss = Tensor.MSE(prediction, outputs);
                loss.Backward();
                model.Adjust(_options.LearningRate);
                model.Zero();
            }

            if (epoch == 1 || epoch % _options.ValidationInterval == 0 || epoch == _options.Epochs)
                history.Add(new LossCheckpoint(epoch, Evaluate(model, dataset.Validation)));
        }

        stopwatch.Stop();
        return new ModelRunResult(
            specification.Name,
            modelSeed,
            model.ParameterCount,
            history[^1].ValidationMse,
            Evaluate(model, dataset.Test),
            stopwatch.ElapsedMilliseconds,
            history);
    }

    private static (Tensor Inputs, Tensor Outputs) CreateBatch(
        RegressionDataset dataset,
        int[] order,
        int start,
        int count)
    {
        float[] inputs = new float[count * dataset.InputSize];
        float[] outputs = new float[count * dataset.OutputSize];

        for (int batchIndex = 0; batchIndex < count; batchIndex++)
        {
            int sourceIndex = order[start + batchIndex];
            Array.Copy(
                dataset.Inputs,
                sourceIndex * dataset.InputSize,
                inputs,
                batchIndex * dataset.InputSize,
                dataset.InputSize);
            Array.Copy(
                dataset.Outputs,
                sourceIndex * dataset.OutputSize,
                outputs,
                batchIndex * dataset.OutputSize,
                dataset.OutputSize);
        }

        return (
            new Tensor(inputs, [count, dataset.InputSize]),
            new Tensor(outputs, [count, dataset.OutputSize]));
    }

    private static float Evaluate(IRegressionModel model, RegressionDataset dataset)
    {
        Tensor prediction = model.Forward(dataset.InputTensor());
        return Tensor.MSE(prediction, dataset.OutputTensor()).GetData()[0];
    }

    private static void Shuffle(int[] values, Random random)
    {
        for (int i = values.Length - 1; i > 0; i--)
        {
            int other = random.Next(i + 1);
            (values[i], values[other]) = (values[other], values[i]);
        }
    }

    private static IReadOnlyList<ModelSummary> Summarize(IReadOnlyList<ModelRunResult> runs)
    {
        return ModelSpecification.All.Select(specification =>
        {
            ModelRunResult[] modelRuns = runs.Where(run => run.Model == specification.Name).ToArray();
            double meanMse = modelRuns.Average(run => run.TestMse);
            double variance = modelRuns.Average(run => Math.Pow(run.TestMse - meanMse, 2));

            return new ModelSummary(
                specification.Name,
                modelRuns[0].ParameterCount,
                meanMse,
                Math.Sqrt(variance),
                modelRuns.Average(run => run.ElapsedMilliseconds));
        }).ToArray();
    }

    private void PrintConfiguration(TextWriter output)
    {
        output.WriteLine($"Experiment: {Id}");
        output.WriteLine($"Problem: {_problem.Id}");
        output.WriteLine(string.Create(
            CultureInfo.InvariantCulture,
            $"Epochs={_options.Epochs}, batch={_options.BatchSize}, learning rate={_options.LearningRate}, validation interval={_options.ValidationInterval}"));
        output.WriteLine(
            $"Samples: train={_options.DatasetSizes.Train}, validation={_options.DatasetSizes.Validation}, test={_options.DatasetSizes.Test}");
        output.WriteLine(
            $"Seeds: dataset={_options.DatasetSeed}, shuffle={_options.ShuffleSeed}, models=[{string.Join(",", _options.ModelSeeds)}]");
        output.WriteLine();
    }

    private static void PrintSummary(TextWriter output, IReadOnlyList<ModelSummary> summary)
    {
        output.WriteLine();
        output.WriteLine("SUMMARY");
        output.WriteLine("Model                 Params   Test MSE mean       StdDev   Mean time ms");

        foreach (ModelSummary row in summary.OrderBy(row => row.MeanTestMse))
        {
            output.WriteLine(string.Create(
                CultureInfo.InvariantCulture,
                $"{row.Model,-21}{row.ParameterCount,7}{row.MeanTestMse,16:0.000000}{row.TestMseStandardDeviation,13:0.000000}{row.MeanElapsedMilliseconds,15:0}"));
        }
    }

    private static void ValidateOptions(KanFunctionApproximationOptions options)
    {
        if (options.Epochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Epoch count must be positive.");
        if (options.BatchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Batch size must be positive.");
        if (options.ValidationInterval <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "Validation interval must be positive.");
        if (!float.IsFinite(options.LearningRate) || options.LearningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(options), "Learning rate must be finite and positive.");
        if (options.ModelSeeds.Count == 0)
            throw new ArgumentException("At least one model seed is required.", nameof(options));
    }
}
