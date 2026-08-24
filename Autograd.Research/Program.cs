using Autograd.Research.Abstractions;
using Autograd.Research.Core;
using Autograd.Research.Experiments.KanFunctionApproximation;

ExperimentCatalog catalog = new(
[
    new KanFunctionApproximationExperiment(KanFunctionApproximationOptions.Default)
]);

if (args.Length == 0 || args[0].Equals("list", StringComparison.OrdinalIgnoreCase))
{
    Console.WriteLine("Available experiments:");
    foreach (IExperiment experiment in catalog.List())
        Console.WriteLine($"  {experiment.Id,-30} {experiment.Description}");

    return 0;
}

if (args.Length == 2 && args[0].Equals("run", StringComparison.OrdinalIgnoreCase))
{
    try
    {
        catalog.GetRequired(args[1]).Run(ExperimentContext.Console);
        return 0;
    }
    catch (KeyNotFoundException exception)
    {
        Console.Error.WriteLine(exception.Message);
        return 2;
    }
}

Console.Error.WriteLine("Usage:");
Console.Error.WriteLine("  Autograd.Research list");
Console.Error.WriteLine("  Autograd.Research run <experiment-id>");
return 1;
