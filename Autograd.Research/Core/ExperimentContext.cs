namespace Autograd.Research.Core;

public sealed record ExperimentContext(TextWriter Output)
{
    public static ExperimentContext Console { get; } = new(System.Console.Out);
}
