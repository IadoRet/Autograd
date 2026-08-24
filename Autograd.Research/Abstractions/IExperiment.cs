using Autograd.Research.Core;

namespace Autograd.Research.Abstractions;

public interface IExperiment
{
    string Id { get; }

    string Description { get; }

    ExperimentResult Run(ExperimentContext context);
}
