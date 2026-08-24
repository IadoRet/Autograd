using Autograd.Research.Abstractions;

namespace Autograd.Research.Core;

public sealed class ExperimentCatalog
{
    private readonly IReadOnlyDictionary<string, IExperiment> _experiments;

    public ExperimentCatalog(IEnumerable<IExperiment> experiments)
    {
        ArgumentNullException.ThrowIfNull(experiments);
        _experiments = experiments.ToDictionary(experiment => experiment.Id, StringComparer.OrdinalIgnoreCase);
    }

    public IReadOnlyList<IExperiment> List()
    {
        return _experiments.Values.OrderBy(experiment => experiment.Id).ToArray();
    }

    public IExperiment GetRequired(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);

        if (_experiments.TryGetValue(id, out IExperiment? experiment))
            return experiment;

        throw new KeyNotFoundException($"Unknown experiment '{id}'. Use 'list' to see available experiments.");
    }
}
