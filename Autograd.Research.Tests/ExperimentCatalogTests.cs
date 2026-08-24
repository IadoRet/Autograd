using Autograd.Research.Abstractions;
using Autograd.Research.Core;

namespace Autograd.Research.Tests;

public class ExperimentCatalogTests
{
    [Fact]
    public void Catalog_ListsExperimentsInStableOrderAndFindsById()
    {
        var second = new StubExperiment("second");
        var first = new StubExperiment("first");
        var catalog = new ExperimentCatalog([second, first]);

        Assert.Equal(["first", "second"], catalog.List().Select(experiment => experiment.Id));
        Assert.Same(first, catalog.GetRequired("FIRST"));
    }

    [Fact]
    public void Catalog_UnknownIdThrowsKeyNotFoundException()
    {
        var catalog = new ExperimentCatalog([]);

        Assert.Throws<KeyNotFoundException>(() => catalog.GetRequired("missing"));
    }

    private sealed class StubExperiment(string id) : IExperiment
    {
        public string Id => id;

        public string Description => id;

        public ExperimentResult Run(ExperimentContext context) => new(Id, [], []);
    }
}
