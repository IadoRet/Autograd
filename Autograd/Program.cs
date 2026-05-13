using Autograd;
using Autograd.Demos;

IDemo[] demos = [
    //new MlpDemo(),
    //new CnnDemo(),
    new KanDemo()
];

foreach (IDemo demo in demos)
{
    Console.WriteLine($"=== {demo.Name} ===");
    demo.Run();
    Console.WriteLine();
}
