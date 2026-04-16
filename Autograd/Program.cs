using Autograd.Demos;

IDemo[] demos = [new MlpDemo(), new CnnDemo()];

foreach (IDemo demo in demos)
{
    Console.WriteLine($"=== {demo.Name} ===");
    demo.Run();
    Console.WriteLine();
}
