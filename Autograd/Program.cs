// See https://aka.ms/new-console-template for more information

using Autograd.Engine.Core;
using Autograd.MLP;

const int epochs = 2000;
const int dataSize = 50;

MLP mlp = MLP.Create(3)
             .WithLayer(16)
             .WithLayer(16)
             .WithOutput(1);

Random random = new (328);

float loss = 0;
for (int i = 0; i < epochs; i++)
{
    loss = 0;
    (Tensor input, Tensor gt)[] trainingData = Enumerable.Range(0, dataSize)
                                                         .Select(_ => CreateData(random))
                                                         .ToArray();
    
    foreach ((Tensor input, Tensor gt) in trainingData)
    {
        Tensor o = mlp.Forward(input);
        Tensor mse = Tensor.MSE(o, gt);
        loss += mse.GetData().Single();
        
        //todo: batch learning
        
        mse.Backward();
        mlp.Adjust(0.001f);
        mlp.Zero();
    }

    loss /= dataSize;
    Console.WriteLine($"EPOCH {i + 1}, LOSS: {loss}");
}

Console.WriteLine($"TRAINING FINISHED. LOSS: {loss}");

var example = CreateData(random);
var exampleOutput = mlp.Forward(example.input);
Console.ForegroundColor = ConsoleColor.Red;

Console.WriteLine($"EXAMPLE. INPUTS: {string.Join(',', example.input.GetData())}, PREDICTED: {exampleOutput.GetData().Single()}, GROUND TRUTH: {example.gt.GetData().Single()}");

(Tensor input, Tensor gt) CreateData(Random r)
{
    float a = r.NextSingle() * r.Next(-3, 3);
    float b = r.NextSingle() * r.Next(-7, 7);
    float c = r.NextSingle() * r.Next(-2, 2);

    float output = Fn(a, b, c);
    
    Tensor input = new Tensor([a, b, c], [1, 3]);
    Tensor gt = new Tensor([output], [1, 1]);

    return (input, gt);
}

float Fn(float a, float b, float c)
{
    return MathF.Pow(a, 3) + b + 3 * c;
}