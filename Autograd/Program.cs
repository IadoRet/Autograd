// See https://aka.ms/new-console-template for more information

using Autograd.Engine.Core;
using Autograd.MLP;

Console.WriteLine("Hello, World!");

Tensor t1 = new Tensor(data: [2, 2, 2, 1, 1, 1], shape: [2, 1, 3]);
Tensor t2 = new Tensor(data: [0, 1, 0, 1, 1, 2, 3, 4, 1, 0, 0, 1, 0, 1, 0, 1, 1, 2, 3, 4, 1, 0, 0, 1], shape: [2, 3, 4]);

Tensor t3 = t1 * t2;
Console.WriteLine(t3);

int epochs = 2000;
int dataSize = 50;

MLP mlp = MLP.Create(3).WithLayer(16).WithLayer(16).WithOutput(1);

Random random = new Random(314);

for (int i = 0; i < epochs; i++)
{
    float loss = 0;
    
    for (int j = 0; j < dataSize; j++)
    {
        (float[] parameters, float output) = CreateData(random);
        Tensor input = new Tensor(parameters, [1, parameters.Length]);
        Tensor gt = new Tensor([output], [1, 1]);
        Tensor o = mlp.Forward(input);
        Tensor mse = Tensor.MSE(o, gt);
        loss += mse.Raw()[0];
        mse.Backward();
        mlp.Adjust(0.001f);
        mlp.Zero();
    }

    loss /= dataSize;
    Console.WriteLine($"EPOCH {i + 1}, LOSS: {loss}");
}

(float[] parameters, float output) CreateData(Random r)
{
    float a = r.NextSingle() * r.Next(-3, 3);
    float b = r.NextSingle() * r.Next(-7, 7);
    float c = r.NextSingle() * r.Next(-2, 2);

    float output = Fn(a, b, c);

    return ([a, b, c], output);
}

float Fn(float a, float b, float c)
{
    return MathF.Pow(a, 3) + b + 3 * c;
}