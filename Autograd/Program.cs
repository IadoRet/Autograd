// See https://aka.ms/new-console-template for more information

using Autograd.Engine.Core;

Console.WriteLine("Hello, World!");

Tensor t1 = new Tensor(data: [2, 2, 2, 1, 1, 1], shape: [2, 1, 3]);
Tensor t2 = new Tensor(data: [0, 1, 0, 1, 1, 2, 3, 4, 1, 0, 0, 1, 0, 1, 0, 1, 1, 2, 3, 4, 1, 0, 0, 1], shape: [2, 3, 4]);

Tensor t3 = t1 * t2;
Console.WriteLine(t3);
