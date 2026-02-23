// See https://aka.ms/new-console-template for more information

using Autograd.Engine.Core;

Console.WriteLine("Hello, World!");

Tensor t1 = new Tensor(new float[] { 2, 1, 1, 2 }, new int[] { 2, 2 });
Tensor t2 = new Tensor(new float[] { 0, 0, 0, 1 }, new int[] { 2, 2 });

Tensor t3 = t1 * t2;
Console.WriteLine(t3);
