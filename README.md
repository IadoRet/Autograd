# Autograd

MLP, CNN, and KAN implementations in .NET. Implemented mostly for learning and experimentation purposes.

## Features

- Classical MLP;
- Classical CNN;
- B-spline, Polynomial and Chebyshev KAN layers;
- Unit test coverage;
- A few visualizers for validating training results.

## Running

At the moment the console app runs the KAN demo by default. Other demos can be enabled in `Autograd/Program.cs`.
The project targets .NET 10.0.

```bash
dotnet run --project Autograd
```

```bash
dotnet test Autograd.slnx
```
