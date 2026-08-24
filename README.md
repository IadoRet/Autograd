# Autograd

MLP, CNN, and KAN implementations in .NET. Implemented mostly for learning and experimentation purposes.

## Features

- Classical MLP;
- Classical CNN;
- B-spline, Polynomial and Chebyshev KAN layers;
- Unit test coverage;
- A few visualizers for validating training results.

## Running

The project targets .NET 10.0.

```bash
dotnet run --project Autograd
```

At the moment the console app runs the KAN demo by default. Other demos can be enabled in `Autograd/Program.cs`.

Reproducible, code-configured experiments live in the separate `Autograd.Research` executable. It prints configuration, per-seed metrics, and the aggregate comparison directly to the console.

```bash
dotnet run --project Autograd.Research -- list
dotnet run --project Autograd.Research -- run kan-function-approximation
```

```bash
dotnet test Autograd.slnx
```
