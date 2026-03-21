namespace Autograd.Engine.Exceptions;

/// <summary>
/// Dimension mismatch exception
/// </summary>
public class TensorDimensionException(string message) : Exception(message);