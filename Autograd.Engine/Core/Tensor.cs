namespace Autograd.Engine.Core;

/// <summary>
/// A dense tensor that records the operations required for reverse-mode automatic differentiation.
/// </summary>
public partial class Tensor
{
    private readonly float[] _data;
    private readonly int[] _shape;
    private readonly float[] _gradients;
    private readonly Tensor[] _parents;

    private Action? _backward;

    public int ElementCount => _data.Length;

    /// <summary>
    /// An empty one-dimensional tensor.
    /// </summary>
    public static Tensor Empty => new([], [0]);

    /// <summary>
    /// Creates a tensor and takes copies of the supplied data and shape.
    /// An empty shape represents a scalar and therefore requires exactly one value.
    /// </summary>
    public Tensor(float[] data, int[] shape)
        : this(data, shape, [], takeOwnership: false)
    {
    }

    private Tensor(float[] data, int[] shape, Tensor[] parents, bool takeOwnership)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(shape);
        ArgumentNullException.ThrowIfNull(parents);

        ValidateElementCount(data.Length, shape);

        _data = takeOwnership ? data : data.ToArray();
        _shape = takeOwnership ? shape : shape.ToArray();
        _parents = parents;
        _gradients = new float[data.Length];
    }

    private static Tensor CreateOperation(float[] data, int[] shape, params Tensor[] parents)
    {
        return new Tensor(data, shape, parents, takeOwnership: true);
    }

    private static void ValidateElementCount(int dataLength, int[] shape)
    {
        long elementCount = 1;

        try
        {
            checked
            {
                foreach (int dimension in shape)
                {
                    if (dimension < 0)
                        throw new ArgumentOutOfRangeException(nameof(shape), "Tensor dimensions must be non-negative.");

                    elementCount *= dimension;
                }
            }
        }
        catch (OverflowException exception)
        {
            throw new ArgumentException("Tensor shape is too large.", nameof(shape), exception);
        }

        if (elementCount != dataLength)
        {
            throw new ArgumentException(
                $"Data length [{dataLength}] does not match the shape element count [{elementCount}].",
                nameof(shape));
        }
    }

    private static bool HaveSameShape(Tensor left, Tensor right)
    {
        return left._shape.AsSpan().SequenceEqual(right._shape);
    }

    /// <summary>
    /// Propagates an all-ones upstream gradient through the recorded graph.
    /// For a non-scalar root this is equivalent to differentiating its sum.
    /// </summary>
    public void Backward()
    {
        List<Tensor> topologicalOrder = [];
        HashSet<Tensor> visited = [];

        Visit(this);

        Array.Fill(_gradients, 1f);

        for (int i = topologicalOrder.Count - 1; i >= 0; i--)
            topologicalOrder[i]._backward?.Invoke();

        return;

        void Visit(Tensor tensor)
        {
            if (!visited.Add(tensor))
                return;

            foreach (Tensor parent in tensor._parents)
                Visit(parent);

            topologicalOrder.Add(tensor);
        }
    }

    /// <summary>
    /// Applies a gradient-descent update to this tensor.
    /// </summary>
    public void Adjust(float rate)
    {
        for (int i = 0; i < _gradients.Length; i++)
            _data[i] -= rate * _gradients[i];
    }

    /// <summary>
    /// Clears all accumulated gradients.
    /// </summary>
    public void Zero()
    {
        Array.Clear(_gradients);
    }

    public float[] GetData() => _data.ToArray();

    public float[] GetGradients() => _gradients.ToArray();

    public int[] GetShape() => _shape.ToArray();
}
