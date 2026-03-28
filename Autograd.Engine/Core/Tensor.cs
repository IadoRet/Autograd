using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Core;

/// <summary>
/// Tensor
/// </summary>
public class Tensor
{
    private readonly float[] _data;
    
    private readonly int[] _shape;
    private readonly int[] _strides; //todo: remove?

    private readonly float[] _gradients;
    
    private readonly Tensor[] _leaves;
    
    private Action? _backward;

    public Tensor(float[] data, int[] shape, params Span<Tensor> leaves)
    {
        _data = data;
        _shape = shape;
        _leaves = leaves.ToArray();
        _backward = null;
        _gradients = new float[_data.Length];
        _strides = new int[shape.Length];
        _strides[shape.Length - 1] = 1;
        for (int i = shape.Length - 2; i >= 0; i--)
            _strides[i] = _strides[i + 1] * shape[i + 1];
    }

    /// <summary>
    /// Matrix multiplication
    /// </summary>
    /// <param name="t1">tensor 1</param>
    /// <param name="t2">tensor 2</param>
    /// <exception cref="TensorDimensionException">Dimension mismatch</exception>
    public static Tensor operator *(Tensor t1, Tensor t2)
    {
        // Dimensions: 
        // T1: m x k, T2: k x n, o: m x n, T1^T: k x m, T2^T: n x k
        
        int m = t1._shape[^2];
        int n = t2._shape[^1];
        int k = t1._shape[^1];
        
        int[] shape = new int[t1._shape.Length];
        for (int i = 0; i < t1._shape.Length - 2; i++)
        {
            if (t1._shape[i] != t2._shape[i])
                throw new TensorDimensionException($"Dimensions do not match. Dimension: [{i}].");
            
            shape[i] = t1._shape[i];
        }
            
        shape[^2] = m;
        shape[^1] = n;
        
        float[] data = Multiply(t1._data, t2._data, shape, m, n, k);
        
        Tensor o = new (data, shape, t1, t2);
        
        o._backward = Backward;
        
        return o;
        
        void Backward()
        {
            //grad(C) * T2^T -> m x k
            float[] t1Gradients = Multiply(o._gradients, t2._data, t1._shape, m, k, n, tposeT2: true);
            //T1^T * grad(C) -> k x n
            float[] t2Gradients = Multiply(t1._data, o._gradients, t2._shape, k, n, m, tposeT1: true);
            
            for (int i = 0; i < t1Gradients.Length; i++)
                t1._gradients[i] += t1Gradients[i];
            
            for (int i = 0; i < t2Gradients.Length; i++)
                t2._gradients[i] += t2Gradients[i];
        }
    }

    /// <summary>
    /// Matrix multiplication
    /// </summary>
    /// <param name="t1">tensor 1</param>
    /// <param name="t2">tensor 2</param>
    /// <param name="shape">shape (should be identical)</param>
    /// <param name="m">m</param>
    /// <param name="n">n</param>
    /// <param name="k">k</param>
    /// <param name="tposeT1">transpose tensor 1</param>
    /// <param name="tposeT2">transpose tensor 2</param>
    // ReSharper disable once InconsistentNaming
    private static float[] Multiply(float[] t1, float[] t2, int[] shape, int m, int n, int k, bool tposeT1 = false, bool tposeT2 = false)
    {
        int dim = shape.Aggregate(1, (a, b) => a * b);
        float[] result = new float[dim];
        int batches = 1;
        for (int i = 0; i < shape.Length - 2; i++)
            batches *= shape[i];

        for (int b = 0; b < batches; b++)
        {
            int offsetT1 = b * m * k;
            int offsetT2 = b * k * n;
            int offsetR = b * m * n;

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    for (int q = 0; q < k; q++)
                    {
                        int t1Index = tposeT1 ?  offsetT1 + q * m + i : offsetT1 + i * k + q;
                        int t2Index = tposeT2 ? offsetT2 + j * k + q : offsetT2 + q * n + j;
                        result[offsetR + i * n + j] += t1[t1Index] * t2[t2Index];
                    }
                }
            }
        }

        return result;
    }
    
    /// <summary>
    /// Matrix additions
    /// </summary>
    /// <param name="t1">tensor 1</param>
    /// <param name="t2">tensor 2</param>
    /// <exception cref="TensorDimensionException">Dimension mismatch</exception>
    public static Tensor operator+(Tensor t1, Tensor t2)
    {
        int len = Math.Max(t1._shape.Length, t2._shape.Length);
        
        //Matching shapes [3, 5] + [5] => [3, 5] + [1, 5]
        Span<int> t1Shape = stackalloc int[len];
        Span<int> t2Shape = stackalloc int[len];

        int[] t1Strides = new int[len];
        int[] t2Strides = new int[len];
        
        int[] shape = new int[len];
        
        int dim = 1;

        for (int i = shape.Length - 1; i >= 0; i--)
        {
            int i1 = i - (shape.Length - t1._shape.Length);
            int s1 = i1 < 0 ? 1 : t1._shape[i1];
            t1Shape[i] = s1;
            int i2 = i - (shape.Length - t2._shape.Length);
            int s2 = i2 < 0 ? 1 : t2._shape[i2];
            t2Shape[i] = s2;

            if (i < shape.Length - 1)
            {
                t1Strides[i] = t1Strides[i + 1] * t1Shape[i + 1];
                t2Strides[i] = t2Strides[i + 1] * t2Shape[i + 1];
            }
            else
            {
                t1Strides[i] = 1;
                t2Strides[i] = 1;
            }

            // invalid operation, for example [5, 4] + [1, 5] (but [5, 4] + [1, 4] - valid).
            if (s1 != s2 && (s1 != 1 || s2 != 1))
                throw new TensorDimensionException($"Dimensions do not match. Dimension: [{i}]. Shapes: [{s1}] <=> [{s2}]");
            
            int max = Math.Max(s1, s2);
            shape[i] = max;
            dim *= max;
        }

        for (int i = 0; i < len; i++)
        {
            t1Strides[i] = t1Shape[i] == 1 ? 0 : t1Strides[i];
            t2Strides[i] = t2Shape[i] == 1 ? 0 : t2Strides[i];
        }

        float[] result = new float[dim];

        for (int i = 0; i < dim; i++)
        {
            int c = i;
            int c1 = 0;
            int c2 = 0;
            
            for (int j = len - 1; j >= 0; j--)
            {
                int jShape = shape[j];
                int coord = c % jShape;
                c /= jShape;
                c1 += coord * t1Strides[j];
                c2 += coord * t2Strides[j];
            }
            
            result[i] = t1._data[c1] + t2._data[c2];
        }

        Tensor o = new(result, shape, t1, t2);
        
        o._backward = Backward;

        return o;

        void Backward()
        {
            for (int i = 0; i < dim; i++)
            {
                int c = i;
                int c1 = 0;
                int c2 = 0;
    
                for (int j = len - 1; j >= 0; j--)
                {
                    int coord = c % shape[j];
                    c /= shape[j];
                    c1 += coord * t1Strides[j];
                    c2 += coord * t2Strides[j];
                }
    
                t1._gradients[c1] += o._gradients[i];
                t2._gradients[c2] += o._gradients[i];
            }
        }
    }

    /// <summary>
    /// Rectified Linear Unit
    /// </summary>
    /// <param name="t">tensor</param>
    public static Tensor ReLU(Tensor t)
    {
        int len = t._data.Length;
        
        float[] result = new float[len];
        
        for (int i = 0; i < len; i++)
            result[i] = Math.Max(t._data[i], 0);
        
        // copy _shape
        Tensor o = new(result, t._shape.ToArray(), t);
        
        o._backward = Backward;

        return o;

        void Backward()
        {
            for (int i = 0; i < t._gradients.Length; i++)
                t._gradients[i] += t._data[i] > 0 ? o._gradients[i] : 0;
        }
    }

    /// <summary>
    /// Hyperbolic tangent
    /// </summary>
    /// <param name="t">tensor</param>
    public static Tensor TanH(Tensor t)
    {
        int len = t._data.Length;
        float[] result = new float[len];

        for (int i = 0; i < len; i++)
            result[i] = MathF.Tanh(t._data[i]);
        
        Tensor o = new(result, t._shape.ToArray(), t);
        
        o._backward = Backward;

        return o;

        void Backward()
        {
            for (int i = 0; i < len; i++)
            {
                float v = o._data[i];
                t._gradients[i] += (1 - v * v) * o._gradients[i];
            }
        }
    }
    
    /// <summary>
    /// Mean square error
    /// </summary>
    /// <param name="p">prediction</param>
    /// <param name="gt">ground truth</param>
    /// <exception cref="TensorDimensionException">Dimension mismatch</exception>
    // ReSharper disable once InconsistentNaming
    public static Tensor MSE(Tensor p, Tensor gt)
    {
        if (gt._data.Length != p._data.Length)
            throw new TensorDimensionException("Dimensions do not match.");

        float mean = 0;
        int len = p._data.Length;
        
        for (int i = 0; i < len; i++)
        {
            float pv = p._data[i];
            float gtv = gt._data[i];

            mean += MathF.Pow(pv - gtv, 2);
        }

        Tensor o = new([mean / len], [1], p, gt);
        
        o._backward = Backward;

        return o;

        void Backward()
        {
            float scale = o._gradients[0] * 2f / len;
    
            for (int i = 0; i < len; i++)
            {
                float diff = p._data[i] - gt._data[i];
                p._gradients[i] += scale * diff;
                gt._gradients[i] -= scale * diff;
            }
        }
    }

    /// <summary>
    /// Backpropagation
    /// </summary>
    public void Backward()
    {
        List<Tensor> topo = [];
        HashSet<Tensor> visited = [];
        
        TopoSort(this);
        
        for (int i = 0; i < _gradients.Length; i++)
            _gradients[i] = 1;
        
        for (int i = topo.Count - 1; i >= 0; i--)
            topo[i]._backward?.Invoke();

        return;
        
        // topologic sort
        void TopoSort(Tensor value)
        {
            if (!visited.Add(value)) 
                return;
            
            foreach (Tensor leaf in value._leaves)
                TopoSort(leaf);
            
            topo.Add(value);
        }
    }
    
    /// <summary>
    /// Adjust values according to gradients
    /// </summary>
    public void Adjust(float rate)
    {
        for (int i = 0; i < _gradients.Length; i++)
            _data[i] -= rate * _gradients[i];
    }

    /// <summary>
    /// Zero out gradients
    /// </summary>
    public void Zero()
    {
        for (int i = 0; i < _gradients.Length; i++)
            _gradients[i] = 0;
    }

    // Copy tensor data
    public float[] GetData() => _data.ToArray();

    // Copy tensor gradients
    public float[] GetGradients() => _gradients.ToArray();
}