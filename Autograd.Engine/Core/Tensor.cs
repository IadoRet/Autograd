using Autograd.Engine.Exceptions;

namespace Autograd.Engine.Core;

public class Tensor
{
    private readonly float[] _data;
    
    private readonly int[] _shape;
    private readonly int[] _strides;

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
                throw new TensorException($"Dimensions do not match. Dimension: [{i}].");
            
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
            float[] t1Gradients = Multiply(o._gradients, t2._data, t1._shape, m, k, n, transposeT2: true);
            //T1^T * grad(C) -> k x n
            float[] t2Gradients = Multiply(t1._data, o._gradients, t2._shape, k, n, m, transposeT1: true);
            
            for (int i = 0; i < t1Gradients.Length; i++)
                t1._gradients[i] += t1Gradients[i];
            
            for (int i = 0; i < t2Gradients.Length; i++)
                t2._gradients[i] += t2Gradients[i];
        }
    }

    private static float[] Multiply(float[] t1, float[] t2, int[] shape, int m, int n, int k, bool transposeT1 = false, bool transposeT2 = false)
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
                        int t1Index = transposeT1 ?  offsetT1 + q * m + i : offsetT1 + i * k + q;
                        int t2Index = transposeT2 ? offsetT2 + j * k + q : offsetT2 + q * n + j;
                        result[offsetR + i * n + j] += t1[t1Index] * t2[t2Index];
                    }
                }
            }
        }

        return result;
    }
    
    public void Adjust(float rate)
    {
        for (int i = 0; i < _gradients.Length; i++)
            _data[i] -= rate * _gradients[i];
    }
}