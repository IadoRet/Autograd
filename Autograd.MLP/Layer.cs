using Autograd.Engine.Core;

namespace Autograd.MLP;

/// <summary>
/// MLP layer
/// </summary>
public class Layer
{
    /// <summary>
    /// Weights
    /// </summary>
    private readonly Tensor _w;
    
    /// <summary>
    /// Biases
    /// </summary>
    private readonly Tensor _b;
    
    /// <summary>
    /// Should activate? (false for output)
    /// </summary>
    private readonly bool _activate;
    
    /// <summary>
    /// Output size (amount of neurons)
    /// </summary>
    public int OutputSize { get; }

    public Layer(int inputSize, int outputSize, Random random, bool activate = true)
    {
        OutputSize = outputSize;
        _activate = activate;
        int size = inputSize * outputSize;
        float[] wData = new float[size];
        float[] bData = new float[outputSize];
        
        for (int i = 0; i < size; i++)
            wData[i] = random.NextSingle() * 2 - 1;
        
        _w = new Tensor(wData, [ inputSize, outputSize ]);
        _b = new Tensor(bData, [1, outputSize]);
    }

    /// <summary>
    /// Forward pass
    /// </summary>
    public Tensor Forward(Tensor input)
    {
        // x * w + b
        Tensor pass = input * _w + _b;

        return _activate ? Tensor.ReLU(pass) : pass;
    }

    /// <summary>
    /// Zero out gradients
    /// </summary>
    public void Zero()
    {
        _w.Zero();
        _b.Zero();
    }

    /// <summary>
    /// Adjust weights and biases according to gradients
    /// </summary>
    public void Adjust(float rate)
    {
        _w.Adjust(rate);
        _b.Adjust(rate);
    }
}