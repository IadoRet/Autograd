using Autograd.Engine.Core;

namespace Autograd.MLP;

public class Layer
{
    private readonly Tensor _w;
    private readonly Tensor _b;
    
    private readonly bool _activate;
    
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

    public Tensor Forward(Tensor input)
    {
        Tensor pass = input * _w + _b;

        return _activate ? Tensor.ReLU(pass) : pass;
    }

    public void Zero()
    {
        _w.Zero();
        _b.Zero();
    }

    public void Adjust(float rate)
    {
        _w.Adjust(rate);
        _b.Adjust(rate);
    }
}