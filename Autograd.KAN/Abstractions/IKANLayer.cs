using Autograd.Engine.Core;

namespace Autograd.KAN.Abstractions;

public interface IKANLayer
{
    int ParameterCount { get; }

    int GetOutputSize();
    
    Tensor Forward(Tensor input);
    
    void Zero();
    
    void Adjust(float learningRate);
}
