#pragma once
#include <vector>

#include "Neuron.h"

struct FLayerParams
{
    bool bUseBias = false;
    unsigned NumNeurons = 1;
};

class Layer
{
public:
    Layer();
    Layer(const FLayerParams& InLayerParams);
    Layer(Layer& InLayer);
    ~Layer() = default;

    void ConnectToLayer(Layer& Other);
    
    void AddNeuron();
    void AddNeuron(const Neuron& InNeuron);
    unsigned GetNeuronNum() const;

    bool IsBiasUsed() const;

    // Operators
    Layer& operator=(const Layer& Other);
    Neuron& operator[](unsigned i);

protected:

private:

    std::vector<Neuron> NeuronList;
    Neuron Bias;

    bool bIsBiasUsed;
};
