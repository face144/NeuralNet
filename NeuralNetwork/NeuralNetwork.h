#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "Layer.h"
#include "Neuron.h"


struct FNetworkParams
{
    unsigned NumInputs;
    unsigned NumHiddenLayers;
    unsigned NumNeuronsPerHiddenLayer;
    unsigned NumOutputs;

    bool bUseBiasInInputLayer;
    bool bUseBiasInHiddenLayers;
    bool bUseBiasInOutputLayer;
};

class NeuralNetwork
{
public:
    NeuralNetwork() = default;
    explicit NeuralNetwork(const FNetworkParams& InNetworkParams);
    virtual ~NeuralNetwork() = default;

    void CreateNetwork(const FNetworkParams& InParams);
    
protected:

private:
    Layer InputLayer;
    std::vector<Layer> HiddenLayers;
    Layer OutputLayer;
};
