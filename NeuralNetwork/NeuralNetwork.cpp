#include "NeuralNetwork.h"

#include <complex>
#include <iostream>
#include <cmath>
#include <fstream>


void NeuralNetwork::CreateNetwork(const FNetworkParams& InParams)
{
    FLayerParams InputLayerParams;
    InputLayerParams.bUseBias = InParams.bUseBiasInInputLayer;
    InputLayerParams.NumNeurons = InParams.NumInputs;
    InputLayer = Layer(InputLayerParams);

    FLayerParams HiddenLayerParams;
    HiddenLayerParams.bUseBias = InParams.bUseBiasInHiddenLayers;
    HiddenLayerParams.NumNeurons = InParams.NumHiddenLayers;
    for (unsigned i = 0; i < InParams.NumNeuronsPerHiddenLayer; ++i)
    {
        HiddenLayers.emplace_back(Layer(HiddenLayerParams));
    }

    FLayerParams OutputLayerParams;
    OutputLayerParams.bUseBias = InParams.bUseBiasInOutputLayer;
    OutputLayerParams.NumNeurons = InParams.NumOutputs;
    OutputLayer = Layer(OutputLayerParams);

    
}
