#include "Layer.h"

Layer::Layer()
{
    bIsBiasUsed = false;
}

Layer::Layer(const FLayerParams& InLayerParams)
{
    for (unsigned i = 0; i < InLayerParams.NumNeurons; ++i)
    {
        NeuronList.push_back(Neuron());
    }

    bIsBiasUsed = InLayerParams.bUseBias;
}

Layer::Layer(Layer& InLayer)
{
    NeuronList = InLayer.NeuronList;
    bIsBiasUsed = InLayer.IsBiasUsed();
}

void Layer::ConnectToLayer(Layer& Other)
{
    for (unsigned i = 0; i < Other.GetNeuronNum(); ++i)
    {
        Other[i].AddInput(&Other[i]);
    }
}

void Layer::AddNeuron()
{
    NeuronList.emplace_back(Neuron());
}

void Layer::AddNeuron(const Neuron& InNeuron)
{
    NeuronList.push_back(InNeuron);
}

unsigned Layer::GetNeuronNum() const
{
    return static_cast<unsigned>(NeuronList.size());
}

bool Layer::IsBiasUsed() const
{
    return bIsBiasUsed;
}

Layer& Layer::operator=(const Layer& Other)
{
    NeuronList = Other.NeuronList;
    bIsBiasUsed = Other.IsBiasUsed();
    return *this;
}

Neuron& Layer::operator[](unsigned i)
{
    return NeuronList.at(i);
}
