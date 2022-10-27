#include "Synapse.h"

#include <iostream>
#include <random>
#include <chrono>

#include "Neuron.h"

Synapse::Synapse()
{
    Weight = Synapse::RandomWeight();
    StartNeuron = nullptr;
    EndNeuron = nullptr;
}

Synapse::Synapse(Neuron* NewStartNeuron, Neuron* NewEndNeuron, long double NewWeight)
{
    StartNeuron = NewStartNeuron;
    EndNeuron = NewEndNeuron;
    Weight = NewWeight;
}

long double Synapse::GetWeight() const
{
    return Weight;
}

void Synapse::SetWeight(const long double NewWeight)
{
    Weight = NewWeight;
}

Neuron* Synapse::GetStartNeuron() const
{
    return StartNeuron;
}

long double Synapse::GetStartNeuronValue() const
{
    return StartNeuron->GetValue();
}

void Synapse::SetStartNeuron(Neuron* NewStartNeuron)
{
    StartNeuron = NewStartNeuron;
}

Neuron* Synapse::GetEndNeuron() const
{
    return EndNeuron;
}

void Synapse::SetEndNeuron(Neuron* NewEndNeuron)
{
    EndNeuron = NewEndNeuron;
}

long double Synapse::RandomWeight()
{
    const long long seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(static_cast<unsigned int>(seed));
    std::normal_distribution<long double> distribution(0.0, 1.0);
    return distribution(generator);
}
