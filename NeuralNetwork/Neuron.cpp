#include "Neuron.h"

#include <iostream>
#include <cmath>
#include <vector>
#include "Synapse.h"

Neuron::Neuron()
{
    Value = 0;
    Inputs = std::vector<Synapse*>();
    Outputs = std::vector<Synapse*>();
}

Neuron::Neuron(std::vector<Synapse*> NewInputs, std::vector<Synapse*> NewOutputs, long double NewValue)
{
    Inputs = NewInputs;
    Outputs = NewOutputs;
    Value = NewValue;
}

void Neuron::AddInput(Neuron* NewInput)
{
    auto* s = new Synapse();
    s->SetStartNeuron(NewInput);
    s->SetEndNeuron(this);
    Inputs.push_back(s);
}

std::vector<Synapse*>* Neuron::GetInputs()
{
    return &Inputs;
}

void Neuron::AddOutput(Neuron* NewOutput)
{
    auto* s = new Synapse();
    s->SetStartNeuron(this);
    s->SetEndNeuron(NewOutput);
    Inputs.push_back(s);
    Outputs.push_back(s);
}

std::vector<Synapse*>* Neuron::GetOutputs()
{
    return &Outputs;
}

void Neuron::SetValue(const long double& NewValue)
{
    Value = NewValue;
}

long double Neuron::GetValue() const
{
    return Value;
}

long double Neuron::GetWeight(unsigned Index) const
{
    return Inputs[Index]->GetWeight();
}

void Neuron::SetWeight(const unsigned Index, const long double NewWeight)
{
    Inputs[Index]->SetWeight(NewWeight);
}

void Neuron::Fire()
{
    long double Temp_Value = CalculateNewValue();
    Value = Sigmoid(Temp_Value);
}

long double Neuron::CalculateNewValue() const
{
    long double NewValue = 0.0;
    
    for (int i = 0; i < Inputs.size(); i++)
    {
        NewValue += Inputs[i]->GetWeight() * Inputs[i]->GetStartNeuronValue();
    }
    
    return NewValue;
}

long double Neuron::Sigmoid(const long double x) const
{
    return 1 / (1 + std::exp(-x));
}
