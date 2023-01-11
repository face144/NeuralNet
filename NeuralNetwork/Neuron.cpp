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
    Delta = 0;
    Type = ENeuronType::Normal;
}

Neuron::Neuron(const std::vector<Synapse*>& NewInputs, const std::vector<Synapse*>& NewOutputs, const long double& NewValue, const ENeuronType& NeuronType)
{
    Inputs = NewInputs;
    Outputs = NewOutputs;
    Value = NewValue;
    Delta = 0;
    Type = NeuronType;
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

Neuron* Neuron::GetInputAt(const int& Index) const
{
    return Inputs[Index]->GetStartNeuron();
}

Neuron* Neuron::GetInputAt(const size_t& Index) const
{
    return Inputs[Index]->GetStartNeuron();
}

Neuron* Neuron::GetOutputAt(const int& Index) const
{
    return Outputs[Index]->GetEndNeuron();
}

Neuron* Neuron::GetOutputAt(const size_t& Index) const
{
    return Outputs[Index]->GetEndNeuron();
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

long double Neuron::GetWeight(const unsigned& Index) const
{
    return Inputs[Index]->GetWeight();
}

long double Neuron::GetWeight(const size_t& Index) const
{
    return Inputs[Index]->GetWeight();
}

void Neuron::SetWeight(const unsigned& Index, const long double& NewWeight)
{
    Inputs[Index]->SetWeight(NewWeight);
}

void Neuron::SetWeight(const size_t& Index, const long double& NewWeight)
{
    Inputs[Index]->SetWeight(NewWeight);
}

long double Neuron::GetDelta() const
{
    return Delta;
}

void Neuron::SetDelta(const long double& NewDelta)
{
    Delta = NewDelta;
}

ENeuronType Neuron::GetNeuronType() const
{
    return Type;
}

void Neuron::SetNeuronType(const ENeuronType& NewNeuronType)
{
    Type = NewNeuronType;
}

void Neuron::Fire()
{
    const long double Temp_Value = CalculateNewValue();
    Value = Sigmoid(Temp_Value);
}

long double Neuron::CalculateNewValue() const
{
    long double NewValue = 0.0;
    
    for (const auto Input : Inputs)
    {
        NewValue += Input->GetWeight() * Input->GetStartNeuronValue();
    }
    
    return NewValue;
}

void Neuron::CalculateDelta(const long double& Error)
{
    Delta = Error * SigmoidDerivative(Value);
}

long double Neuron::Sigmoid(const long double& x) const
{
    return 1 / (1 + std::exp(-x));
}

long double Neuron::SigmoidDerivative(const long double& x) const
{
    return x * (1 - x);
}
