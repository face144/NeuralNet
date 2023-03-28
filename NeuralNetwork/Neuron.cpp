#include "Neuron.h"

#include <iostream>
#include <cmath>
#include <vector>
#include "Synapse.h"

Neuron::Neuron()
{
    Value = 0;
    Delta = 0;
}

void Neuron::AddInput(Neuron* NewInput)
{
    Inputs.push_back(NewInput);
    NewInput->AddOutput(this);
}

Neuron* Neuron::GetInputNeuronAt(const unsigned& Index) const
{
    return Inputs[Index];
}

Neuron* Neuron::GetOutputNeuronAt(const unsigned& Index) const
{
    return Outputs[Index];
}

void Neuron::AddOutput(Neuron* NewOutput)
{
    Outputs.push_back(NewOutput);
}
