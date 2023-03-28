#pragma once

#include <vector>
#include "Synapse.h"


class Neuron
{
public:
    Neuron();
    virtual ~Neuron() = default;

    void AddInput(Neuron* NewInput);
    Neuron* GetInputNeuronAt(const unsigned& Index) const;
    Neuron* GetOutputNeuronAt(const unsigned& Index) const;

    
protected:
    void AddOutput(Neuron* NewOutput);

private:
    std::vector<Neuron*> Inputs;
    std::vector<Neuron*> Outputs;

    long double Value;
    long double Delta;
};
