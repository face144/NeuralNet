#pragma once

#include <vector>
#include "Synapse.h"

class Neuron
{
public:
    Neuron();
    Neuron(std::vector<Synapse*> NewInputs, std::vector<Synapse*> NewOutputs, long double NewValue);

    void AddInput(Neuron* NewInput);
    std::vector<Synapse*>* GetInputs();

    void AddOutput(Neuron* NewOutput);
    std::vector<Synapse*>* GetOutputs();

    void SetValue(const long double& NewValue);
    long double GetValue() const;

    long double GetWeight(unsigned Index) const;
    void SetWeight(const unsigned Index, const long double NewWeight);

    virtual void Fire();
    
protected:

    virtual long double CalculateNewValue() const;

    long double Sigmoid(long double x) const;

private:
    std::vector<Synapse*> Inputs;
    std::vector<Synapse*> Outputs;

    long double Value;
    
};
