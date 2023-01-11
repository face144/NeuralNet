#pragma once

#include <vector>
#include "Synapse.h"

enum class ENeuronType
{
    Normal,
    Bias
};

class Neuron
{
public:
    Neuron();
    Neuron(const std::vector<Synapse*>& NewInputs, const std::vector<Synapse*>& NewOutputs, const long double& NewValue, const ENeuronType& NeuronType);
    virtual ~Neuron() = delete;

    void AddInput(Neuron* NewInput);
    std::vector<Synapse*>* GetInputs();
    Neuron* GetInputAt(const int& Index) const;
    Neuron* GetInputAt(const size_t& Index) const;

    void AddOutput(Neuron* NewOutput);
    std::vector<Synapse*>* GetOutputs();
    Neuron* GetOutputAt(const int& Index) const;
    Neuron* GetOutputAt(const size_t& Index) const;

    void SetValue(const long double& NewValue);
    long double GetValue() const;

    long double GetWeight(const unsigned& Index) const;
    long double GetWeight(const size_t& Index) const;
    void SetWeight(const unsigned& Index, const long double& NewWeight);
    void SetWeight(const size_t& Index, const long double& NewWeight);

    long double GetDelta() const;
    void SetDelta(const long double& NewDelta);

    ENeuronType GetNeuronType() const;
    void SetNeuronType(const ENeuronType& NewNeuronType);

    virtual void Fire();
    
    void CalculateDelta(const long double& Error);
protected:

    virtual long double CalculateNewValue() const;
    
    long double Sigmoid(const long double& x) const;
    long double SigmoidDerivative(const long double& x) const;

private:
    std::vector<Synapse*> Inputs;
    std::vector<Synapse*> Outputs;

    long double Value;
    long double Delta;

    ENeuronType Type;
};
