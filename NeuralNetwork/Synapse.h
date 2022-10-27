#pragma once

class Neuron;

class Synapse
{
public:
    Synapse();
    Synapse(Neuron* NewStartNeuron, Neuron* NewEndNeuron, long double NewWeight);

    long double GetWeight() const;
    void SetWeight(const long double NewWeight);

    Neuron* GetStartNeuron() const;
    long double GetStartNeuronValue() const;
    void SetStartNeuron(Neuron* NewStartNeuron);
    
    Neuron* GetEndNeuron() const;
    void SetEndNeuron(Neuron* NewEndNeuron);
    
protected:
    virtual long double RandomWeight();

private:
    Neuron* StartNeuron;
    Neuron* EndNeuron;
    
    long double Weight;
};
