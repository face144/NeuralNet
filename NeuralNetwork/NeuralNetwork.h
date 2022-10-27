#pragma once

#include <vector>
#include "Neuron.h"

struct FTrainingData
{
    std::vector<std::vector<long double>> Inputs = std::vector<std::vector<long double>>();
    std::vector<std::vector<long double>> Outputs = std::vector<std::vector<long double>>();
};

class NeuralNetwork
{
public:
    NeuralNetwork();
    virtual ~NeuralNetwork();
    void CreateNetwork();
    void Train(long double LearningRate = 0.01);
    void SetTrainingData(const FTrainingData& NewTrainingData);
    std::vector<long double> GetOutputValues() const;
    std::vector<long double> GetWeights() const;
    
protected:
    std::vector<Neuron*>* GetInputLayer();
    std::vector<std::vector<Neuron*>>* GetHiddenLayers();
    std::vector<Neuron*>* GetOutputLayer();
    FTrainingData GetTrainingData() const;

    virtual void ForwardPropagation();
    virtual void BackwardsPropagation(long double LearningRate);

    static long double SquaredError(const long double& ActualValue, const long double& ExpectedValue);

private:
    unsigned NumInputs;
    unsigned NumHiddenLayers;
    unsigned NumHiddenNeurons;
    unsigned NumOutputs;
    
    std::vector<Neuron*> Neurons;

    std::vector<Neuron*> InputLayer;
    std::vector<std::vector<Neuron*>> HiddenLayers;
    std::vector<Neuron*> OutputLayer;

    FTrainingData TrainingData;
    
};
