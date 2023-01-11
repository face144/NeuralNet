#pragma once

#include <chrono>
#include <string>
#include <vector>
#include "Neuron.h"

namespace std
{
    namespace chrono
    {
        struct steady_clock;
    }
}

enum class ETimeType 
{
    Seconds,
    Minutes,
    Hours
};

struct FTrainingData
{
    std::vector<long double> Inputs = std::vector<long double>();
    std::vector<long double> Outputs = std::vector<long double>();
};

class NeuralNetwork
{
public:
    NeuralNetwork();
    virtual ~NeuralNetwork() = default;
    
    void CreateNetwork(const int& NumInputs, const int& NumOutputs, const int& NumHiddenLayers, const int& NumHiddenNeurons);
    void CreateNetworkWithBias();
    
    void Train(const int& MaxIterations = 1, long double LearningRate = 0.01);
    void SetTrainingData(const FTrainingData& NewTrainingData);

    void Test(const std::vector<long double>& Inputs);

    float GetTrainingTime(ETimeType TimeType = ETimeType::Seconds) const;
    
    std::vector<long double> GetOutputValues() const;
    std::vector<long double> GetWeights() const;

    bool Save(std::string Filename);
    bool Load(std::string Filename);
    
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
    std::vector<Neuron*> BiasNeurons;

    FTrainingData TrainingData;

    std::chrono::steady_clock::time_point OperationStartTime;
    std::chrono::steady_clock::time_point OperationEndTime;
};
