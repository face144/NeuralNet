#include "NeuralNetwork.h"

#include <complex>
#include <iostream>
#include <cmath>
#include <fstream>

NeuralNetwork::NeuralNetwork()
{
    NumInputs = 3;
    NumHiddenLayers = 2;
    NumHiddenNeurons = 1;
    NumOutputs = 1;

    Neurons = std::vector<Neuron*>();
    InputLayer = std::vector<Neuron*>();
    HiddenLayers = std::vector<std::vector<Neuron*>>();
    OutputLayer = std::vector<Neuron*>();
}

void NeuralNetwork::CreateNetwork(const int& NumInputs, const int& NumOutputs, const int& NumHiddenLayers, const int& NumHiddenNeurons)
{
    this->NumInputs = NumInputs;
    this->NumHiddenLayers = NumHiddenLayers;
    this->NumHiddenNeurons = NumHiddenNeurons;
    this->NumOutputs = NumOutputs;
    
    /// Add Neurons
    // Input layer
    for (int i = 0; i < NumInputs; i++)
    {
        auto n = new Neuron();
        Neurons.push_back(n);
        InputLayer.push_back(n);
    }

    // Hidden layers
    for (int i = 0; i < NumHiddenLayers; i++)
    {
        // Todo : Revert if crashing
        // HiddenLayers.emplace_back(std::vector<Neuron*>());
        HiddenLayers.emplace_back();
        for (int j = 0; j < NumHiddenNeurons; j++)
        {
            auto n = new Neuron();
            Neurons.push_back(n);
            HiddenLayers[i].push_back(n);
        }
    }

    // Output layer
    for (int i = 0; i < NumOutputs; i++)
    {
        auto n = new Neuron();
        Neurons.push_back(n);
        OutputLayer.push_back(n);
    }

    
    /// Add synapses
    // Input layer
    for (const auto in : InputLayer) // Input neurons
    {
        for (const auto hn : HiddenLayers[0]) // Hidden neurons
        {
            in->AddOutput(hn);
            hn->AddInput(in);
        }
    }

    // Hidden layer
    for (int i = 0; i < static_cast<int>(HiddenLayers.size()); i++)
    {
        for (int j = 0; j < static_cast<int>(HiddenLayers[i].size()); j++)
        {
            if (i + 1 < static_cast<int>(HiddenLayers.size()))
            {
                for (int h = 0; h < static_cast<int>(HiddenLayers[i + 1].size()); h++)
                {
                    HiddenLayers[i + 1][h]->AddOutput(HiddenLayers[i][j]);
                    HiddenLayers[i][j]->AddInput(HiddenLayers[i + 1][h]);
                }
            }
        }
    }

    // Output layer
    for (const auto on : OutputLayer)
    {
        for (const auto hn : HiddenLayers.back())
        {
            on->AddInput(hn);
            hn->AddOutput(on);
        }
    }
}

void NeuralNetwork::CreateNetworkWithBias()
{
    /// Add Neurons
    // Input layer
    for (unsigned i = 0; i < NumInputs; i++)
    {
        auto n = new Neuron();
        Neurons.push_back(n);
        InputLayer.push_back(n);
    }
    // Bias
    const auto InputBias = new Neuron();
    InputBias->SetNeuronType(ENeuronType::Bias);
    Neurons.push_back(InputBias);
    InputLayer.push_back(InputBias);
    BiasNeurons.push_back(InputBias);

    // Hidden layers
    for (unsigned i = 0; i < NumHiddenLayers; i++)
    {
        // Todo : Revert if crashing
        // HiddenLayers.emplace_back(std::vector<Neuron*>());
        HiddenLayers.emplace_back();
        for (unsigned j = 0; j < NumHiddenNeurons; j++)
        {
            auto n = new Neuron();
            Neurons.push_back(n);
            HiddenLayers[i].push_back(n);
        }
        // Bias
        const auto HiddenBias = new Neuron();
        HiddenBias->SetNeuronType(ENeuronType::Bias);
        Neurons.push_back(HiddenBias);
        HiddenLayers[i].push_back(HiddenBias);
        BiasNeurons.push_back(HiddenBias);
    }

    // Output layer
    for (unsigned i = 0; i < NumOutputs; i++)
    {
        auto n = new Neuron();
        Neurons.push_back(n);
        OutputLayer.push_back(n);
    }

    
    /// Add synapses
    // Input layer
    for (const auto in : InputLayer) // Input neurons
    {
        for (const auto hn : HiddenLayers[0]) // Hidden neurons
        {
            in->AddOutput(hn);
            hn->AddInput(in);
        }
    }

    // Hidden layer
    for (int i = 0; i < static_cast<int>(HiddenLayers.size()); i++)
    {
        for (int j = 0; j < static_cast<int>(HiddenLayers[i].size()); j++)
        {
            for (int h = 0; h < static_cast<int>(HiddenLayers[i + 1].size()); h++)
            {
                HiddenLayers[i + 1][h]->AddOutput(HiddenLayers[i][j]);
                HiddenLayers[i][j]->AddInput(HiddenLayers[i + 1][h]);
            }
        }
    }

    // Output layer
    for (const auto on : OutputLayer)
    {
        for (const auto hn : HiddenLayers.back())
        {
            on->AddInput(hn);
            hn->AddOutput(on);
        }
    }
}

void NeuralNetwork::Train(const int& MaxIterations, long double LearningRate)
{
    for (size_t i = 0; i < InputLayer.size(); i++)
    {
        InputLayer[i]->SetValue(TrainingData.Inputs[i]);
    }
    
    OperationStartTime = std::chrono::steady_clock::now();
    
    for (int i = 0; i < MaxIterations; ++i)
    {
        ForwardPropagation();
        BackwardsPropagation(LearningRate);
    }
    
    OperationEndTime = std::chrono::steady_clock::now();
}

void NeuralNetwork::SetTrainingData(const ::FTrainingData& NewTrainingData)
{
    TrainingData.Inputs = NewTrainingData.Inputs;
    TrainingData.Outputs = NewTrainingData.Outputs;
}

void NeuralNetwork::Test(const std::vector<long double>& Inputs)
{
    for (size_t i = 0; i < InputLayer.size(); i++)
    {
        InputLayer[i]->SetValue(Inputs[i]);
    }
    ForwardPropagation();
}

float NeuralNetwork::GetTrainingTime(ETimeType TimeType) const
{
    switch (TimeType)
    {
    case ETimeType::Seconds:
        {
            const std::chrono::duration<float> TimeElapsed = OperationEndTime - OperationStartTime;
            return TimeElapsed.count();   
        }
        
    case ETimeType::Minutes:
        {
            const std::chrono::duration<float, std::ratio<60>> TimeElapsed = OperationEndTime - OperationStartTime;
            return TimeElapsed.count();
            
        }

    case ETimeType::Hours:
        {
            const std::chrono::duration<float, std::ratio<3600>> TimeElapsed = OperationEndTime - OperationStartTime;
            return TimeElapsed.count();
        }
    }
    
    return 0.f;
}

std::vector<long double> NeuralNetwork::GetOutputValues() const
{
    std::vector<long double> Outputs;
    for (auto i : OutputLayer)
    {
        long double Output = i->GetValue();
        Outputs.push_back(Output);
    }

    return Outputs;
}

std::vector<long double> NeuralNetwork::GetWeights() const
{
    std::vector<long double> Weights;

    for (const auto Neuron : Neurons)
    {
        for (size_t i = 0; i < Neuron->GetInputs()->size(); ++i)
        {
            Weights.push_back(Neuron->GetWeight(i));
        }
    }
    return Weights;
}

bool NeuralNetwork::Save(std::string Filename)
{
    const std::string Suffix = ".nnet";
    Filename.append(Suffix);
    
    std::ofstream ofs(Filename, std::ios::binary);
    ofs.write(reinterpret_cast<char*>(this), sizeof(*this));
    ofs.close();
}

bool NeuralNetwork::Load(std::string Filename)
{
    
}

std::vector<Neuron*>* NeuralNetwork::GetInputLayer()
{
    return &InputLayer;
}

std::vector<std::vector<Neuron*>>* NeuralNetwork::GetHiddenLayers()
{
    return &HiddenLayers;
}

std::vector<Neuron*>* NeuralNetwork::GetOutputLayer()
{
    return &OutputLayer;
}

FTrainingData NeuralNetwork::GetTrainingData() const
{
    return TrainingData;
}

void NeuralNetwork::ForwardPropagation()
{
    for (auto n : Neurons)
    {
        n->Fire();
    }
}

void NeuralNetwork::BackwardsPropagation(long double LearningRate)
{
    // Output layer 
    for (size_t i = 0; i < OutputLayer.size(); ++i)
    {
        for (size_t j = 0; j < OutputLayer[i]->GetInputs()->size(); ++j)
        {            
            const long double Error = OutputLayer[i]->GetValue() - TrainingData.Outputs[i];
            OutputLayer[i]->CalculateDelta(Error);
            
            const long double OldWeight = OutputLayer[i]->GetWeight(j);
            const long double NewWeight = OldWeight - LearningRate * (OutputLayer[i]->GetDelta() * OutputLayer[i]->GetInputAt(j)->GetValue());
            
            OutputLayer[i]->SetWeight(j, NewWeight);
        }
    }

    // Hidden layer
    for (auto& HiddenLayer : HiddenLayers)
    {
        for (size_t h = 0; h < HiddenLayer.size(); ++h)
        {
            for (size_t j = 0; j < HiddenLayer[h]->GetInputs()->size(); ++j)
            {
                long double Error = 0;
                Error += HiddenLayer[h]->GetInputAt(j)->GetDelta() * HiddenLayer[h]->GetInputAt(j)->GetWeight(h);
                HiddenLayer[h]->CalculateDelta(Error);

                const long double OldWeight = HiddenLayer[h]->GetWeight(j);
                const long double NewWeight = OldWeight - LearningRate * (HiddenLayer[h]->GetDelta() * HiddenLayer[h]->GetInputAt(j)->GetValue());
                HiddenLayer[h]->SetWeight(j, NewWeight);
            }
        }
    }
}

long double NeuralNetwork::SquaredError(const long double& ActualValue, const long double& ExpectedValue)
{
    return std::pow(ActualValue - ExpectedValue, 2);
}
