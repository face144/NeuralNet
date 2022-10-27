#include "NeuralNetwork.h"

#include <complex>
#include <iostream>
#include <cmath>

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

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::CreateNetwork()
{
    /// Add Neurons
    // Input layer
    for (unsigned i = 0; i < NumInputs; i++)
    {
        auto n = new Neuron();
        Neurons.push_back(n);
        InputLayer.push_back(n);
    }

    // Hidden layers
    for (unsigned i = 0; i < NumHiddenLayers; i++)
    {
        HiddenLayers.emplace_back(std::vector<Neuron*>());
        for (unsigned j = 0; j < NumHiddenNeurons; j++)
        {
            auto n = new Neuron();
            Neurons.push_back(n);
            HiddenLayers[i].push_back(n);
        }
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
    for (auto in : InputLayer) // Input neurons
    {
        for (auto hn : HiddenLayers[0]) // Hidden neurons
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
    for (auto on : OutputLayer)
    {
        for (auto hn : HiddenLayers.back())
        {
            on->AddInput(hn);
            hn->AddOutput(on);
        }
    }
}

void NeuralNetwork::Train(long double LearningRate)
{
    ForwardPropagation();
    BackwardsPropagation(LearningRate);
}

void NeuralNetwork::SetTrainingData(const ::FTrainingData& NewTrainingData)
{
    TrainingData.Inputs = NewTrainingData.Inputs;
    TrainingData.Outputs = NewTrainingData.Outputs;
}

std::vector<long double> NeuralNetwork::GetOutputValues() const
{
    std::vector<long double> Outputs;
    for (int i = 0; i < OutputLayer.size(); i++)
    {
        long double Output = OutputLayer[i]->GetValue();
        Outputs.push_back(Output);
    }

    return Outputs;
}

std::vector<long double> NeuralNetwork::GetWeights() const
{
    std::vector<long double> Weights;

    for (auto Neuron : Neurons)
    {
        for (int i = 0; i < Neuron->GetInputs()->size(); ++i)
        {
            Weights.push_back(Neuron->GetWeight(i));
        }
    }
    return Weights;
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
    for (int i = 0; i < OutputLayer.size(); ++i)
    {
        for (int j = 0; j < static_cast<int>(OutputLayer[i]->GetInputs()->size()); ++j)
        {
            long double OldWeight = OutputLayer[i]->GetWeight(j);
            long double ActualOutput = OutputLayer[i]->GetValue();
            long double Error = SquaredError(TrainingData.Outputs[0][0], ActualOutput);
            long double NewWeight = OldWeight - LearningRate * (Error / OldWeight);
            
            OutputLayer[i]->SetWeight(j, NewWeight);
            // std::cout << "Old weight: " << OldWeight << std::endl;
            // std::cout << "Actual output: " << ActualOutput << std::endl;
            // std::cout << "Expected output: " << TrainingData.Outputs[0][0] << std::endl;
            // std::cout << "Error: " << Error << std::endl;
            // std::cout << "New weight: " << NewWeight << std::endl;
        }
    }

    for (int i = 0; i < HiddenLayers.size(); ++i)
    {
        for (int h = 0; h < HiddenLayers[i].size(); ++h)
        {
            for (int j = 0; j < static_cast<int>(HiddenLayers[i][h]->GetInputs()->size()); ++j)
            {
                long double OldWeight = HiddenLayers[i][h]->GetWeight(j);
                long double ActualOutput = HiddenLayers[i][h]->GetValue();
                long double Error = SquaredError(TrainingData.Outputs[0][0], ActualOutput);
                long double NewWeight = OldWeight - LearningRate * (Error / OldWeight);
            
                HiddenLayers[i][h]->SetWeight(j, NewWeight);
                // std::cout << "Old weight: " << OldWeight << std::endl;
                // std::cout << "Actual output: " << ActualOutput << std::endl;
                // std::cout << "Expected output: " << TrainingData.Outputs[0][0] << std::endl;
                // std::cout << "Error: " << Error << std::endl;
                // std::cout << "New weight: " << NewWeight << std::endl;
            }
        }
    }
}

long double NeuralNetwork::SquaredError(const long double& ExpectedValue, const long double& ActualValue)
{
    long double SquaredError = std::pow(ExpectedValue - ActualValue, 2);
    
    return SquaredError;
}
