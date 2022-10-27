#include <iostream>
#include <string>

#include "NeuralNetwork.h"

int main()
{
    NeuralNetwork NN;

    NN.CreateNetwork();

    std::vector<std::vector<long double>> Inputs;
    Inputs.push_back(std::vector<long double>());
    Inputs[0].push_back(1);
    Inputs[0].push_back(0);
    Inputs[0].push_back(1);

    std::vector<std::vector<long double>> Outputs;
    Outputs.push_back(std::vector<long double>());
    Outputs[0].push_back(0);
    
    FTrainingData Data;
    Data.Inputs = Inputs;
    Data.Outputs = Outputs;
    NN.SetTrainingData(Data);

    for (auto weight : NN.GetWeights())
    {
        std::cout << "Weight: " << weight << std::endl;
    }
    
    for (int i = 0; i < 50000; ++i)
    {
        NN.Train();

        std::vector<long double> ActualOutputs = NN.GetOutputValues();

        std::cout << "Output " << i << ": " <<  ActualOutputs[0] << std::endl;
    }

    for (auto weight : NN.GetWeights())
    {
        std::cout << "Weight: " << std::to_string(weight) << std::endl;
    }

    system("pause");
    
    return 0;
}