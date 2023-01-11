#include <iostream>
#include <string>

#include "NeuralNetwork.h"

int main()
{
    NeuralNetwork NN;

    NN.CreateNetwork(4, 1, 3, 3);


    std::cout << "Starting training 1" << std::endl;
    {
        std::vector<long double> Input1;
        Input1.push_back(1);
        Input1.push_back(0);
        Input1.push_back(1);
        Input1.push_back(0);
        std::vector<long double> Output1;
        Output1.push_back(1);

        FTrainingData Data;
        Data.Inputs = Input1;
        Data.Outputs = Output1;
        NN.SetTrainingData(Data);

        NN.Train(1000000);
    }

    std::cout << "Starting training 2" << std::endl;
    {
        std::vector<long double> Input2;
        Input2.push_back(1);
        Input2.push_back(1);
        Input2.push_back(0);
        Input2.push_back(0);
        std::vector<long double> Output2;
        Output2.push_back(0);

        FTrainingData Data;
        Data.Inputs = Input2;
        Data.Outputs = Output2;
        NN.SetTrainingData(Data);

        NN.Train(10000000);
    }

    std::cout << "Starting training 3" << std::endl;
    {
        std::vector<long double> Input3;
        Input3.push_back(0);
        Input3.push_back(0);
        Input3.push_back(1);
        Input3.push_back(1);
        std::vector<long double> Output3;
        Output3.push_back(1);

        FTrainingData Data;
        Data.Inputs = Input3;
        Data.Outputs = Output3;
        NN.SetTrainingData(Data);

        NN.Train(10000000);
    }

    std::cout << "Training complete" << std::endl;

    {
        std::vector<long double> Input4;
        Input4.push_back(0);
        Input4.push_back(1);
        Input4.push_back(0);
        Input4.push_back(1);
        std::vector<long double> Output4;
        Output4.push_back(1);

        FTrainingData Data;
        Data.Inputs = Input4;
        Data.Outputs = Output4;
        NN.SetTrainingData(Data);

        NN.Train(10000000);
    }

    std::vector<long double> Input;
    Input.push_back(0);
    Input.push_back(1);
    Input.push_back(1);
    Input.push_back(1);
    
    FTrainingData Data;
    Data.Inputs = Input;
    NN.SetTrainingData(Data);

    NN.Test(Input);
    
    const std::vector<long double> ActualOutputs = NN.GetOutputValues();
    std::cout << "Test Output " << ": " <<  ActualOutputs[0] << std::endl;
    
    return 0;
}
