// SmallNN.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include <Common.h>
#include <NeuralNetwork/NeuralNetwork.h>
#include <NeuralNetwork/NetState.h>


int main()
{
    std::string command = "";
    NN::NeuralNetwork net;

 /*   NN::NetState* state = net.GetState();
    NN::Layer* CL1 = new NN::ConvolutionalLayer(8, 4, 4, 2, NN::ActivationFunctionType::RELU);
    CL1->SetMaximumWeightChange(0.1f);
    net.GetState()->AddLayer(&net, CL1);

    NN::Layer* PL1 = new NN::PoolingLayer(2, 2, 2);
    net.GetState()->AddLayer(&net, PL1);

    NN::Layer* CL2 = new NN::ConvolutionalLayer(8, 3, 3, 1, NN::ActivationFunctionType::RELU);
    CL2->SetMaximumWeightChange(0.1f);
    net.GetState()->AddLayer(&net, CL2);

    NN::ActivationFunction* a1 = new NN::LogisticActivationFunction;
    NN::Layer* FCL1 = new NN::Layer(56, a1);
    net.GetState()->AddLayer(&net, FCL1);

    NN::DataBlob* TrainDB = new NN::DataBlob("K:\\Blobs\\BlobTrain.blob", "K:\\TranslationUnits\\CharTranslation.trun");
    TrainDB->resize(22, true);
    TrainDB->transform(NN::TransformationType::sin);
    net.GetState()->AddTrainData(&net, TrainDB);


    NN::DataBlob* TestDB = new NN::DataBlob("K:\\Blobs\\BlobTest.blob", "K:\\TranslationUnits\\CharTranslation.trun");
    TestDB->resize(22, true);
    TestDB->transform(NN::TransformationType::sin);
    net.GetState()->AddTestData(&net, TestDB);

    NN::LossFunction* LF = new NN::MultiLabelCrossEntropyLoss;
    net.GetState()->SetLossFunction(&net, LF);

    net.GetState()->SeedWeights(&net, -0.2f, 0.2f);

    net.GetState()->SetTrainingOptions(&net, 10, 200, 0.005f);

    net.GetState()->Train(&net);
    */
    while (command != "exit") {
        std::getline(std::cin, command);
        net.RunCommand(command);
    }
    std::cout << "Exited\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
