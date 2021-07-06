#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <Proj/NeuralNetwork/Layers/LayerFactory.h>
#include <Proj/NeuralNetwork/LossFunctionFactory.h>
#include <Proj/NeuralNetwork/ActivationFunctionFactory.h>
#include <Proj/NeuralNetwork/Object.h>

#include <Proj/NeuralNetwork/Blobs/OriginalSample.h>
#include <Proj/NeuralNetwork/NetTelemetry.h>
#include <Proj/NeuralNetwork/Estimate.h>
#include <Proj/NeuralNetwork/TrainingTelemetry.h>


namespace NN {

enum class NetType{vanilla, convolutional};

class NeuralNetwork : public Object {
public:
    NeuralNetwork();
    NeuralNetwork(const std::string& filename);
    ~NeuralNetwork();
    
    struct TrainingOptions {
        int epoches=0;
        int batchSize=0;
        float learningRate=0;
        void saveStd(std::fstream &stream) const;
        void loadStd(std::fstream &stream);
        void print() const;
        bool isSet();
    };


    void train(TrainingOptions options);
    void train();
    void train(const int epochs, const int batchSize, const float learing_rate);
    bool connect();
    float RunOnce(bool print_estimate, DataBlob *blobX);
    void RunOnce();
    Estimate getEstimate(float *X);


    void addLayer(Layer *layer);

    void addTrainingData(NN::DataBlob* blob);
    void addTestingData(NN::DataBlob* blob);

    void SeedWeights(const float lower_bound=0.0f, const float upper_bound=0.01f);

    void SetLossFunction(LossFunction* function);
    void SetLossFunction(LossFunctionType type);
    void SetLossFunction(std::string type);

    void LoadNetFromFile();
    void RollBack();
    void CreateBackup();

    bool RunCommand(std::string command);
    bool LoadConfigFromFile(std::string filename);
    bool LoadConfigFromFile();
    void saveStd(const std::string& filename) const;
    void loadStd(const std::string& filename);
    void clear();


    NetTelemetry status;
    TrainingTelemetry training_telemetry;
    TranslationUnit translation;
    std::chrono::duration<double> training_duration;
    TrainingOptions options;

    void print() const;
    void printConfiguration() const;
    void printTrainingParameters() const;
    void printInputParameters() const;

    NN::DataBlob* trainBlob=nullptr;
    NN::DataBlob* testBlob=nullptr;

    int edge_length=0;
    std::string net_name="";

    int getNeuronsCount();
    int getWeightsCount();

    void ExecuteCommand(NetCommand &cmd) override;

    void SelectDataset(std::string which_one);
    DataBlob* GetCurrentDataset();
private:
    void RunMKL(const int no);
    void UpdateWeights(const int batchSize);
    void RunForward(float *X, float *WX,    std::vector<float*> &poolingMaps, float* Derivative=nullptr);
    void RunForward(float *X, float *WX);

    //calls LayerFactoryRegister to register layer available layer types in LayerFactory
    void RegisterLayers(); 
    void RegisterLossFunctions();
    void RegisterComponents();

    NN::DataBlob* currentDataset=nullptr;
    NN::Layer* currentLayer=nullptr;

    NN::LossFunction* lossFunc=nullptr;
    int weights_total=0;
    int neurons_total=0;
    int net_rows=0;
    int net_input_sz=0;
    int net_output_sz=0;
    float m_loss=0.0f;

    std::vector<Layer*> layers;
    DesignPatterns::Factory<NN::Layer, std::string, NN::Layer* (*)()> layerFactory;
    DesignPatterns::Factory<LossFunction, std::string, LossFunction* (*)()> lossFunctionFactory;
    Object* executioner = nullptr;
};

}


#endif // NEURALNETWORK_H
