#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <NeuralNetwork/Layers/LayerFactory.h>
#include <NeuralNetwork/LossFunctionFactory.h>
#include <NeuralNetwork/ActivationFunctionFactory.h>
#include <NeuralNetwork/Object.h>

#include <NeuralNetwork/Blobs/OriginalSample.h>
#include <NeuralNetwork/NetTelemetry.h>
#include <NeuralNetwork/Estimate.h>
#include <NeuralNetwork/TrainingTelemetry.h>



namespace NN {

    class NetState;

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

    /*Trains net*/
    void SetTrainingOptions(const int epochs, const int batchSize, const float learing_rate);
    void train();

    /*reverse last training weights update*/
    void RollBack();

    /*connecting all layers and blobs, this function is nessessary to seed weights and train/run net*/
    bool connect();

    /*forward pass for set datablob, first variable is responsible for console output*/
    float RunOnce(bool print_estimate, DataBlob *blobX);
    /*runs on current datablob, chosen with SelectDataset function*/
    void RunOnce();
    /*predicts estimate for provided sample. 
    Performs autoresizing via padding which can cause incorrect predictions if data was resized normally (not just padded to fit)
    TODO: refactor to take only properly sized input of type Sample*/
    Estimate getEstimate(float *X);

    /*push layers to the net, allows only consequential connections*/
    /*TODO: connect layers vith graph structure*/
    void addLayer(Layer *layer);

    /*adds respective datablobs*/
    void addTrainingData(NN::DataBlob* blob);
    void addTestingData(NN::DataBlob* blob);

    /*seeds weights within lower and upper bounds, seeding is performed with uniform distribution*/
    /*TODO: add other types of distributions*/
    void SeedWeights(const float lower_bound=0.0f, const float upper_bound=0.01f);
    
    /*Sets loss function to the net*/
    void SetLossFunction(LossFunction* function);

    /*Provides file dialog to chose net from file, Windows only*/
    void LoadNetFromFile();
    /*loads net from filename*/
    void loadStd(const std::string& filename);
    /*save net to file*/
    void saveStd(const std::string& filename) const;

    /*configures empty net from file*/
    bool LoadConfigFromFile(std::string filename);
    bool LoadConfigFromFile();

    /*string type commands*/
    void RunCommand(std::string command);
    /*runs command in cmd type, derived from pure virtual Object class*/
    void ExecuteCommand(NetCommand& cmd) override;

    void print() const;
    void printConfiguration() const;
    void printTrainingParameters() const;
    void printInputParameters() const;

    /*select for running net and displaying blob info of that dataset: "train" or "test"*/
    void SelectDataset(std::string which_one);

    NetState* GetState() { return m_state; }
private:
    friend class NetState;

    void CreateBackup();
    void RunMKL(const int no);
    void UpdateWeights(const int batchSize);
    void RunForward(float *X, float *WX,    std::vector<float*> &poolingMaps, float* Derivative=nullptr);
    void RunForward(float *X, float *WX);
    void SetLossFunction(LossFunctionType type);
    void ChangeState(NetState* state);

    //calls LayerFactoryRegister to register layer available layer types in LayerFactory
    /*Probably better to make registers static*/
    void RegisterLayers(); 
    void RegisterLossFunctions();
    void RegisterComponents();
    int getNeuronsCount();
    int getWeightsCount();
    /*getter*/
    DataBlob* GetCurrentDataset();
private:
    NetState* m_state=nullptr;

    NN::DataBlob* currentDataset=nullptr;
    NN::Layer* currentLayer = nullptr;
    NN::DataBlob* trainBlob = nullptr;
    NN::DataBlob* testBlob = nullptr;
    NN::LossFunction* lossFunc=nullptr;

    int weights_total=0;
    int neurons_total=0;
    int net_rows=0;
    int net_input_sz=0;
    int net_output_sz=0;
    float m_loss=0.0f;
    int edge_length = 0;
    std::string net_name = "";

    NetTelemetry status;
    TrainingTelemetry training_telemetry;
    std::chrono::duration<double> training_duration=std::chrono::seconds(0);
    TrainingOptions options;

    std::vector<Layer*> layers;
    static DesignPatterns::Factory<NN::Layer, std::string, NN::Layer* (*)()> layerFactory;
    static DesignPatterns::Factory<LossFunction, std::string, LossFunction* (*)()> lossFunctionFactory;
    Object* executioner = nullptr;
};

    

}


#endif // NEURALNETWORK_H
