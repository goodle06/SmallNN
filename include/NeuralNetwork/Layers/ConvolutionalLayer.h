#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include <NeuralNetwork/Layers/Layer.h>
#include <NeuralNetwork/Blobs/DataBlob.h>


namespace NN {


class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(int filter_count,  int window_height, int window_width, int stride,  NN::ActivationFunctionType f_type, int padding=0);
    ConvolutionalLayer() {}
    ConvolutionalLayer(std::initializer_list<int> params);
    ~ConvolutionalLayer();
    LayerType mLayerType=LayerType::Conv;

    void SetParams(std::initializer_list<int> params) override;
    bool SetConnections(Layer* prev) override;
    bool SetConnections(DataBlob& blob) override;
    void assignMemory() override;
    void assignMemory(float *src) override;

    float* RunWX(float *X, float* dest) override;
    void RunBackwards(float *vector, float *destination,float *derivative, float *poolingMap=nullptr) override;
    void SeedWeights(const float lower_bound=-0.2f, const float upper_bound=0.2f) override;
    void UpdateWeights(const int batchSize, const float learning_rate) override;

    void print(bool with_weights=false) const override;
    void printGradient() const override;
    void printNeuron(const int no) const override;

    void Rollback() override;
    void CreateBackup() override;

    void SaveStd(std::fstream &stream) override;
    void LoadStd(std::fstream &stream) override;
    LayerType Type() const override {return LayerType::Conv;}

    std::vector<float> getlog(int weight_no) override;

    struct Volume {
        int height = 0; int width = 0; int depth = 0;
        int Length() { return height * width; }
    };
    struct Filter  {
        int height = 0; int width = 0; int stride = 0; int padding = 0;
        int Length() { return height * width; }
    };

    Volume GetOutputVolume() { return m_output_volume; }
    void SetInputVolume(Volume vol) { m_input_volume = vol; }


protected:
    Volume m_input_volume;
    Volume m_output_volume;
    Filter m_filter;


    virtual void MapLayer();
    virtual bool CalcInternalParams();
    void CopyValues(float *src);
    void assignMemoryInternal();

    sparse_matrix_t A=nullptr;
    matrix_descr descrA = {};

    float *nonzero_values=nullptr;
    float *values_source=nullptr;

    float *offset_vector_dense=nullptr;
    float *offset_gradient_source=nullptr;

    float *echo_values=nullptr;

    float **pointers_to_gradient_values=nullptr;
    float *gradient_values_source=nullptr;

    std::vector<float> weights_log;

    int *columns=nullptr;
    int *pointerB=nullptr;
    int *pointerE=nullptr;

    int dense_weights_count = 0;
    int dense_filter_length = 0;

    struct SparseMatrixData {

        SparseMatrixData(size_t rows, size_t cols);
        ~SparseMatrixData();

        sparse_matrix_t A = nullptr;
        matrix_descr descrA = {};
        float* nonzero_values = nullptr;
        int* columns = nullptr;
        int* pointerB = nullptr;
        int* pointerE = nullptr;

    };

};



}


#endif // CONVOLUTIONALLAYER_H
