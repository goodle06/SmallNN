#ifndef POOLINGLAYER_H
#define POOLINGLAYER_H

#include <Proj/NeuralNetwork/Layers/ConvolutionalLayer.h>

namespace NN {

class PoolingLayer: public ConvolutionalLayer {
public:
    PoolingLayer( int height, int width, int stride);
    PoolingLayer() {}
    PoolingLayer(std::initializer_list<int> params);

    void SetParams(std::initializer_list<int> params) override;
    bool SetConnections(DataBlob& blob) override;
    bool SetConnections(Layer* prev) override;

    void RunActivate(float *X, float* dest) override;

    float* RunWX(float *X, float* dest) override;
    void RunDerivative(float *WX,float *D) override;
    void CalculateGradient(float *derivative, float *X) override;

    void RunBackwards(float *vector, float *destination,float *derivative, float *poolingMap) override;
    void UpdateWeights(const int batchSize, const float learning_rate) override;
    void SeedWeights(const float lower_bound=-0.2f, const float upper_bound=0.2f) override;


    void print(bool with_weights=false) const override;
    void printGradient() const override;

    void SaveStd(std::fstream &stream) override;
    void LoadStd(std::fstream &stream) override;

    LayerType Type() const override {return LayerType::MaxPool;}

    void Test();

    LayerType mLayerType=LayerType::MaxPool;
private:
    void MapLayer() override;
    bool CalcInternalParams() override;
};




}

#endif // POOLINGLAYER_H
