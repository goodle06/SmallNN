#include <NeuralNetwork/ActivationFunction.h>

namespace NN {

/***************************************************************************************************************************************
 * RELU class implementation
 ***************************************************************************************************************************************/
ActivationFunctionType RELUActivationFunction::type() const {
    return ActivationFunctionType::RELU;
}

void RELUActivationFunction::calc(const int lenth, float *X, float *Y) {
    float *tmp=(float*)mkl_calloc(lenth,sizeof (float),64);
    vsFmax(lenth, X,tmp,Y);
    mkl_free(tmp);
}

void RELUActivationFunction::derivative(const int lenth, float *X, float *Y)  {
    for (int i=0;i<lenth; i++) X[i]>0? Y[i]=1.0f : Y[i]=0.0f;
}


/***************************************************************************************************************************************
 * LeakyRELU class implementation
 ***************************************************************************************************************************************/
ActivationFunctionType LeakyRELUActivationFunction::type() const {
    return ActivationFunctionType::LeakyRELU;
}

void LeakyRELUActivationFunction::calc(const int lenth, float *X, float *Y) {
    for (int i=0;i<lenth; i++) X[i]>0? Y[i]=X[i] : Y[i]=X[i]*leakage;
}

void LeakyRELUActivationFunction::derivative(const int lenth, float *X, float *Y)  {
    for (int i=0;i<lenth; i++) X[i]>0? Y[i]=1.0f : Y[i]=leakage;
}

/***************************************************************************************************************************************
 * Sinusoid class implementation
 ***************************************************************************************************************************************/
void SinusoidActivationFunction::calc(const int length, float *X, float *Y) {
    vsSin(length, X, Y);
}
void SinusoidActivationFunction::derivative(const int length, float *X, float *Y)  {
    vsCos(length,X,Y);
}
ActivationFunctionType SinusoidActivationFunction::type() const {
    return ActivationFunctionType::Sinusoid;
}

/***************************************************************************************************************************************
 * Logistic class implementation
 ***************************************************************************************************************************************/
void LogisticActivationFunction::calc(const int length, float *X, float *Y) {
    float *minusX=(float *)mkl_calloc(length,sizeof (float),64);
    cblas_saxpy(length,-1.0f,X,1,minusX,1);
    vsExp(length,minusX,minusX);

    for (int i=0;i<length;i++)
        if (std::isinf(minusX[i]))
            minusX[i]=(std::numeric_limits<float>::max)();

    vsLinearFrac(length,Y,minusX,0.0f,1.0f,1.0f,1.0f,Y);
    mkl_free(minusX);
}

void LogisticActivationFunction::derivative(const int length, float *X, float *Y) {

    this->calc(length,X, Y);
    float *minusX=(float *)mkl_calloc(length,sizeof (float),64);
    vsLinearFrac(length,Y,minusX,-1.0f,1.0f,0.0f,1.0f,minusX);
    vsMul(length,Y,minusX,Y);
    mkl_free(minusX);

}

ActivationFunctionType LogisticActivationFunction::type() const {return ActivationFunctionType::Logistic;}





}
