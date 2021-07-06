#ifndef LOSSFUNTION_H
#define LOSSFUNTION_H
#include <Proj/Common.h>

namespace NN {

enum class LossFunctionType :int {Quadratic=0, Softmax=1, BinaryCrossEntropy=2};

class LossFunction {
public:
    LossFunction() {}
//    virtual void CalculateDerivative(float* error, float *derivative, const int length, const int class_no) =0;
    virtual void CalculateDerivative(float* error, float *derivative, const int length, float *class_no) =0;

//    virtual float CalculateLoss(float *Y, const int length, const int class_no) = 0;
    virtual float CalculateLoss(float *Y, const int length, float* class_no) = 0;

//    virtual void CalculateError(float* Y, const int class_no, float* error, const int length) = 0;
    virtual void CalculateError(float* Y, float* class_no, float* error, const int length) = 0;

    virtual void CalculateY(float *X, const int length, float *Y) = 0;
    virtual LossFunctionType type() const = 0;
    virtual std::string print() const = 0;
};

class QuadraticLossFuntion : public LossFunction {
public:
    QuadraticLossFuntion() {}
//    float CalculateLoss(float *Y, const int length, const int class_no) override;
    float CalculateLoss(float *Y, const int length, float* class_no) override;
//    void CalculateDerivative(float* Y, float *derivative, const int length, const int class_no) override;
    void CalculateDerivative(float* error, float *derivative, const int length, float *class_no) override;
//    void CalculateError(float* Y, const int class_no, float* error, const int length) override;
    void CalculateError(float* Y, float *class_no, float* error, const int length) override;

    void CalculateY(float *X, const int length, float *Y) override;
    std::string print() const override {return "Quadratic";}
    LossFunctionType type() const override {return LossFunctionType::Quadratic;}
};

class SoftmaxLoss : public LossFunction {
public:
    SoftmaxLoss() {}
//    float CalculateLoss(float *Y, const int length, const int class_no) override;
    float CalculateLoss(float *Y, const int length, float* class_no) override;
//    void CalculateDerivative(float* error, float *derivative, const int length, const int class_no) override;
    void CalculateDerivative(float* error, float *derivative, const int length, float *class_no) override;
//    void CalculateError(float* Y, const int class_no, float* error, const int length) override;
    void CalculateError(float* Y, float* class_no, float* error, const int length) override;
    void CalculateY(float *X, const int length, float *Y) override;
    std::string print() const override {return "Softmax";}
    LossFunctionType type() const override {return LossFunctionType::Softmax;}
private:
    float m_logsum=0;
};


class MultiLabelCrossEntropyLoss : public LossFunction {
public:
    MultiLabelCrossEntropyLoss() {}
    float CalculateLoss(float *Y, const int length, float* class_no) override;
    void CalculateDerivative(float* error, float *derivative, const int length, float *class_no) override;
    void CalculateY(float *X, const int length, float *Y) override;
    void CalculateError(float* Y, float* class_no, float* error, const int length) override;
    std::string print() const override {return "BinaryCrossEntropy";}
    LossFunctionType type() const override {return LossFunctionType::BinaryCrossEntropy;}

};

}

#endif // LOSSFUNTION_H
