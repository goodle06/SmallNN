#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H

#include <Proj/Common.h>

namespace NN {


enum class ActivationFunctionType : int { RELU, LeakyRELU, Sinusoid, Sigmoid, Logistic};

class ActivationFunction {
public:
    ActivationFunction() {}

    virtual void calc(const int lenth, float *X, float *Y) = 0;
    virtual void derivative(const int lenth, float *X, float *Y)= 0;
    virtual ActivationFunctionType type() const = 0;
    virtual std::string print() const = 0;
    virtual ~ActivationFunction() {};
};


class RELUActivationFunction : public ActivationFunction {
public:
    RELUActivationFunction() {}
    void calc(const int lenth, float *X, float *Y) override;
    void derivative(const int length, float *X, float *Y) override;
    std::string print() const override {return "RELU";}
    ActivationFunctionType type() const override;
};

class LeakyRELUActivationFunction : public ActivationFunction {
public:
    LeakyRELUActivationFunction() {}
    LeakyRELUActivationFunction(float leak) : leakage(leak) {}
    void calc(const int lenth, float *X, float *Y) override;
    void derivative(const int length, float *X, float *Y)  override;

    ActivationFunctionType type() const override;
    std::string print() const override {return "Leaky RELU";}
    float leakage=0.0001f;
};


class SinusoidActivationFunction : public ActivationFunction {
public:
    SinusoidActivationFunction() {}
    void calc(const int lenth, float *X, float *Y) override;
    void derivative(const int length, float *X, float *Y) override;

    ActivationFunctionType type() const override;
    std::string print() const override {return "Sinusoid";}

};


class LogisticActivationFunction : public ActivationFunction {
public:
    LogisticActivationFunction() {}
    void calc(const int lenth, float *X, float *Y) override;
    void derivative(const int length, float *X, float *Y) override;
    std::string print() const override {return "Logistic";}
    ActivationFunctionType type() const override;
};



}

#endif // ACTIVATIONFUNCTION_H
