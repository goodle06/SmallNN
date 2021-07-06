#include <NeuralNetwork/LossFunction.h>

namespace NN {

/***************************************************************************************************************************************
 * Base class implementation
 ***************************************************************************************************************************************/


/***************************************************************************************************************************************
 * END base class implementation
 ***************************************************************************************************************************************/



//float QuadraticLossFuntion::CalculateLoss(float* Y, const int length, const int class_no)  {
//    float *error=(float *)mkl_calloc(length,sizeof (float),64);
//    this->CalculateError(Y,class_no,error,length);
//    vsMul(length,error, error, error);
//    float res=cblas_sasum(length,error,1);
//    mkl_free(error);
//    return res;
//}

float QuadraticLossFuntion::CalculateLoss(float* Y, const int length, float* class_no)  {
    float *error=(float *)mkl_calloc(length,sizeof (float),64);
    this->CalculateError(Y,class_no,error,length);
    vsMul(length,error, error, error);
    float res=cblas_sasum(length,error,1);
    mkl_free(error);
    return res;
}

//void QuadraticLossFuntion::CalculateDerivative(float *Y, float *derivative, const int length, const int class_no) {
//    float *error=(float *)mkl_calloc(length,sizeof (float),64);
//    this->CalculateError(Y,class_no,error,length);
//     std::memset(derivative,0,length*sizeof(float));
//    cblas_saxpy(length,-2.0f,error,1,derivative,1);
//    mkl_free(error);
//}

void QuadraticLossFuntion::CalculateDerivative(float *Y, float *derivative, const int length, float *class_no) {
    float *error=(float *)mkl_calloc(length,sizeof (float),64);
    this->CalculateError(Y,class_no,error,length);
     std::memset(derivative,0,length*sizeof(float));
    cblas_saxpy(length,-2.0f,error,1,derivative,1);
    mkl_free(error);

}

//void QuadraticLossFuntion::CalculateError(float *Y, const int class_no, float *error, const int length) {
//    cblas_saxpy(length,-1.0f,Y,1,error,1);
//    error[class_no]=1.0f+error[class_no];
//}

void QuadraticLossFuntion::CalculateError(float *Y, float* class_no, float *error, const int length) {
    vsSub(length,class_no,Y,error);
}

void QuadraticLossFuntion::CalculateY(float *X, const int length, float *Y) {
    std::memcpy(Y,X,length*sizeof(float));
}



//void SoftmaxLoss::CalculateDerivative(float* Y, float *derivative, const int length, const int class_no) {
//    float *error=(float *)mkl_calloc(length,sizeof (float),64);
//    vsExp(length,Y,error);
//    m_logsum=1.0f/cblas_sasum(length,error,1);

//    std::memset(derivative,0,length*sizeof (float));
//    cblas_saxpy(length,m_logsum,error,1,derivative,1);
//    derivative[class_no]=-1.0f+derivative[class_no];
//    mkl_free(error);
//}

void SoftmaxLoss::CalculateDerivative(float* Y, float *derivative, const int length, float *class_no) {
    float *error=(float *)mkl_calloc(length,sizeof (float),64);
    vsExp(length,Y,error);
    m_logsum=1.0f/cblas_sasum(length,error,1);

    std::memset(derivative,0,length*sizeof (float));
    cblas_saxpy(length,m_logsum,error,1,derivative,1);
    vsSub(length,derivative,class_no,derivative);

//    for (int i=0;i<length;i++) LOG << derivative[i] << "\n";

    mkl_free(error);

}

void SoftmaxLoss::CalculateY(float *X, const int length, float *Y) {
    float *error=(float *)mkl_calloc(length,sizeof (float),64);
    vsExp(length,X,error);
    m_logsum=1.0f/cblas_sasum(length,error,1);
    std::memset(Y,0,length*sizeof (float));
    cblas_saxpy(length,m_logsum,error,1,Y,1);
}


//void SoftmaxLoss::CalculateError(float *Y, const int class_no, float *error, const int length) {
//    std::memset(error,0,length*sizeof(float));
//    vsExp(length,Y,error);
//    m_logsum=1.0f/cblas_sasum(length,error,1);
//    cblas_saxpy(length,m_logsum,error,1,error,1);
//    error[class_no]=-1.0f+error[class_no];
//}

void SoftmaxLoss::CalculateError(float *Y, float* class_no, float *error, const int length) {
    std::memset(error,0,length*sizeof(float));
    vsExp(length,Y,error);
    m_logsum=1.0f/cblas_sasum(length,error,1);
    cblas_saxpy(length,m_logsum,error,1,error,1);
    vsSub(length,error,class_no,error);
//    std::memset(error,0,length*sizeof(float));
//    cblas_saxpy(length,-1.0f,Y,1,error,1);
//    error[class_no]=1.0f+error[class_no];
}


//float SoftmaxLoss::CalculateLoss(float* Y, const int length, const int class_no)  {
//    float *error=(float *)mkl_calloc(length,sizeof (float),64);
//    float y=-Y[class_no];
//    vsExp(length,Y,error);
//    float res=cblas_sasum(length,error,1);
//    res=std::logf(res);
//    res+=y;
//    mkl_free(error);
//    return res;
//}

float SoftmaxLoss::CalculateLoss(float* Y, const int length, float* class_no)  {
    float *error=(float *)mkl_calloc(length,sizeof (float),64);
    int i=0;
    while(!class_no[i]) {
        i++;
    }
    float y=-Y[i];
    vsExp(length,Y,error);
    float res=cblas_sasum(length,error,1);
    res=std::logf(res);
    res+=y;
    mkl_free(error);
    return res;
}


float MultiLabelCrossEntropyLoss::CalculateLoss(float *X, const int length, float *class_no) {

    float *yCapped=(float *)mkl_calloc(length,sizeof (float),64);


    float *entropy=(float *)mkl_calloc(length,sizeof (float),64);
    float *lnYCapped=(float *)mkl_calloc(length,sizeof (float),64);

    vsLn(length,X,lnYCapped);

    for (int i=0;i<length;i++) {
        if (std::isinf(lnYCapped[i])) {
            lnYCapped[i]=(std::numeric_limits<float>::min)();
        }
    }
    vsMul(length,class_no,lnYCapped,entropy);


    vsLinearFrac(length,class_no,lnYCapped,-1.0f,1.0f,0.0f,1.0f,lnYCapped);
    vsLinearFrac(length,X,X,-1.0f,1.0f,0.0f,1.0f,yCapped);
    vsLn(length,yCapped,yCapped);

    for (int i=0;i<length;i++) {
        if (std::isinf(yCapped[i])) {
            yCapped[i]=(std::numeric_limits<float>::min)();
        }
    }

    vsMul(length,lnYCapped,yCapped,yCapped);
    vsAdd(length,entropy,yCapped,entropy);

//    LOG << "entropy:\n";
//    for (int i=0;i<length; i++) LOG << QString::number(entropy[i],'f',4) << "\n";

    float res=cblas_sasum(length,entropy,1);
    res=res/length;

//    LOG << "entropy: " << res << "\n";
    mkl_free(entropy);
    mkl_free(yCapped);
    mkl_free(lnYCapped);
    return res;
}

void MultiLabelCrossEntropyLoss::CalculateY(float *X, const int length, float *Y) {
    std::memcpy(Y,X,length*sizeof(float));
//    float *minusX=(float *)mkl_calloc(length,sizeof (float),64);
//    cblas_saxpy(length,-1.0f,X,1,minusX,1);
//    vsExp(length,minusX,minusX);
//    vsLinearFrac(length,Y,minusX,0.0f,1.0f,1.0f,1.0f,Y);
//    mkl_free(minusX);
}

void MultiLabelCrossEntropyLoss::CalculateDerivative(float *X, float *derivative, const int length, float *labels) {
//    for (int i=0;i<length; i++) LOG << QString::number(X[i],'f',4) << "\n";

//    LOG << "X:\n";
//    for (int i=0;i<length; i++) LOG << QString::number(X[i],'f',4) << "\n";
//    LOG << "Labels:\n";
//    for (int i=0;i<length; i++) LOG << QString::number(labels[i],'f',4) << "\n";

    float *entropy=(float *)mkl_calloc(length,sizeof (float),64);

    vsSub(length,X,labels,entropy);
    vsSqr(length,X,derivative);
    vsSub(length,X,derivative,derivative); //denominator
    vsLinearFrac(length,entropy,derivative,1.0f,0.0f,1.0f,0.001f,derivative); //shift b 0.1 to avoid overflow

//    LOG << "Derivative:\n";
//    for (int i=0;i<length; i++) LOG << QString::number(derivative[i],'f',4) << "\n";
    mkl_free(entropy);
}

void MultiLabelCrossEntropyLoss::CalculateError(float *X, float *class_no, float *error, const int length) {
    vsSub(length,class_no,X,error);
}


}
