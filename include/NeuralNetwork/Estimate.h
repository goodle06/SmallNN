#ifndef ESTIMATE_H
#define ESTIMATE_H
#include <Common.h>
#include <NeuralNetwork/Blobs/TranslationUnit.h>
#include <NeuralNetwork/LossFunction.h>



namespace NN {


enum class EstimateType {SinlgleLabel, MultiLabel};

class Estimate
{
public:


    Estimate() {}
    Estimate(float *, int length, LossFunction *loss_function);
    Estimate(float *Y, int length, LossFunction *loss_function, float *labels);

    struct Prediction {
        int label_no;
        float eY;
        float Y;
    };
    std::vector<Prediction> predictions;
    bool loaded=false;

    Prediction get(int no) {return predictions[no];}
    std::vector<int> getLabels();
    bool isCorrect(int est_no);


    float pY() {return Y[0].second;}

    float probabilityOf(std::string class_name, TranslationUnit translation);
    std::vector<std::pair<int,float>> Y;
    std::vector<std::pair<std::string, float>> Ystr;
    void print(int orders, bool newline=false);
    void print(int orders, TranslationUnit translation,bool newline=false);
    bool exists() {return Y.size();}
    LossFunction *m_lossFunction=nullptr;
    std::vector<float> m_labels;
private:
    void setType();
    float m_threshold=0.5f;
    EstimateType type=EstimateType::MultiLabel;

};





}

#endif // ESTIMATE_H
