#include "NeuralNetwork/Estimate.h"

namespace NN {

Estimate::Estimate(float *data, int length, LossFunction *loss) {
    m_lossFunction=loss;
    this->setType();
    float* estimate=(float*)mkl_calloc(length,sizeof(float),64);
    m_lossFunction->CalculateY(data,length,estimate);

    for (int i=0;i<length;i++) {
        predictions.push_back({i,estimate[i],0});
    }
    std::sort(predictions.begin(),predictions.end(),[](auto a, auto b) { return a.eY>b.eY;});


    std::vector<float> rs(estimate,estimate+length);
    Y.resize(length);
    for (auto &el : rs) el=std::fabs(el-1);
    for (auto i=0;i<length;i++) Y[i]={i,rs[i]};
    std::sort(Y.begin(),Y.end(),[](auto a1, auto a2) {return a1.second<a2.second;});
    for (auto i=0;i<length;i++) Y[i]={Y[i].first,1.0f-Y[i].second};
    for (auto i=0;i<length;i++) Y[i].second=Y[i].second*100.0f;
    mkl_free(estimate);
}

Estimate::Estimate(float *data, int length, LossFunction *loss, float *labels) {
    loaded=true;
    m_lossFunction=loss;
    this->setType();
    float* estimate=(float*)mkl_calloc(length,sizeof(float),64);
    m_lossFunction->CalculateY(data,length,estimate);

    for (int i=0;i<length;i++) {
        predictions.push_back({i,estimate[i],labels[i]});
    }
    std::sort(predictions.begin(),predictions.end(),[](auto a, auto b) { return a.eY>b.eY;});


    std::vector<float> rs(estimate,estimate+length);
    Y.resize(length);
    for (auto &el : rs) el=std::fabs(el-1);
    for (auto i=0;i<length;i++) Y[i]={i,rs[i]};
    std::sort(Y.begin(),Y.end(),[](auto a1, auto a2) {return a1.second<a2.second;});
    for (auto i=0;i<length;i++) Y[i]={Y[i].first,1.0f-Y[i].second};
    for (auto i=0;i<length;i++) Y[i].second=Y[i].second*100.0f;
    mkl_free(estimate);
}

void Estimate::print(int orders, bool newline) {
    for (int i=0; i<orders;i++) {
        std::cout << std::to_string(predictions[i].label_no) << "(" << std::setprecision(2) << predictions[i].eY << "%)";
        if (newline) std::cout << "\n";
        else std::cout << "\\ ";
    }
}

void Estimate::print(int orders, TranslationUnit translation, bool newline) {
    if (type==EstimateType::MultiLabel) {
//        for (auto &p : predictions) {
//            if (p.eY>0.5f&&p.Y==1.0f) {
//                output+=translation.toText(p.label_no) + "(" + QString::number(p.eY*100.0f,'f',2) + "%, " + QString::number(p.Y) +")";
//                if (newline) output+="\n";
//                else output+="\\ ";
//            }
//        }
        for (int i=0; i<orders&&i<(int)predictions.size();i++) {
            auto p=predictions[i];
            std::cout << "(" << translation.toText(p.label_no) << ", " << std::setprecision(2) << p.eY * 100.0f << "%" << ")";
            if (newline) std::cout << "\n";
            else std::cout << " \\ ";
        }
    }
    else {
        for (int i=0; i<orders&&i<(int)predictions.size();i++) {
            std::cout << translation.toText(predictions[i].label_no) << "(" << std::setprecision(2) << predictions[i].eY*100.0f  << "%)";
            if (newline) std::cout << "\n";
            else std::cout << "\\ ";
        }
        auto lbls=this->getLabels();
        for (auto &lbl : lbls) {
            std::cout << translation.toText(lbl) + " ";
        }
    }
    for (auto &p : predictions) {
        if (p.Y==1.0f) std::cout << "<" << translation.toText(p.label_no) << ">, ";
    }
}

bool Estimate::isCorrect(int est_no) {
    if (!loaded) return false;
    if (type==EstimateType::MultiLabel) {
        std::vector<bool> res(predictions.size(),false);
        for (size_t i=0;i<predictions.size();i++) {
            if (predictions[i].eY>=m_threshold)
                res[i]=true;
        }
        for (size_t i=0;i<predictions.size();i++) {
            if (predictions[i].Y!=res[i]) {
                return false;
            }
        }
        return true;
    }
    else {
        if (predictions[est_no].Y) return true;
    }
    return false;
}

std::vector<int> Estimate::getLabels() {
    if (!loaded) return {};
    std::vector<int> res;
    for (size_t i=0;i<predictions.size();i++)
        if (predictions[i].Y) res.push_back(predictions[i].label_no);
    return res;
}

void Estimate::setType() {
    if (!m_lossFunction) {
        LOG << "ERROR: loss function for estimate is not set\n";
        return;
    }

    if(m_lossFunction->type()==LossFunctionType::BinaryCrossEntropy)
        type=EstimateType::MultiLabel;
    else type=EstimateType::SinlgleLabel;
}

float Estimate::probabilityOf(std::string class_name, TranslationUnit translation) {
    int class_no=translation.fromText(class_name);
    auto res=std::find_if(predictions.begin(),predictions.end(),[class_no](auto a) {return class_no==a.label_no;});
    return res->eY;
}

}


