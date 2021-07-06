#ifndef TRAININGTELEMETRY_H
#define TRAININGTELEMETRY_H

#include <Common.h>


namespace NN {

class TrainingTelemetry {
public:
    TrainingTelemetry() {}

    std::vector<float> loss_progress;
    std::vector<float> loss_progress_test;

    void displayAnyGraph(std::vector<float> log, std::string name);
    void displayLossGraph();
    void clear();

};



}


#endif // TRAININGTELEMETRY_H
