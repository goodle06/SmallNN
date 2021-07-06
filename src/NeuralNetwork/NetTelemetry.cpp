#include <Proj/NeuralNetwork/NetTelemetry.h>

namespace NN {

bool NetTelemetry::is_ready() {
    bool res=true;
    for (auto &param : tel) {
        if (!param.second) {
            LOG << param.first << " is not ready" << "\n";
            res=false;
        }
    }
    return res;
}


bool NetTelemetry::is_test_ready() {
    bool res=true;
    for (auto &param : tel) {
        if (!param.second) {
            if (param.first!="learning rate") {
                LOG << param.first << " is not ready" << "\n";
                res=false;
            }
        }
    }
    return res;
}


}


