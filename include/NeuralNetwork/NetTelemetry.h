#ifndef NETTELEMETRY_H
#define NETTELEMETRY_H
#include <Proj/Common.h>


namespace NN {

class NetTelemetry {
public:
    std::map<std::string,bool> tel {
        {"testing data", false},
        {"training data", false},
        {"layers", false},
        {"weights", false},
        {"connections", false},
        {"loss function", false},
        {"learning rate", false},
    };
    bool is_ready();
    bool is_test_ready();
};



}



#endif // NETTELEMETRY_H
