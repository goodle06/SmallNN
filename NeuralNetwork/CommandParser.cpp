#include <Proj/NeuralNetwork/CommandParser.h>


namespace NN {

NetCommand::NetCommand(const std::string &cmd) {
    std::string buf = "";
    for (auto& s : cmd) {
        if (std::isblank(s) && !buf.empty()) {
            params_queue.push(buf);
            buf.clear();
        }
        else
            buf += s;
    }
    params_queue.push(buf);
    m_cmd = params_queue.front();
    params_queue.pop();
}

std::string NetCommand::GetParameter() {
    if (params_queue.empty()) return "";
    std::string res = params_queue.front();
    params_queue.pop();
    return res;
}

}