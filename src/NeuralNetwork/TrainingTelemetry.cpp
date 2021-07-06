#include <NeuralNetwork/TrainingTelemetry.h>

namespace NN {

void TrainingTelemetry::displayLossGraph() {
    std::vector<double> x((std::max)(loss_progress.size(),loss_progress_test.size())), y1(loss_progress.size()), y2(loss_progress_test.size());
    for (size_t i = 0; i < x.size(); i++)
        x[i] = i;
    for (size_t i = 0; i < y1.size(); i++)
        y1[i] = loss_progress[i];
    for (size_t i = 0; i < y2.size(); i++)
        y2[i] = loss_progress_test[i];

    auto axes = CvPlot::makePlotAxes();
    axes.create<CvPlot::Series>(x, y2, "-b").setName("Test data");
    axes.create<CvPlot::Series>(x, y1, "-r").setName("Training data");
    CvPlot::show("Loss graph", axes);
}

void TrainingTelemetry::displayAnyGraph(std::vector<float> log, std::string name) {
    std::vector<double> x(log.size()), y1(log.size());
    for (size_t i = 0; i < x.size(); i++)
        x[i] = i;
    for (size_t i = 0; i < y1.size(); i++)
        y1[i] = log[i];

    auto axes = CvPlot::makePlotAxes();
    axes.create<CvPlot::Series>(x, y1, "-r").setName(name);
    CvPlot::show(name, axes);
}

void TrainingTelemetry::clear() {
    loss_progress.clear();
    loss_progress_test.clear();
}

}
