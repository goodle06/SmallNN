#ifndef ORIGINALSAMPLE_H
#define ORIGINALSAMPLE_H
#include <Proj/NeuralNetwork/Blobs/DataTransformation.h>
#include <3party/avir.h>

namespace NN {


enum class DistortionType{ rotation, decimation };
enum class BlobType{X, Y};


struct OriginalSample {
//    OriginalSample(std::vector<uchar> original_uchar_data, uchar lbl, uchar cols, uchar rows);
//    OriginalSample(std::vector<uchar> original_uchar_data, std::vector<float> lbls, uchar cols, uchar rows);
    OriginalSample() {}
    OriginalSample(std::vector<uchar> original_uchar_data, uchar cols, uchar rows);
    OriginalSample(std::vector<uchar> original_uchar_data, std::vector<uchar> lbls, uchar cols, uchar rows, int class_count);
    std::vector<uchar> data_matrix;
    std::vector<float> data_matrix_f;
    std::vector<float> labels;
    std::vector<uchar> labels_short;
    uchar matrix_cols=0;
    uchar matrix_rows=0;
    void Pad(const int nsize, bool random_shift=false);
    void resize(int nsize, bool preserve_aspect_ratio=false);
};





}

#endif // ORIGINALSAMPLE_H
