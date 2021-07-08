#ifndef DATABLOB_H
#define DATABLOB_H

#include <NeuralNetwork/Blobs/OriginalSample.h>
#include <NeuralNetwork/Blobs/TranslationUnit.h>
#include <NeuralNetwork/Object.h>

namespace NN {


class DataBlob : public Object {
public:
    DataBlob() {}
    DataBlob(std::string blobName);
    DataBlob(std::string filename, std::string translation_filename );
    ~DataBlob();
    void loadFromFile();
    void loadTranslation();

    int getMemorySize();
    size_t size() const {return maindata.size();}
    int vector_size() {return vector_dimension;}
    void uploadSample(cv::Mat sample, const int character);
    void uploadSample(OriginalSample sample);
    void addSample(cv::Mat sample, const std::vector<uchar> labels);
    void removeSample(const int no) ;
    void removeClass(const std::string class_character);
    void correctLabel(const int no, const int correct_label) ;
    void transform(Transformation tranform) ;
    void saveBlob();
    void splitBlob(std::string new_blob_name, int percentage_of_data_going_to_new_blob);

    void transferSample(DataBlob *where, int no);
    struct XY {
        float *X;
        float *Y;
    };
    XY at(int no);
    std::string getLabelsString(int no);


    size_t count() { return maindata.size();}
    void print() ;
    void display(int no, bool original=false);
    void resize(int nsize, bool pad, bool preserve_aspect_ratio=false, bool random_padding=false);

    void printSample(int no);
    void printDistribution();
    void distributionGraph();

    cv::Mat toCvMat(const int sample);

    std::string file_src;
    std::vector<OriginalSample> maindata;
    std::vector<OriginalSample> originals;

    TranslationUnit translation;
    std::vector<Transformation> transforms;
    int vector_dimension=0;
    int width=0;
    int height=0;

    void ExecuteCommand(NetCommand& cmd) override;
protected:

private:
//    qint32 memory_sz=0;
    float* data=nullptr;
    uchar* original_data=nullptr;
    void slice(int memory_sz);
    bool rangeCheck(int no);

};

}

#endif // DATABLOB_H
