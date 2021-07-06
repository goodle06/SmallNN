#include <WrappersAndServices/FStreamInterface.h>

void UploadToStream(int *data, std::fstream &stream, int count) {
    stream.write(reinterpret_cast<char*>(data),sizeof(int)*count);
}

int DownloadIntFromStream(std::fstream &stream) {
    int dst;
    stream.read(reinterpret_cast<char*>(&dst),sizeof(int));
    return dst;
}

void UploadToStream(float *data, std::fstream &stream, int count) {
     stream.write(reinterpret_cast<char*>(data),sizeof(float)*count);
}

float DownloadFloatFromStream(std::fstream &stream) {
    float dst;
    stream.read(reinterpret_cast<char*>(&dst),sizeof(float));
    return dst;
}

void UploadToStream(std::string *data, std::fstream &stream) {
    int count=data->size();
    UploadToStream(&count,stream,1);
    stream.write(reinterpret_cast<char*>(data->data()),sizeof(char)*count);
}

std::string DownloadStringFromStream(std::fstream &stream) {
    int count=DownloadIntFromStream(stream);
    std::string dst;
    dst.resize(count);
    stream.read(reinterpret_cast<char*>(dst.data()),sizeof(char)*count);
    return dst;
}
