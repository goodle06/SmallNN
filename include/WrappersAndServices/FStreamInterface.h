#ifndef FSTREAMINTERFACE_H
#define FSTREAMINTERFACE_H

#include <Common.h>

void UploadToStream(int *data, std::fstream &stream, int count) ;
int DownloadIntFromStream(std::fstream &stream) ;

void UploadToStream(float *data, std::fstream &stream, int count) ;
float DownloadFloatFromStream(std::fstream &stream) ;

void UploadToStream(std::string *data, std::fstream &stream) ;
std::string DownloadStringFromStream(std::fstream &stream);

#endif // FSTREAMINTERFACE_H
