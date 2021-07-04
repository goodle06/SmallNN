#ifndef DATATRANSFORMATION_H
#define DATATRANSFORMATION_H
#include <Proj/Common.h>

namespace NN {

enum class TransformationType{sin, cos, arcsin, arccos, division, multiplication, shift};

/*Class perfoms simple data transformations and reverse data transformations defined in enum ClassTransformationType*/
class Transformation {
public:
    Transformation(TransformationType Type, float coeff=1.0f, int offset=0);
    Transformation(std::string type, float coeff = 1.0f, int offset = 0);

    TransformationType type;
    TransformationType reverse_tranform;
    float value=1.0f;
    int m_offset=0;
    void apply(float *x, long length);
    void reverse(float *x, long length);
private:
    static std::map<std::string, TransformationType> string_mapping;
    static std::map<TransformationType, TransformationType> mirror_mapping;
};

}


#endif // DATATRANSFORMATION_H
