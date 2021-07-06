#include <NeuralNetwork/Blobs/DataTransformation.h>


namespace NN {

    std::map<std::string, TransformationType> Transformation::string_mapping = {
            {"sin",TransformationType::sin},
            {"cos",TransformationType::cos},
            {"arcsin",TransformationType::arcsin},
            {"arccos",TransformationType::arccos},
            {"mult",TransformationType::multiplication},
            {"div",TransformationType::division},
            {"shift",TransformationType::shift},
    };
    std::map<TransformationType, TransformationType> Transformation::mirror_mapping = {
            {TransformationType::arcsin,TransformationType::sin},
            {TransformationType::arccos,TransformationType::cos},
            {TransformationType::sin,TransformationType::arcsin},
            {TransformationType::cos,TransformationType::arccos},
            {TransformationType::division,TransformationType::multiplication},
            {TransformationType::multiplication,TransformationType::division},
            {TransformationType::shift,TransformationType::shift},
    };

Transformation::Transformation(TransformationType in_type, float coeff, int offset) :    
    type(in_type), value(coeff), m_offset(offset) {
    reverse_tranform = mirror_mapping.at(type);
}

Transformation::Transformation(std::string in_type, float coeff, int offset)
    : type(string_mapping.at(in_type)), value(coeff), m_offset(offset) {
    reverse_tranform = mirror_mapping.at(type);
}


void Transformation::apply(float *x, long length) {
    switch (type) {
    case TransformationType::sin: {
        for (int i=m_offset; i<length; i++) {
            x[i]=std::sin(x[i]);
        }
        break;
    }
    case TransformationType::cos: {
        for (int i=m_offset; i<length; i++) {
            x[i]=std::cos(x[i]);
        }
        break;
    }
    case TransformationType::arcsin: {
        for (int i=m_offset; i<length; i++) {
            x[i]=std::asin(x[i]);
        }
        break;
    }
    case TransformationType::arccos: {
        for (int i=m_offset; i<length; i++) {
            x[i]=std::acos(x[i]);
        }
        break;
    }
    case TransformationType::division: {
        for (int i=m_offset; i<length; i++) {
            x[i]/=value;
        }
        break;
    }
    case TransformationType::multiplication: {
        for (int i=m_offset; i<length; i++) {
            x[i]*=value;
        }
        break;
    }
    case TransformationType::shift: {
        for (int i=m_offset; i<length; i++) {
            x[i]+=value;
        }
        break;
    }
    }
}

void Transformation::reverse(float *x, long length) {
    switch (reverse_tranform) {
    case TransformationType::sin: {
        for (int i=m_offset; i<length; i++) {
            x[i]=std::sin(x[i]);
        }
        break;
    }
    case TransformationType::cos: {
        for (int i=m_offset; i<length; i++) {
            x[i]=std::cos(x[i]);
        }
        break;
    }
    case TransformationType::arcsin: {
        for (int i=m_offset; i<length; i++) {
            x[i]=std::asin(x[i]);
        }
        break;
    }
    case TransformationType::arccos: {
        for (int i=m_offset; i<length; i++) {
            x[i]=std::acos(x[i]);
        }
        break;
    }
    case TransformationType::division: {
        for (int i=m_offset; i<length; i++) {
            x[i]/=value;
        }
        break;
    }
    case TransformationType::multiplication: {
        for (int i=m_offset; i<length; i++) {
            x[i]*=value;
        }
        break;
    }
    case TransformationType::shift: {
        for (int i=m_offset; i<length; i++) {
            x[i]-=value;
        }
        break;
    }
    }
}




}
