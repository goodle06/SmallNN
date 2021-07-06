#include <NeuralNetwork/Layers/PoolingLayer.h>

namespace NN {


PoolingLayer::PoolingLayer(int height, int width, int stride) {
    m_window_height=height;
    m_window_stride=stride;
    m_window_width=width;

}

PoolingLayer::PoolingLayer(std::initializer_list<int> params)
{
    SetParams(params);
}

void PoolingLayer::SetParams(std::initializer_list<int> params)
{
    auto p = params.begin();
    if (p != params.end()) m_window_height = *p;
    if (p+1 != params.end()) m_window_width = *(++p);
    if (p+1 != params.end()) m_window_stride = *(++p);
}

bool PoolingLayer::SetConnections(DataBlob &blob) {
    std::cout << "ERROR: couldn't connect pooling layer to datablob\n";
    return false;
}

bool PoolingLayer::SetConnections(Layer *prev) {

    ConvolutionalLayer *child=dynamic_cast<ConvolutionalLayer*>(prev);
    inVolumeHeight=child->outVolumeHeight;
    inVolumeWidth=child->outVolumeWidth;
    inVolumeDepth=child->m_filter_count;
    this->m_filter_count=inVolumeDepth;
    if (!CalcInternalParams()) return false;
    assignMemoryInternal();
    MapLayer();
    return true;
}

bool PoolingLayer:: CalcInternalParams() {
    this->dense_filter_length=m_window_height*m_window_width;
    this->dense_weights_count=dense_filter_length*m_filter_count;

    /* Filter stride check*/
    if ((this->inVolumeHeight-m_window_height)%m_window_stride!=0
            ||(this->inVolumeWidth-m_window_width)%m_window_stride!=0) {
        std::cout << "Window stride leads to necessity of input padding. This functionality isn't ready yet\n";
        return false;
    }

    this->outVolumeHeight=(this->inVolumeHeight-m_window_height)/m_window_stride+1;
    this->outVolumeWidth=(this->inVolumeWidth-m_window_width)/m_window_stride+1;
    rows=this->outVolumeHeight*this->outVolumeWidth*m_filter_count;
    cols=inVolumeDepth*inVolumeHeight*inVolumeWidth;
    total_count=rows*cols;

    return true;
//    LOG << rows << " " << cols << " " << total_count << "\n";
}

void PoolingLayer::MapLayer() {
    int sparse_index=0;
    int f_row, f_col;
    for (int f=0;f<m_filter_count;f++) {
        for (int y=0;y<outVolumeHeight;y++) {
            for (int x=0;x<outVolumeWidth;x++) {
                f_row=x+y*outVolumeWidth+f*outVolumeHeight*outVolumeWidth; //
                pointerB[f_row]=f_row*m_window_height*m_window_width;
                pointerE[f_row]=pointerB[f_row]+m_window_height*m_window_width;
                for (int wy=0;wy<m_window_height;wy++) {
                    for (int wx=0;wx<m_window_width;wx++) {
                        f_col=f*inVolumeHeight*inVolumeWidth // input filter column shift
                                + (y*m_window_stride+wy)*inVolumeWidth+(x*m_window_stride+wx); //window shift
                        columns[sparse_index]=f_col;
                        sparse_index++;
                    }
                }
            }
        }
    }


}

float* PoolingLayer::RunWX(float *X, float *dest) {
    float *res=(float*)mkl_calloc(dense_filter_length*rows,sizeof(float),64);

    int src_index, max_ind;
    int sparse_ind=0;
    float val, max_val;
    for (int y=0;y<this->rows;y++) {
        max_val=0;
        max_ind=sparse_ind;
        while (sparse_ind<pointerE[y]) {
            src_index=columns[sparse_ind]; // column index of sparse matrix corresponds to input vector index
            val=X[src_index]; // getting value of input vector
            if (val>max_val) {
                max_val=val;
                max_ind=sparse_ind;
            }
            sparse_ind++; // next element index of sparse matrix
        }
        dest[y]=max_val; // destination gets only max value, total count of max values equal number of rows of pooling layer
        res[max_ind]=1.0f; // sets activation weight to 1 while others are 0
    }
    return res;
}
void PoolingLayer::RunDerivative(float *WX, float *D) {}

/* Performs pooling and returns activation map (indices of max values) for backwards propagation */
void PoolingLayer::RunActivate(float *X, float *dest) {
    std::memcpy(dest,X, this->rows*sizeof(float));
}

/* Propagates gradient back with only activated weights being considered */
void PoolingLayer::RunBackwards(float *vector, float *destination, float *derivative, float *poolingMatrix) {
    sparse_matrix_t B;
    matrix_descr descrB;
    descrB.type=SPARSE_MATRIX_TYPE_GENERAL;

    auto stat=mkl_sparse_s_create_csr(&B, SPARSE_INDEX_BASE_ZERO,rows,cols,&pointerB[0], &pointerE[0],&columns[0],&poolingMatrix[0]);
    if (stat==SPARSE_STATUS_SUCCESS) {
        mkl_sparse_s_mv(SPARSE_OPERATION_TRANSPOSE,1.0f,B,descrB,vector,0.0f,destination);
    }
    else {
        std::cout << "Unable to create sparse matrix for pooling layer backwards propagation\n";
    }
    mkl_sparse_destroy(B);
    mkl_free(poolingMatrix);
}
void PoolingLayer::CalculateGradient(float *derivative, float *X) {}

void PoolingLayer::SeedWeights(const float lower_bound, const float upper_bound) {}   //overload, no weigth seeding needed

void PoolingLayer::UpdateWeights(const int batchSize, const float learning_rate) {}   // overload, no updates for pooling layers

void PoolingLayer::print(bool weights) const {
    LOG << "Layer type: pooling, ";
    LOG << "Size: " << m_window_height << "x" << m_window_width << "x" << m_filter_count << ", " << "stride " << m_window_stride << "\n";
    if (weights) std::cout << "No weights available for pooling layers\n";
}

void PoolingLayer::printGradient() const {
    std::cout << "Gradients unavailable for pooling layers\n";
}

void PoolingLayer::SaveStd(std::fstream &stream) {

    int ltype=static_cast<int>(this->Type());
    int ww=m_window_width;
    int wh=m_window_height;
    int ws=m_window_stride;
    int fc=m_filter_count;
    int vh=inVolumeHeight;
    int vw=inVolumeWidth;
    int vd=inVolumeDepth;

    std::vector<int> data_for_stream={ltype, ww,wh,ws,fc,vh,vw,vd};
    for (auto d : data_for_stream) UploadToStream(&d,stream,1);
}

void PoolingLayer::LoadStd(std::fstream &stream) {
    LayerType m_string_type;
    std::vector<int> data_from_stream(7);

    m_string_type=static_cast<LayerType>(DownloadIntFromStream(stream));
    for (auto &d : data_from_stream) d=DownloadIntFromStream(stream);

    m_window_width=data_from_stream[0];
    m_window_height=data_from_stream[1];
    m_window_stride=data_from_stream[2];
    m_filter_count=data_from_stream[3];

    inVolumeHeight=data_from_stream[4];
    inVolumeWidth=data_from_stream[5];
    inVolumeDepth=data_from_stream[6];

    CalcInternalParams();
    assignMemoryInternal();
    MapLayer();
}


void PoolingLayer::Test() {

    this->inVolumeHeight=4;
    this->inVolumeWidth=4;
    this->inVolumeDepth=3;
    int input_sz=this->inVolumeHeight*this->inVolumeWidth*this->inVolumeDepth;
    this->m_filter_count=inVolumeDepth;
    CalcInternalParams();
    assignMemoryInternal();
    MapLayer();

    float *input_vec=(float*)mkl_malloc(input_sz*sizeof(float),64);
    std::random_device dev;
    std::default_random_engine e1(dev());
    std::uniform_real_distribution<float> dist(-0.2,0.2);
    for (int i=0;i<input_sz;i++) input_vec[i]=dist(e1);

    std::cout << "Input volume\n";
    for (int d=0;d<this->inVolumeDepth;d++) {
        std::cout << "filter #" << d << ":\n";
        for (int y=0;y<this->inVolumeHeight;y++) {
            for (int x=0;x<this->inVolumeWidth;x++) {
                std::cout << input_vec[d*this->inVolumeHeight*this->inVolumeWidth+y*this->inVolumeWidth+x] << " ";
            }
            std::cout << "\n";
        }
    }

    float *dest=(float*)mkl_calloc(this->rows,sizeof(float),64);
    float *map=this->RunWX(input_vec, dest);

    std::cout << "Output volume\n";
    for (int d=0;d<this->m_filter_count;d++) {
        std::cout << "filter #" << d << ":\n";
        for (int y=0;y<this->outVolumeHeight;y++) {
            for (int x=0;x<this->outVolumeWidth;x++) {
                std::cout << dest[d*this->outVolumeHeight*this->outVolumeWidth+y*this->outVolumeWidth+x] << " ";
            }
            std::cout << "\n";
        }
    }

    std::cout << "Activation map\n";
    int sparse_index=0;
    for (int y=0;y<this->rows;y++) {
        while (sparse_index<pointerE[y]) {
            std::cout << map[sparse_index] << " ";
            sparse_index++;
        }
        std::cout << "\n";
    }

    float *Y=(float*)mkl_malloc(this->rows*sizeof(float),64);
    float *D=(float*)mkl_calloc(this->cols, sizeof(float),64);
    for (int i=0;i<rows;i++) Y[i]=dist(e1);

    float *derivative=(float*)mkl_malloc(this->rows*sizeof(float),64);

    this->RunBackwards(Y,D,derivative, map);

    std::cout << "Backpropagation:\n";
    for (int g=0;g<cols;g++) {
        std::cout << D[g] << "\n";
    }

}



}
