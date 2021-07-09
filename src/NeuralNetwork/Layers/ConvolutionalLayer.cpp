#include <NeuralNetwork/Layers/ConvolutionalLayer.h>

namespace NN {



ConvolutionalLayer::ConvolutionalLayer(int filter_count, int window_height, int window_width, int stride, NN::ActivationFunctionType f_type, int padding) {
     m_filter.height=window_height;
     m_filter.width=window_width;
     m_filter.stride=stride;
     m_output_volume.depth=filter_count;
     m_filter.padding=padding;
     this->setActivationFunction(f_type);
}


ConvolutionalLayer::ConvolutionalLayer(std::initializer_list<int> params)
{
    SetParams(params);
}

void ConvolutionalLayer::SetParams(std::initializer_list<int> params)
{
    auto p = params.begin();
    if (p != params.end()) m_output_volume.depth = *p;
    if (p + 1 != params.end()) m_filter.height = *(++p);
    if (p + 1 != params.end()) m_filter.width = *(++p);
    if (p + 1 != params.end()) m_filter.stride = *(++p);
    if (p + 1 != params.end()) m_filter.padding = *(++p);
}


ConvolutionalLayer::~ConvolutionalLayer() {

    mkl_sparse_destroy(A);

    if (values) mkl_free(values) ;
    if (columns) mkl_free(columns) ;
    if (pointerB) mkl_free(pointerB) ;
    if (pointerE) mkl_free(pointerE) ;
    if (gradient_values_source) mkl_free(gradient_values_source);
    if (pointers_to_gradient_values) mkl_free(pointers_to_gradient_values) ;
    if (echo_values) mkl_free(echo_values);

    if (values_source) mkl_free(values_source);
}


bool ConvolutionalLayer::SetConnections(Layer *prev) {

    ConvolutionalLayer *child=dynamic_cast<ConvolutionalLayer*>(prev);
    m_input_volume.height=child->m_output_volume.height;
    m_input_volume.width=child->m_output_volume.width;
    m_input_volume.depth=child->m_output_volume.depth;
    if (!CalcInternalParams()) return false;
    assignMemory();
    return true;
}

bool ConvolutionalLayer::SetConnections(DataBlob &blob) {
    m_input_volume.height=blob.height;
    m_input_volume.width=blob.width;
    m_input_volume.depth=1;
    if (!CalcInternalParams()) return false;
    assignMemory();
    return true;
}

void ConvolutionalLayer::assignMemory() {
    values_source=(float*)mkl_malloc(sizeof(float)*(m_input_volume.depth*m_filter.height*m_filter.width*m_output_volume.depth+m_output_volume.depth),64);
    assignMemoryInternal();
    MapLayer();
}

void ConvolutionalLayer::assignMemory(float *src) {
    values_source=src;
    assignMemoryInternal();
    MapLayer();
}

void ConvolutionalLayer::assignMemoryInternal() {
    offset_vector_dense=values_source+m_input_volume.depth*m_filter.height*m_filter.width*m_output_volume.depth;

    values=(float*)mkl_calloc(dense_filter_length*rows+rows, sizeof(float),64);
    offset_vector=values+dense_filter_length*rows;

    columns=(long long*)mkl_malloc(sizeof(long long)*dense_filter_length*rows,64);
    pointerB=(long long*)mkl_malloc(sizeof(long long)*rows,64);
    pointerE=(long long*)mkl_malloc(sizeof(long long)*rows,64);

    gradient_values_source=(float*)mkl_calloc(dense_filter_length*m_output_volume.depth+m_output_volume.depth,sizeof(float),64);
    offset_gradient_source=gradient_values_source+dense_filter_length*m_output_volume.depth;

    pointers_to_gradient_values=(float**)mkl_malloc(sizeof(float*)*dense_filter_length*rows,64);
    layer_gradient=(float*)mkl_calloc(total_count,sizeof (float),64);
    offset_gradient=layer_gradient+total_count-rows;

}



bool ConvolutionalLayer:: CalcInternalParams() {
    this->dense_filter_length=m_input_volume.depth*m_filter.height*m_filter.width;
    this->dense_weights_count=dense_filter_length*m_output_volume.depth+m_output_volume.depth;

    /* Filter stride check*/
    if ((this->m_input_volume.height-m_filter.height)%m_filter.stride!=0
            ||(this->m_input_volume.width-m_filter.width)%m_filter.stride!=0) {
        std::cout << "Window stride leads to necessity of input padding. This functionality isn't ready yet\n";
        return false;
    }

    this->m_output_volume.height=(this->m_input_volume.height-m_filter.height)/m_filter.stride+1;
    this->m_output_volume.width=(this->m_input_volume.width-m_filter.width)/m_filter.stride+1;
    rows=this->m_output_volume.height*this->m_output_volume.width*m_output_volume.depth;
    cols=m_input_volume.depth*m_input_volume.height*m_input_volume.width;
    total_count=rows*cols+rows;
    return true;
}


float *getMatrixPointer(CBLAS_LAYOUT layout,int row, int col, int rows_count, int cols_count, float *mat) {
    if (layout==CblasRowMajor)
        return &mat[col + row*cols_count];
    else
        return &mat[row + col*rows_count];
}

int getMatrixIndex(CBLAS_LAYOUT layout,int row, int col, int rows_count, int cols_count) {
    if (layout==CblasRowMajor)
        return col + row*cols_count;
    else
        return row + col*rows_count;
}


/* Mapping is performed only once, when layer is being connected to others within the net
   Major issue might be with matrix layout, while all input width and heigth is equal nothing will go wrong, but when input dimensions differ I
   don't know what will follow and it should be addressed
*/
void ConvolutionalLayer::MapLayer() {
    int sparse_index=0;
    int f_row, f_col;
    for (int f=0;f<m_output_volume.depth;f++) {
        for (int y=0;y<m_output_volume.height;y++) {
            for (int x=0;x<m_output_volume.width;x++) {
                f_row=x+y*m_output_volume.width+f*m_output_volume.height*m_output_volume.width; /*one filter gives ((input_width-filter_width)/stride+1)x((input_height-filter_height)/stride+1)
                                                                              or m_output_volume.width*m_output_volume.height
                                                                              rows to the sparse matrix*/
                pointerB[f_row]=f_row*m_filter.height*m_filter.width*m_input_volume.depth; /* begining of row is rows index multiplied by the number of non-zero values in one row which equals filter length X*/
                pointerE[f_row]=pointerB[f_row]+m_filter.height*m_filter.width*m_input_volume.depth;
                for (int d=0;d<m_input_volume.depth;d++) {
                    for (int wy=0;wy<m_filter.height;wy++) {
                        for (int wx=0;wx<m_filter.width;wx++) {
                            f_col=d*m_input_volume.height*m_input_volume.width // input filter column shift
                                    + (y*m_filter.stride+wy)*m_input_volume.width+(x*m_filter.stride+wx); //window shift
                            columns[sparse_index]=f_col;
                            pointers_to_gradient_values[sparse_index]=getMatrixPointer(CblasColMajor,f_row,f_col,rows,cols,layer_gradient);
                            sparse_index++;
                        }
                    }
                }
            }
        }
    }

    descrA.type=SPARSE_MATRIX_TYPE_GENERAL;
    auto stat=mkl_sparse_s_create_csr(&A, SPARSE_INDEX_BASE_ZERO,rows,cols,&pointerB[0], &pointerE[0],&columns[0],&values[0]);
}

void ConvolutionalLayer::CopyValues(float *src) {
    for (int f=0;f<m_output_volume.depth;f++) {
        for (int shift=0;shift<m_output_volume.height*m_output_volume.width;shift++) {
            std::memcpy(&values[f*dense_filter_length*m_output_volume.height*m_output_volume.width+shift*dense_filter_length],&src[f*dense_filter_length],dense_filter_length*sizeof (float));
        }
    }
    int dummy1=m_output_volume.height*m_output_volume.width;
    float* dummy2=offset_vector;
    for (int f=0;f<m_output_volume.depth;f++ ) {
        std::fill(dummy2,dummy2+dummy1,offset_vector_dense[f]);
        dummy2+=dummy1;
    }

}
void ConvolutionalLayer::SeedWeights(const float lower_bound, const float upper_bound) {
    std::random_device dev;
    std::default_random_engine e1(dev());
    std::uniform_real_distribution<float> dist(lower_bound,upper_bound);

    for (int f=0;f<m_output_volume.depth;f++) {
        for (int w=0;w<dense_filter_length;w++) {
            values_source[w+f*dense_filter_length]=dist(e1);
        }
        offset_vector_dense[f]=dist(e1); // offset vector seed
    }
    this->CopyValues(values_source);

    if (weights_logging)
        weights_log.insert(weights_log.end(),&values_source[0],&values_source[dense_weights_count]);
}

float *ConvolutionalLayer::RunWX(float *X, float *dest) {
    std::memcpy(dest,offset_vector,sizeof(float)*rows); // perform offset
    mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0f,A,descrA,X,1.0f,dest); // aWX+bY
    return nullptr;
}

void ConvolutionalLayer::RunBackwards(float *vector, float *destination,float *derivative, float *poolingMap) {
    vsMul(this->rows,vector,derivative,derivative);
    mkl_sparse_s_mv(SPARSE_OPERATION_TRANSPOSE,1.0f,A,descrA,vector,0.0f,destination);
}

void ConvolutionalLayer::UpdateWeights(const int batchSize, const float learning_rate) {

    if (open) {
        int filter_rows=m_output_volume.height*m_output_volume.width;
        float learning_coeff=-learning_rate/(float)batchSize;
        float *off_grad=offset_gradient;
        for (int f=0;f<m_output_volume.depth;f++) {
            for (int f_row=0;f_row<filter_rows;f_row++) {
                for (int g=0;g<dense_filter_length;g++) {
                    gradient_values_source[f*dense_filter_length+g]+=*pointers_to_gradient_values[f*dense_filter_length*filter_rows+f_row*dense_filter_length+g];
                }
            }
            offset_gradient_source[f]=std::accumulate(off_grad, off_grad+filter_rows,0.0f); //calc grad of offset
            off_grad+=filter_rows;
        }
        if (max_change) {
            float max_grad=max_change/-learning_coeff;
            for (int i=0;i<dense_filter_length*m_output_volume.depth+m_output_volume.depth;i++) {
                if (gradient_values_source[i]>max_grad)
                    gradient_values_source[i]=max_grad;
                else if(gradient_values_source[i]<-max_grad)
                    gradient_values_source[i]=-max_grad;
            }
        }

        cblas_saxpy(dense_filter_length*m_output_volume.depth+m_output_volume.depth,learning_coeff,gradient_values_source,1,values_source,1);
        this->CopyValues(values_source);
        if (weights_logging)
            weights_log.insert(weights_log.end(),&values_source[0],&values_source[dense_weights_count]); //copying weights to the log
    }

    std::memset(layer_gradient,0,total_count*sizeof(float));
    std::memset(gradient_values_source,0,dense_weights_count*sizeof(float));
}


void ConvolutionalLayer::print(bool weights) const {

    LOG << "Layer type: convolutional, ";
    LOG << "Size: " << m_filter.height << "x" << m_filter.width << "x" << m_output_volume.depth << ", " << "stride " << m_filter.stride << ", ";
    LOG << "type: " << m_activation_function->print() << "\n";
    if (weights) {
        for (int i=0;i<m_output_volume.depth;i++) {
            for (int j=0;j<dense_filter_length;j++) {
                LOG << values_source[i*dense_filter_length+j] << " ";
            }
            LOG << "\n";
        }
    }
}

void ConvolutionalLayer::printNeuron(const int no) const {
    if (no >=m_output_volume.depth) {
        std::cout << "filter count is less then no\n";
        return;
    }
    std::cout << "Weights of " << no << " neuron:\n";
    int newline=0;
    for (int i=0;i<dense_filter_length;i++) {
        std::cout <<  std::fixed << std::setprecision(4) << values_source[i+no*dense_filter_length] << " ";
        newline++;
        if (newline%m_filter.width==0) std::cout << "\n";
    }
    std::cout << "b: " << values_source[dense_weights_count-dense_weights_count+no] << "\n";
}

void ConvolutionalLayer::printGradient() const {

    LOG << "Layer type: convolutional, ";
    LOG << "Size: " << m_filter.height << "x" << m_filter.width << "x" << m_output_volume.depth << ", " << "stride " << m_filter.stride << ", ";
    LOG << "type: " << m_activation_function->print() << "\n";
    LOG << "Gradients:\n";
    for (int i=0;i<m_output_volume.depth;i++) {
        for (int j=0;j<dense_filter_length;j++) {
            LOG << gradient_values_source[i*dense_filter_length+j] << " ";
        }
        LOG << "\n";
    }

}

void ConvolutionalLayer::CreateBackup() {
    int val_count=m_input_volume.depth*m_filter.height*m_filter.width*rows;
    if (!echo_values) echo_values=(float*)mkl_malloc(val_count*sizeof (float),64);
    std::memcpy(echo_values,values,val_count*sizeof(float));
}

void ConvolutionalLayer::Rollback() {
    int val_count=m_input_volume.depth*m_filter.height*m_filter.width*rows;
    std::memcpy(values,echo_values,val_count*sizeof(float));
}


void ConvolutionalLayer::SaveStd(std::fstream &stream) {

    int ltype=static_cast<int>(this->Type());
    int atype=static_cast<int>(this->m_activation_function->type());
    int ww=m_filter.width;
    int wh=m_filter.height;
    int ws=m_filter.stride;
    int fc=m_output_volume.depth;
    int vh=m_input_volume.height;
    int vw=m_input_volume.width;
    int vd=m_input_volume.depth;

    std::vector<int> data_for_stream={ltype, atype, ww,wh,ws,fc,vh,vw,vd};
    for (auto d : data_for_stream) UploadToStream(&d,stream,1);
    UploadToStream(this->values_source,stream, dense_weights_count);
}



void ConvolutionalLayer::LoadStd(std::fstream &stream) {
    ActivationFunctionType act_func;
    LayerType m_string_type;

    std::vector<int> data_from_stream(7);

    m_string_type=static_cast<LayerType>(DownloadIntFromStream(stream));
    act_func=static_cast<ActivationFunctionType>(DownloadIntFromStream(stream));
    for (auto &d : data_from_stream) d=DownloadIntFromStream(stream);


    setActivationFunction(act_func);
    m_filter.width=data_from_stream[0];
    m_filter.height=data_from_stream[1];
    m_filter.stride=data_from_stream[2];
    m_output_volume.depth=data_from_stream[3];

    m_input_volume.height=data_from_stream[4];
    m_input_volume.width=data_from_stream[5];
    m_input_volume.depth=data_from_stream[6];

    CalcInternalParams();

    char *buffer=new char[sizeof(float)*dense_weights_count];
    stream.read(reinterpret_cast<char*>(buffer),dense_weights_count*sizeof(float));
    float *mem_p=(float*)(buffer);

    this->assignMemory(mem_p);
    this->CopyValues(values_source);
}

std::vector<float> ConvolutionalLayer::getlog(int weight_no) {
    std::vector<float> log;
    log.reserve(weights_log.size()/dense_weights_count);
    for (size_t i=weight_no;i<weights_log.size();i=i+dense_weights_count) {
        log.emplace_back(weights_log[i]);
    }
    return log;
}



}
