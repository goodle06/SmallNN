#include <Proj/NeuralNetwork/Layers/Layer.h>

namespace NN {



Layer::Layer(int dim, ActivationFunction* func) {
    RegisterActivationFunctions();
    SetSize(dim);
    m_activation_function=func;
}

Layer::Layer() {
    RegisterActivationFunctions();
}

Layer::Layer(std::initializer_list<int> params)
{
    RegisterActivationFunctions();
    this->SetParams(params);
}

void Layer::RegisterActivationFunctions() {
    activationFunctionFactoryString.Register("relu", &SpawnActivationFunction<RELUActivationFunction>);
    activationFunctionFactoryString.Register("logistic", &SpawnActivationFunction<LogisticActivationFunction>);

    activationFunctionFactoryInt.Register(static_cast<int>(ActivationFunctionType::RELU), &SpawnActivationFunction<RELUActivationFunction>);
    activationFunctionFactoryInt.Register(static_cast<int>(ActivationFunctionType::Logistic), &SpawnActivationFunction<LogisticActivationFunction>);
}

Layer::~Layer() {
    if (layer_weights) mkl_free(layer_weights);
    if (layer_gradient) mkl_free(layer_gradient);
}

void Layer::assignMemory() {
    if (!layer_weights) {
        layer_weights=(float*)mkl_malloc(total_count*sizeof (float),64);
    }
    layer_gradient=(float*)mkl_calloc(total_count,sizeof (float),64);
    offset_vector=layer_weights+total_count-rows;
    offset_gradient=layer_gradient+total_count-rows;

}

void Layer::assignMemory(float *load_weights) {
    layer_weights=load_weights;
    layer_gradient=(float*)mkl_calloc(total_count,sizeof (float),64);
    offset_vector=layer_weights+total_count-rows;
    offset_gradient=layer_gradient+total_count-rows;
}

void Layer::Rollback() {
    std::memcpy(layer_weights,echo_weights,total_count*sizeof(float));
}
void Layer::CreateBackup() {
    if (!echo_weights) echo_weights=(float*)mkl_malloc(total_count*sizeof (float),64);
    std::memcpy(echo_weights,layer_weights,total_count*sizeof(float));
}


float* Layer::RunWX(float *X, float *dest) {
    std::memcpy(dest,offset_vector,sizeof(float)*rows);
    cblas_sgemv(CblasColMajor,CblasNoTrans,rows,cols,1.0f,layer_weights,rows,X,1,1.0f,dest,1);
    return nullptr;
}

void Layer::RunActivate(float *WX, float *Y) {
    m_activation_function->calc(rows,WX,Y);
}

void Layer::RunDerivative(float *WX, float *D) {
    m_activation_function->derivative(rows,WX,D);
}

void Layer::RunBackwards(float *d, float* Y, float *derivative, float *poolingMap) {
    vsMul(this->rows,d,derivative,derivative);
    cblas_sgemv(CblasColMajor,CblasTrans,rows,cols,1.0f, layer_weights,rows,d,1,0.0f, Y, 1);
}

void Layer::CalculateGradient(float *derivative, float *X) {
#if RUN_PARALLEL
    std::lock_guard<std::mutex> lock(m_mutex);
#endif
    cblas_sger(CblasColMajor,this->rows,this->cols,1.0f,derivative,1,X,1,this->layer_gradient,this->rows);
    std::memcpy(offset_gradient,derivative,sizeof(float)*rows);
}


void Layer::setActivationFunction(ActivationFunction* actFunc) {
    m_activation_function=actFunc;
}

void Layer::setActivationFunction(std::string actFunc) {
    m_activation_function=activationFunctionFactoryString.SpawnObject(actFunc);
}

void Layer::setActivationFunction(ActivationFunctionType actFunc)
{
    m_activation_function = activationFunctionFactoryInt.SpawnObject(static_cast<int>(actFunc));
}

void Layer::SetSize(const int dim) {
    rows=dim;
}

bool Layer::SetConnections(Layer* layer) {
    cols=layer->rows;
    total_count=rows*cols+rows;
    assignMemory();
    return true;
}

void Layer::SetParams(std::initializer_list<int> params)
{
    auto p = params.begin();
    if (p != params.end()) this->rows = *p;
}

bool Layer::SetConnections(DataBlob &blob) {
    cols=blob.vector_dimension;
    total_count=rows*cols+rows;
    assignMemory();
    return true;
}

void Layer::SeedWeights(const float lower_bound, const float upper_bound) {
    std::random_device dev;
    std::default_random_engine e1(dev());
    std::uniform_real_distribution<float> dist(lower_bound,upper_bound);
    for (int i=0;i<total_count;i++) layer_weights[i]=dist(e1);
    weights_log.clear(); // clearing weights log since they are being reseeded
    if (weights_logging) weights_log.insert(weights_log.end(),&layer_weights[0],&layer_weights[total_count]); //copying weights to the log
}



void Layer::UpdateWeights(const int batchSize, const float learning_rate) {

    float rt=-learning_rate/(batchSize*1.0f);
    if (max_change) {
        float max_grad=max_change/-rt;
        for (int i=0;i<total_count;i++) {
            if (layer_gradient[i]>max_grad)
                layer_gradient[i]=max_grad;
            else if(layer_gradient[i]<-max_grad)
                layer_gradient[i]=-max_grad;
        }
    }
    cblas_saxpy(total_count,rt,layer_gradient,1,layer_weights,1);

    std::memset(layer_gradient,0,total_count*sizeof(float));
    if (weights_logging) weights_log.insert(weights_log.end(),&layer_weights[0],&layer_weights[total_count]); //copying weights to the log
}


void Layer::SaveStd(std::fstream &stream) {
    int ltype=static_cast<int>(this->Type());
    int atype=static_cast<int>(this->m_activation_function->type());
    int trows=this->rows;
    int ttotal_count=this->total_count;

    UploadToStream(&ltype,stream,1);
    UploadToStream(&atype,stream,1);
    UploadToStream(&trows,stream,1);
    UploadToStream(&ttotal_count,stream,1);
    UploadToStream(this->layer_weights,stream,this->total_count);

//    stream.write(reinterpret_cast<char*>(this->layer_weights),this->total_count*sizeof(float));
}

void Layer::LoadStd(std::fstream &stream) {
    ActivationFunctionType act_func;
    LayerType m_string_type;
    int s_rows, s_count;

    m_string_type=static_cast<LayerType>(DownloadIntFromStream(stream));
    act_func=static_cast<ActivationFunctionType>(DownloadIntFromStream(stream));
    s_rows=DownloadIntFromStream(stream);
    s_count=DownloadIntFromStream(stream);
    this->total_count=s_count;
    this->rows=s_rows;
    this->cols=(this->total_count-rows)/this->rows;
    this->setActivationFunction(activationFunctionFactoryInt.SpawnObject(static_cast<int>(act_func)));
    char *buffer=new char[sizeof(float)*s_count];
    stream.read(reinterpret_cast<char*>(buffer),s_count*sizeof(float));
    float *mem_p=(float*)(buffer);

    this->assignMemory(mem_p);
}

std::vector<float> Layer::getlog(int weight_no) {
    std::vector<float> log;
    log.reserve(weights_log.size()/total_count);
    for (size_t i=weight_no;i<weights_log.size();i=i+total_count) {
        log.emplace_back(weights_log[i]);
    }
    return log;
}

void Layer::print(bool weights) const {
    LOG << "Layer type: fully connected, ";
    LOG << "Size: " << rows << "x" << cols << ", ";
    LOG << "type: " << m_activation_function->print() << "\n";
    if (weights) {
        for (int i=0;i<cols;i++) {
            for (int j=0;j<rows;j++) {
                LOG << layer_weights[i*rows+j] << " ";
            }
            LOG << "\n";
        }
    }
}

void Layer::printNeuron(const int no) const {
    if (no >=rows) {
        std::cout << "neuron count is less then no\n";
        return;
    }
    std::cout << "Weights of " << no << " neuron:\n";
    for (int i=0;i<cols;i++) {
        std::cout <<  std::fixed << std::setprecision(4) << layer_weights[i*rows+no] << "\n";
    }
    std::cout << "b: " << offset_vector[no] << "\n";
}
void Layer::printGradient() const {
    LOG << "Layer type: FC, ";
    LOG << "Size: " << rows << "x" << cols << ", ";
    LOG << "type: " << m_activation_function->print() << "\n";
    for (int i=0;i<cols;i++) {
        for (int j=0;j<rows;j++) {
            LOG << layer_gradient[i*rows+j] << " ";
        }
        LOG << "\n";
    }
}


void Layer::ExecuteCommand(NetCommand &cmd)
{

    if (cmd.GetCommand() == "commands") {
        std::cout
            << "set_activation_function\n"
            << "freeze\n"
            << "set_max_change\n"
            << "print_neuron\n"
            << "print\n"
            << "set_parameters\n";
    }

    std::string act = cmd.GetCommand();
    if (act == "set_activation_function")
        setActivationFunction(cmd.GetParameter());

    if (act == "freeze")
        FreezeWeights();

    if (act == "set_max_change") {
        SetMaximumWeightChange(std::stof(cmd.GetParameter()));
        std::cout << "max weight change: " << max_change << "\n";
    }
    
    if (act == "print_neuron")
        printNeuron(std::stoi(cmd.GetParameter()));

    if (act == "print")
        print();

    if (act == "set_parameters") {
        std::string p;
        std::vector<int> vector_param_list;
        while (!(p= cmd.GetParameter()).empty()) {
            vector_param_list.push_back(std::stoi(p));
        }
        int* p1 = &vector_param_list.front();
        int* p2 = &vector_param_list.back();
        ++p2;
        std::initializer_list<int> param_list(p1, p2);
        SetParams(param_list);
    }



}

}
