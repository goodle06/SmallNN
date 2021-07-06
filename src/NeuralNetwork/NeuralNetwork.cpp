#include <Proj/NeuralNetwork/NeuralNetwork.h>

namespace NN {


void NeuralNetwork::RegisterLossFunctions() {
    lossFunctionFactory.Register("softmax", &SpawnLossFunction<SoftmaxLoss>);
    lossFunctionFactory.Register("binary", &SpawnLossFunction<MultiLabelCrossEntropyLoss>);
}
    
void NeuralNetwork::RegisterComponents() {
    this->RegisterLayers();
    this->RegisterLossFunctions();
}

void NeuralNetwork::RegisterLayers() {
    layerFactory.Register("fully_connected", &SpawnLayer <Layer>);
    layerFactory.Register("convolutional", &SpawnLayer <ConvolutionalLayer>);
    layerFactory.Register("pooling", &SpawnLayer <PoolingLayer>);
}

NeuralNetwork::NeuralNetwork() {
    this->RegisterComponents();
}

NeuralNetwork::NeuralNetwork(const std::string& filename) {
    this->RegisterComponents();
    this->loadStd(filename);
}


void NeuralNetwork::addLayer(Layer *layer) {
    layers.push_back(layer); 
    status.tel.at("layers")=true;
    status.tel.at("connections")=false;
    status.tel.at("weights")=false;
}

void NeuralNetwork::train(TrainingOptions new_options) {
    options=new_options;
    this->train();
}

void NeuralNetwork::train(const int epochs, const int batchSize, const float learning_rate) {
    options=TrainingOptions({epochs,batchSize,learning_rate});
    this->train();
}

void NeuralNetwork::train() {
    this->CreateBackup();

    training_telemetry.clear();    
    auto start_time=std::chrono::steady_clock::now();
    status.tel.at("learning rate")=true;
    if (!status.is_ready()) {
        LOG << "ERROR: net isn't ready";
        return;
    }
    if (!options.isSet()) {
        LOG << "ERROR: training options aren't set\n";
        return;
    }

    auto ctm=std::chrono::system_clock::now();
    std::time_t local_time=std::chrono::system_clock::to_time_t(ctm);
    std::tm dst{};
    localtime_s(&dst, &local_time);

    LOG << "Training started at " << std::put_time(&dst, "%c %Z") << "\n";
    LOG << "Training will continue for " << options.epoches << " epoches\n";

    int j;
    size_t trainSamplesCount=trainBlob->size();
    int batchCount=trainSamplesCount/options.batchSize;
    std::vector<int> deck(trainSamplesCount);
    std::iota(deck.begin(),deck.end(),0);
    std::random_device dev;
    std::default_random_engine e1(dev());

    for (int epoch=0; epoch<options.epoches; epoch++) {
        std::shuffle(deck.begin(),deck.end(),e1);
        for (int i=0; i<=batchCount;i++) {
#if RUN_PARALLEL
        std::vector<std::future<void>> futures;
#endif
            for (j=0; j<options.batchSize&&i*options.batchSize+j<trainSamplesCount; j++) {
                int no=deck[i*options.batchSize+j];
#if RUN_PARALLEL
                futures.push_back(std::async(&NeuralNetwork::RunMKL, this, no));
#else
                RunMKL(no);
#endif
            }
#if RUN_PARALLEL
            for (auto& f : futures)
                f.get();
#endif
            UpdateWeights(j+1);
        }

        training_telemetry.loss_progress.push_back(RunOnce(false,trainBlob));
        if (testBlob) training_telemetry.loss_progress_test.push_back(RunOnce(false,testBlob));
        LOG << "Training progress: " << epoch + 1 << " epochs out of " << options.epoches << "\r";
    }
    LOG << "Train samples run:\n";
    RunOnce(true,trainBlob);

    training_duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time);
    print();
}


void NeuralNetwork::RunForward(float *X, float *WX) {
    for (auto it=layers.begin(); it!=layers.end(); ++it){
        auto p=(*it)->RunWX(X,WX);
        (*it)->RunActivate(WX,X);
        mkl_free(p);
    }
}

void NeuralNetwork::RunForward(float *X, float *WX, std::vector<float*> &poolingMaps, float *Derivative ) {
    if (Derivative) {
        int k=net_input_sz;
        for (size_t it=0; it<layers.size(); ++it){
            float *p=layers[it]->RunWX(X-(layers[it])->GetCols()+k,WX+k);
            layers[it]->RunDerivative(WX+k,Derivative+k);
            layers[it]->RunActivate(WX+k,X+k);
            poolingMaps[it]=p;
            k+=layers[it]->GetRows();
        }
    }
    else {
        RunForward(X, WX);
    }
}

#if RUN_PARALLEL
    //std::mutex grad_mutex;
#endif

void NeuralNetwork::RunMKL(const int no) {

    float* X, *WX, *D, *Z;
    float *src1, *src2, *src3, *src4;
    std::vector<float*> poolingMaps(layers.size());

    /* Allocate memory */
    src1=(float *)mkl_malloc((neurons_total+net_input_sz)*sizeof( float ),64);
    src2=(float *)mkl_malloc((neurons_total+net_input_sz)*sizeof( float ),64);
    src3=(float *)mkl_malloc((neurons_total+net_input_sz)*sizeof( float ),64);
    src4=(float *)mkl_malloc((neurons_total+net_input_sz)*sizeof( float ),64);
    /* --- */
    std::memcpy(src1,trainBlob->at(no).X,net_input_sz*sizeof (float)); // copying input data

    X=src1;
    WX=src2;
    D=src3;
    Z=src4;

    this->RunForward(X,WX,poolingMaps,D);
    int k=neurons_total+net_input_sz-layers.back()->GetRows();
    lossFunc->CalculateDerivative(&X[k], Z, net_output_sz,trainBlob->at(no).Y);

    size_t poolingMapsCounter=poolingMaps.size()-1;
    for (auto it=layers.rbegin();it!=layers.rend();++it) {
        (*it)->RunBackwards(Z,WX, D+k, poolingMaps[poolingMapsCounter]);
#if RUN_PARALLEL
       // grad_mutex.lock();
#endif
        (*it)->CalculateGradient(D+k,X+k-(*it)->GetCols()); // Gradient is a shared memory
#if RUN_PARALLEL
      //  grad_mutex.unlock();
#endif

        poolingMapsCounter--;
        Z=WX;
        k-=(*it)->GetCols();
    }

    /* Free all allocated memory */
    mkl_free(src1);
    mkl_free(src2);
    mkl_free(src3);
    mkl_free(src4);
    /* ---- */
}

void NeuralNetwork::RunOnce() {
    RunOnce(true, currentDataset);
}

Estimate NeuralNetwork::getEstimate(float *Z) {
    float *Y, *X;
    float *src1, *src2;
    src1=(float *)mkl_malloc(net_rows*sizeof( float ),64);
    src2=(float *)mkl_malloc(net_rows*sizeof( float ),64);

    std::memcpy(src1,Z,net_input_sz*sizeof (float));
    X=src1;
    Y=src2;
    this->RunForward(X,Y);

    Estimate result(X,net_output_sz,this->lossFunc);
    mkl_free(src1);
    mkl_free(src2);
    return result;
}

float NeuralNetwork::RunOnce(bool print_est,DataBlob* blob) {
    m_loss=0;
    int correctly_classified=0;
    int sb_correctly_classified=0;
    size_t number_of_samples=blob->size();
    float* X, *Y;
    float *src1, *src2;
    src1=(float *)mkl_malloc(net_rows*sizeof( float ),64);
    src2=(float *)mkl_malloc(net_rows*sizeof( float ),64);
    for (int i=0; i<number_of_samples; i++) {
        std::memcpy(src1,blob->at(i).X,net_input_sz*sizeof (float));
        X=src1;
        Y=src2;
        this->RunForward(X,Y);
        m_loss+=lossFunc->CalculateLoss(X,net_output_sz,blob->at(i).Y);
        if (print_est) {
            Estimate result(X,net_output_sz,this->lossFunc,blob->at(i).Y);
            if(result.isCorrect(0)) correctly_classified++;
            else {
                LOG << "1st/2nd/actual "; 
                result.print(2, blob->translation);
                std::cout << "(#" << i << ")\n";
                if (result.isCorrect(1)) sb_correctly_classified++;
            }

        }
    }

    if (print_est) {
        LOG << "Average loss: " << std::setprecision(6) << m_loss/(number_of_samples*1.0f) << "\n";
        LOG << "Number of samples: " << number_of_samples << "\n";
        LOG << "Correctly estimated: " << correctly_classified << "\n";
        LOG << "Stats:\nAccuracy: " << std::setprecision(4) << correctly_classified*1.0f/(number_of_samples*1.0f)*100.0f << "%\n";
        LOG << "Second estimate accuracy:" << (correctly_classified+sb_correctly_classified)*1.0f/(number_of_samples*1.0f)*100.0f << "%\n";
    }
    mkl_free(src1);
    mkl_free(src2);
    return m_loss/(float)blob->size();
}


void NeuralNetwork::addTrainingData(NN::DataBlob *blob) {
    trainBlob=blob;
    currentDataset=trainBlob;
    status.tel.at("training data")=true;
    std::cout << "Training data is loaded\n";
    trainBlob->print();
}

void NeuralNetwork::addTestingData(NN::DataBlob *blob) {
    testBlob=blob;
    status.tel.at("testing data")=true;
    std::cout << "Test data is loaded\n";
    testBlob->print();
}

int NeuralNetwork::getNeuronsCount() {
    neurons_total=0;
    for (auto it=layers.begin();it!=layers.end(); ++it)
        neurons_total+=(*it)->GetRows();
    return neurons_total;
}

int NeuralNetwork::getWeightsCount() {
    weights_total=0;
    for (auto it=layers.begin();it!=layers.end(); ++it)
        weights_total+=(*it)->GetTotalCount();
    return weights_total;
}

bool NeuralNetwork::connect() {

    if (trainBlob->translation.size()!=layers.back()->GetRows()) LOG << "ERROR: Output dimension (" << layers.back()->GetRows() << ") doesn't match number of classes (" << trainBlob->translation.size() << ")\n";

    layers.front()->SetConnections(*trainBlob);
    for (auto it=layers.begin()+1;it!=layers.end();it++) {
        if(!(*it)->SetConnections(*(it-1))) return false;
    }

    net_input_sz=layers.front()->GetCols();
    net_output_sz=layers.back()->GetRows();
    getWeightsCount();
    getNeuronsCount();
    auto sz2=(*std::max_element(layers.begin(),layers.end(),[](const auto& l1, const auto& l2){return   l1->GetRows()<l2->GetRows();}))->GetRows();
    net_rows=(sz2>trainBlob->vector_size()? sz2:trainBlob->vector_size());

    status.tel.at("connections")=true;
    return true;
}



void NeuralNetwork::SeedWeights(const float lower, const float upper) {
    for (auto &l : layers) l->SeedWeights(lower,upper);
    status.tel.at("weights")=true;
}

void NeuralNetwork::UpdateWeights(const int batchSize) {
    for (auto &layer : layers) {
        layer->UpdateWeights(batchSize,options.learningRate);
    }
}

void NeuralNetwork::SetLossFunction(LossFunction *function) {
    lossFunc=function;
    lossFunc->print();
    status.tel.at("loss function")=true;
}

void NeuralNetwork::SetLossFunction(LossFunctionType type) {
    switch (type) {
    case LossFunctionType::Quadratic:
        lossFunc=new QuadraticLossFuntion;
        status.tel.at("loss function")=true;
        break;
    case LossFunctionType::Softmax:
        lossFunc=new SoftmaxLoss;
        status.tel.at("loss function")=true;
        break;
    case LossFunctionType::BinaryCrossEntropy:
        lossFunc=new MultiLabelCrossEntropyLoss;
        status.tel.at("loss function")=true;
        break;
    default:
        LOG << "unknown type";
        break;
    }
}

void NeuralNetwork::SetLossFunction(std::string type) {
    if (type.find("quadratic")!=std::string::npos) SetLossFunction(LossFunctionType::Quadratic);
    else if(type.find("softmax")!=std::string::npos) SetLossFunction(LossFunctionType::Softmax);
    else if(type.find("multiLabelCrossEntropy")!=std::string::npos) SetLossFunction(LossFunctionType::BinaryCrossEntropy);
    else LOG << "ERROR: unknown loss function\n";
}

NeuralNetwork::~NeuralNetwork() {
    for (int i = 0; i < layers.size(); i++) delete layers[i];
    delete testBlob;
    delete trainBlob;
}

void NeuralNetwork::saveStd(const std::string& filename) const {

    int s_netSz=weights_total;
    int s_netInSz=net_input_sz;
    int s_netOutSz=net_output_sz;
    int s_netRows=net_rows;
    int ls=(int)layers.size();
    int s_loss=static_cast<int>(lossFunc->type());

    std::fstream out(filename,std::ios_base::out | std::ios_base::binary);
    if (out.is_open()) {
        UploadToStream(&s_netSz,out,1);
        UploadToStream(&s_netInSz,out,1);
        UploadToStream(&s_netOutSz,out,1);
        UploadToStream(&s_netRows,out,1);
        UploadToStream(&s_loss,out,1);
        UploadToStream(&ls,out,1);

        for (auto &l : layers) {
            int ltype=static_cast<int>(l->Type());
            UploadToStream(&ltype,out,1);
        }
        for (auto &l : layers)
            l->SaveStd(out);
        options.saveStd(out);

        out.close();
        std::cout << "Net " << net_name << " saved\n";
    }
    else {
        std::cerr << "ERROR: couldn't open stream for uploading net" << std::endl;
    }

}



void NeuralNetwork::TrainingOptions::saveStd(std::fstream &stream) const{
    int ep=epoches;
    int bs=batchSize;
    float lr=learningRate;
    UploadToStream(&ep,stream,1);
    UploadToStream(&bs,stream,1);
    UploadToStream(&lr,stream,1);
}

void NeuralNetwork::TrainingOptions::loadStd(std::fstream &stream) {
    int ep, batch;
    float lr;
    stream.read(reinterpret_cast<char*>(&ep),sizeof(int));
    stream.read(reinterpret_cast<char*>(&batch),sizeof(int));
    stream.read(reinterpret_cast<char*>(&lr), sizeof(float));
    epoches=ep;
    batchSize=batch;
    learningRate=lr;
}

bool NeuralNetwork::TrainingOptions::isSet() {
    if (batchSize>0&&learningRate>0.0f&&epoches>0) return true;
    else return false;
}

void NeuralNetwork::TrainingOptions::print() const {
    LOG << "Training options:\nEpoches: " << epoches << ", ";
    LOG << "batch size: " << batchSize << ", ";
    LOG << "learing rate: " << learningRate << "\n";
}

void NeuralNetwork::loadStd(const std::string& filename) {
    layers.clear();
    std::fstream in(filename, std::ios_base::in | std::ios_base::binary);
    if (in.is_open()) {
        int s_netSz, s_netInSz, s_netOutSz, s_netRows, s_layer_c;;
        LossFunctionType s_loss;
        s_netSz=DownloadIntFromStream(in);
        s_netInSz=DownloadIntFromStream(in);
        s_netOutSz=DownloadIntFromStream(in);
        s_netRows=DownloadIntFromStream(in);
        s_loss=static_cast<LossFunctionType>(DownloadIntFromStream(in));
        s_layer_c=DownloadIntFromStream(in);


        SetLossFunction( s_loss);
        status.tel.at("loss function")=true;
        weights_total=s_netSz;
        net_input_sz=s_netInSz;
        edge_length=std::sqrt(net_input_sz);
        net_output_sz=s_netOutSz;
        net_rows=s_netRows;


        std::vector<LayerType> layers_types;
        LayerType lType;
        for (int i=0;i<s_layer_c;i++) {
            lType=static_cast<LayerType>(DownloadIntFromStream(in));
            layers_types.push_back(lType);
        }

        for (auto ltype : layers_types) {
            if (ltype==LayerType::FC) {
                Layer *layer=new Layer;
                layer->LoadStd(in);
                this->addLayer(layer);
            }
            else if(ltype==LayerType::Conv) {
                ConvolutionalLayer *layer=new ConvolutionalLayer;
                layer->LoadStd(in);
                this->addLayer(layer);
            }
            else if (ltype==LayerType::MaxPool) {
                PoolingLayer *layer=new PoolingLayer;
                layer->LoadStd(in);
                this->addLayer(layer);
            }
        }

        options.loadStd(in);

        getWeightsCount();
        getNeuronsCount();
        status.tel.at("weights")=true;
        status.tel.at("connections")=true;
        this->printConfiguration();
        in.close();
    }
    else {
        std::cerr << "ERROR: couldn't open in stream of the net file" << std::endl;
    }
}

void NeuralNetwork::clear() {
    
    for (int i = 0; i < layers.size(); i++) delete layers[i];
    layers.clear();
    delete testBlob;
    testBlob = nullptr;
    delete trainBlob;
    trainBlob = nullptr;

    lossFunc=nullptr;
    edge_length=0;
    net_name="";

    weights_total=0;
    net_rows=0;
    net_input_sz=0;
    net_output_sz=0;
    m_loss=0.0f;
}

void NeuralNetwork::printConfiguration() const {
    LOG << "Input dimension: " << net_input_sz << "(" << edge_length << "x" << edge_length << ")\n";
    LOG << "Classes count: " << net_output_sz << "\n";
    LOG << "Net rows: " << net_rows << "\n";
    LOG << "Net size: " << weights_total << "\n";
    for(auto& l : layers) l->print();
    if (lossFunc) LOG << "Loss function: " << lossFunc->print() << "\n";
    this->printTrainingParameters();
}

void NeuralNetwork::printTrainingParameters() const {
    LOG << "Training options:\nEpoches: " << options.epoches << ", ";
    LOG << "batch size: " << options.batchSize << ", ";
    LOG << "learing rate: " << options.learningRate << "\n";
}

void NeuralNetwork::printInputParameters() const {
    if (trainBlob) trainBlob->print();
}

void NeuralNetwork::LoadNetFromFile() {
    std::string filename = OpenWindowsFileDialog("Nets", ".net", "K:\\Nets\\");
       // QFileDialog::getOpenFileName(nullptr, "Load net", "K:\\Nets\\", "Net Files (*.net)");
    if (!filename.empty()) {
        auto pos=filename.find(".net");
        std::string nname=filename.substr(pos);
        size_t ind=nname.length()-nname.find_last_of("/");
        nname=nname.substr(ind-1);
        net_name=nname;
        LOG << "Net " << net_name << " is loaded\n";
//        load(filename);
        loadStd(filename);
    }
    else {
        LOG << "ERROR: net wasn't loaded\n";
    }
}

void NeuralNetwork::RollBack() {
    for (auto &layer : layers) layer->Rollback();
}
void NeuralNetwork::CreateBackup() {
    for (auto &layer : layers) layer->CreateBackup();
}

bool NeuralNetwork::RunCommand(std::string command) {
    std::string fnameTrain="K:\\Blobs\\ConvolutionalBlobTrain.blob";
    std::string fnameTest="K:\\Blobs\\ConvolutionalBlobTest.blob";
    std::string fnameTranslation="K:\\TranslationUnits\\CharTranslationUnit.trun";

    if (command == "reset"|| executioner == nullptr) executioner = this;

    NetCommand cmd(command);
    executioner->ExecuteCommand(cmd);

    if (command=="default") {
        std::string TableTrainBlob="K:\\Blobs\\TrainTTB.blob";
        std::string TableTestBlob="K:\\Blobs\\TestTTB.blob";
        std::string TableTranslation="K:\\TranslationUnits\\TTB.trun";
        NN::ConvolutionalLayer *conv1=new NN::ConvolutionalLayer(3,3,3,16, NN::ActivationFunctionType::RELU);
        this->addLayer(conv1);
        NN::Layer* fc1 = new NN::Layer({ 24 });
        fc1->setActivationFunction("relu");
        NN::Layer* fc2 = new NN::Layer({ 2 });
        fc2->setActivationFunction("logistic");
        this->addLayer(fc1);
        this->addLayer(fc2);

        NN::DataBlob* blobOne=new NN::DataBlob(TableTrainBlob, TableTranslation);
        NN::DataBlob* blobTwo=new NN::DataBlob(TableTestBlob, TableTranslation);
        edge_length=24;
        blobOne->resize(edge_length,true);
        blobTwo->resize(edge_length,true);
        this->addTrainingData(blobOne);
        this->addTestingData(blobTwo);
        this->connect();

        Transformation t1(TransformationType::sin);
        blobOne->transform(t1);
        blobTwo->transform(t1);

        this->SetLossFunction(NN::LossFunctionType::BinaryCrossEntropy);

        this->SeedWeights(-0.2f,0.2f);
        return true;
    }

    if (command=="default2") {
        NN::Layer* fc1 = new NN::Layer({ 56 });
        fc1->setActivationFunction("logistic");
        this->addLayer(fc1);

        NN::DataBlob* blobOne=new NN::DataBlob(fnameTrain, fnameTranslation);
        NN::DataBlob* blobTwo=new NN::DataBlob(fnameTest, fnameTranslation);
        edge_length=24;
        blobOne->resize(edge_length,true, false);
        blobTwo->resize(edge_length,true, false);
        this->addTrainingData(blobOne);
        this->addTestingData(blobTwo);
        this->connect();

        Transformation t1(TransformationType::sin);
        blobOne->transform(t1);
        blobTwo->transform(t1);

        this->SetLossFunction(NN::LossFunctionType::BinaryCrossEntropy);

        this->SeedWeights(-0.3f,0.3f);
        return true;
    }

    if (command=="default3") {
        //NN::ConvolutionalLayer *conv1=new NN::ConvolutionalLayer(8, 4,4,2, NN::ActivationFunctionType::RELU);
        NN::ConvolutionalLayer* conv1 = new NN::ConvolutionalLayer({ 8,4,4,2 });
        conv1->setActivationFunction("relu");
        NN::PoolingLayer *pool1=new NN::PoolingLayer(2,2,2);
        NN::ConvolutionalLayer *conv2=new NN::ConvolutionalLayer(8, 3,3,1, NN::ActivationFunctionType::RELU);
        NN::Layer* fc1 = new NN::Layer({ 56 });
        fc1->setActivationFunction("logistic");
        this->addLayer(conv1);
        this->addLayer(pool1);
        this->addLayer(conv2);
        conv1->SetMaximumWeightChange(0.1f);
        conv2->SetMaximumWeightChange(0.1f);
        this->addLayer(fc1);

        NN::DataBlob* blobOne=new NN::DataBlob(fnameTrain, fnameTranslation);
        NN::DataBlob* blobTwo=new NN::DataBlob(fnameTest, fnameTranslation);
        edge_length=22;
        blobOne->resize(edge_length,true);
        blobTwo->resize(edge_length,true);
        this->addTrainingData(blobOne);
        this->addTestingData(blobTwo);
        if (this->connect()) {
            Transformation t1(TransformationType::sin);
            blobOne->transform(t1);
            blobTwo->transform(t1);
            this->SetLossFunction(NN::LossFunctionType::BinaryCrossEntropy);
            this->SeedWeights(-0.2f,0.2f);
        }
        this->print();
        return true;
    }

    if (command=="default4") {
        NN::ConvolutionalLayer *conv1=new NN::ConvolutionalLayer(5,5,2,16, NN::ActivationFunctionType::RELU);
        NN::ConvolutionalLayer *conv2=new NN::ConvolutionalLayer(3,3,1,16, NN::ActivationFunctionType::RELU);
        NN::ConvolutionalLayer *conv3=new NN::ConvolutionalLayer(3,3,1,16, NN::ActivationFunctionType::RELU);
        this->addLayer(conv1);
        this->addLayer(conv2);
        this->addLayer(conv3);
        NN::Layer* fc1 = new NN::Layer({ 56 });
        fc1->setActivationFunction("logistic");
        this->addLayer(fc1);

        NN::DataBlob* blobOne=new NN::DataBlob(fnameTrain, fnameTranslation);
        NN::DataBlob* blobTwo=new NN::DataBlob(fnameTest, fnameTranslation);
        edge_length=25;
        blobOne->resize(edge_length,true, true);
        blobTwo->resize(edge_length,true, true);
        this->addTrainingData(blobOne);
        this->addTestingData(blobTwo);
        this->connect();

        Transformation t1(TransformationType::sin);
        blobOne->transform(t1);
        blobTwo->transform(t1);

        this->SetLossFunction(NN::LossFunctionType::BinaryCrossEntropy);

        this->SeedWeights(-0.2f,0.2f);
        return true;
    }
    return false;
}

bool NeuralNetwork::LoadConfigFromFile(std::string config_filename)
{
    std::fstream file(config_filename, std::ios::in);
    std::string cmd;
    if (file.is_open()) {
        while (!file.eof()) {
            std::getline(file, cmd);
            RunCommand(cmd);
        }
        return true;
    }
    else return false;
}

bool NeuralNetwork::LoadConfigFromFile()
{
    std::string config_filename = OpenWindowsFileDialog("net config", "netcf");
    return LoadConfigFromFile(config_filename);
}

void NeuralNetwork::print() const {
    options.print();
    if (trainBlob) trainBlob->print();
    for (auto& l : layers) l->print();

    LOG << "Loss function: " << lossFunc->print() << "\n";
    LOG << "Elapsed time: " << training_duration.count() << " seconds\n";
}

DataBlob* NeuralNetwork::GetCurrentDataset() {
    return currentDataset;
}

void NeuralNetwork::SelectDataset(std::string which_one) {
    if (which_one == "train" && trainBlob) currentDataset = trainBlob;
    else if (which_one == "test" && testBlob) currentDataset = testBlob;
}

void NeuralNetwork::ExecuteCommand(NetCommand &cmd) {

    if (cmd.GetCommand() == "commands") {
        std::cout
            << "load_config\n"
            << "add_layer\n"
            << "set_loss\n"
            << "train\n"
            << "seed_weights\n"
            << "estimate\n"
            << "run\n"
            << "load_train_data\n"
            << "load_test_data\n"
            << "select_dataset\n"
            << "select_layer\n"
            << "save\n"
            << "load\n"
            << "print\n"
            << "rollback\n"
            << "display_loss_graph\n"
            << "connect\n";
    }

    if (cmd.GetCommand()== "load_config") {
        std::string config_filename = cmd.GetParameter();
        if (config_filename.empty()) LoadConfigFromFile();
        else LoadConfigFromFile(config_filename);
    }

    if (cmd.GetCommand() == "add_layer") {
        Layer *layer=layerFactory.SpawnObject(cmd.GetParameter());
        if (!layer) {
            std::cout << "ERROR: Invalid layer type\n";
            return;
        }
        cmd.SetCommand("set_activation_function");
        layer->ExecuteCommand(cmd);
        cmd.SetCommand("set_parameters");
        layer->ExecuteCommand(cmd);
        addLayer(layer);
        layer->print();
        executioner = layer;
    }
    if (cmd.GetCommand() == "set_loss") {
        SetLossFunction(lossFunctionFactory.SpawnObject(cmd.GetParameter()));
    }
    if (cmd.GetCommand() == "train") {
        options.epoches = std::stoi(cmd.GetParameter());
        options.batchSize = std::stoi(cmd.GetParameter());
        options.learningRate = std::stof(cmd.GetParameter());
        train();
    }
    if (cmd.GetCommand() == "seed_weights") {
        float range = std::stof(cmd.GetParameter());
        SeedWeights(-range, range);
    }
    if (cmd.GetCommand() == "estimate") {
        int no = std::stoi(cmd.GetParameter());
        getEstimate(currentDataset->at(no).X).print(2, currentDataset->translation);
    }
    if (cmd.GetCommand() == "run") {
        RunOnce(true, currentDataset);
        print();
    }
    if (cmd.GetCommand() == "load_train_data") {
        std::string path_to_blob = cmd.GetParameter();
        std::string path_to_translation = cmd.GetParameter();
        if (path_to_blob.empty()) 
            path_to_blob = OpenWindowsFileDialog("Datablob", ".blob", "K:\\Blobs\\");
        if (path_to_translation.empty())
            path_to_translation = OpenWindowsFileDialog("Translation", ".trun", "K\\TranslationUnits\\");
        DataBlob* blob= new DataBlob(path_to_blob, path_to_translation);
        executioner = blob;
        addTrainingData(blob);
    }

    if (cmd.GetCommand() == "load_test_data") {
        std::string path_to_blob = cmd.GetParameter();
        std::string path_to_translation = cmd.GetParameter();
        if (path_to_blob.empty())
            path_to_blob = OpenWindowsFileDialog("Datablob", ".blob", "K:\\Blobs\\");
        if (path_to_translation.empty())
            path_to_translation = OpenWindowsFileDialog("Translation", ".trun", "K\\TranslationUnits\\");
        DataBlob *blob = new DataBlob(path_to_blob, path_to_translation);
        executioner = blob;
        addTestingData(blob);
    }

    if (cmd.GetCommand() == "select_dataset") {
        SelectDataset(cmd.GetParameter());
        if (GetCurrentDataset() == nullptr) std::cout << "ERROR: Dataset is unavailable\n";
        else executioner = currentDataset;
    }

    if (cmd.GetCommand() == "select_layer") {
        std::string p = cmd.GetParameter();
        if (!p.empty()) 
            executioner = layers.at(std::stoi(p));
        else 
            std::cout << "ERROR: Please enter layer id\n";
    }
    
    if (cmd.GetCommand() == "save") {
        std::string par = cmd.GetParameter();
        if (!par.empty()) net_name = par;
        std::string folder_name = OpenWindowsFileDialog();
        std::string fname = folder_name + "\\" + net_name + ".net";
        this->saveStd(fname);
    }
    if (cmd.GetCommand() == "load") {
        LoadNetFromFile();
    }

    if (cmd.GetCommand() == "print") 
        print();

    if (cmd.GetCommand() == "rollback")
        RollBack();

    if(cmd.GetCommand()=="display_loss_graph") 
        training_telemetry.displayLossGraph();

    if (cmd.GetCommand() == "connect") {
        connect();
    }

}

}
