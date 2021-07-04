#include <Proj/NeuralNetwork/Blobs/DataBlob.h>


namespace NN {


DataBlob::DataBlob(std::string blobname) {
    std::string savePath = OpenWindowsFileDialog();
    file_src=savePath+"/"+blobname + ".blob";
}

DataBlob::DataBlob(std::string filename, std::string translation_filename) {
    translation=TranslationUnit(translation_filename);
    file_src=filename;
    std::fstream in (filename, std::ios_base::in | std::ios_base::binary);
    if (in.is_open()) {
    int memory_sz;
    in >> memory_sz; //memory in quantity of type (x*sizeof(type))

    char *buffer=new char[sizeof(uchar)*memory_sz];
    in.read(reinterpret_cast<char*>(buffer), sizeof(uchar)*memory_sz);

    data=(float*)std::malloc(sizeof (float)*memory_sz);
    original_data=(uchar*)std::malloc(sizeof (uchar)*memory_sz);
    for (int i=0;i<memory_sz;i++) {
        data[i]=(float)((uchar)buffer[i]);
        original_data[i]=(uchar)buffer[i];
    }

    slice(memory_sz);
    delete [] buffer;
    }
    else {
        std::cerr << "ERROR: couldn't open filestream of blob" << std::endl;
    }
}

DataBlob::~DataBlob() {
    std::free(data);
    std::free(original_data);
}

void DataBlob::ExecuteCommand(NetCommand& cmd)
{
    if (cmd.GetCommand() == "commands") {
        std::cout
            << "resize\n"
            << "save\n"
            << "display\n"
            << "correct\n"
            << "remove\n"
            << "print_distribution\n"
            << "transform\n";
    }

    if (cmd.GetCommand() == "resize") {
        int nsize = std::stoi(cmd.GetParameter());
        int aspect_ratio = false;
        int random_padding = false;
        auto pad = true;
        std::string padding_str=cmd.GetParameter();
        if (!padding_str.empty()) pad = (bool)std::stoi(padding_str);
        auto aspect_ratio_str = cmd.GetParameter();
        if (!aspect_ratio_str.empty()) aspect_ratio = (bool)std::stoi(aspect_ratio_str);
        auto random_padding_str = cmd.GetParameter();
        if (!aspect_ratio_str.empty()) random_padding = (bool)std::stoi(random_padding_str);
        resize(nsize, pad, aspect_ratio, random_padding);
        std::cout << "resized to " << nsize << "x" << nsize << "\n";
    }
    
    if (cmd.GetCommand() == "save")
        saveBlob();

    if (cmd.GetCommand() == "display") {
        int no = std::stoi(cmd.GetParameter());
        display(no);
        std::cout << "class: " << getLabelsString(no) << "\n";
    }

    if (cmd.GetCommand() == "correct") {
        int sample_no = std::stoi(cmd.GetParameter());
        int correct_lbl = std::stoi(cmd.GetParameter());
        correctLabel(sample_no, correct_lbl);
        std::cout << sample_no << " has been corrected to" << translation.toText(correct_lbl) << "\n";
    }

    if (cmd.GetCommand() == "remove") {
        int sample_no = std::stoi(cmd.GetParameter());
        removeSample(sample_no);
        std::cout << sample_no << " has been removed\n";
    }

    if (cmd.GetCommand() == "print_distribution")
        printDistribution();

    if (cmd.GetCommand() == "transform") {
        transform(Transformation(cmd.GetParameter()));
    }
        
}

void DataBlob::slice(int memory_sz) {
    int i=0;
    if (!translation.size()) LOG << "ERROR: corrupt translation (size equals to 0)\n";
    while (i<memory_sz) {
        uchar s_width=original_data[i];
        uchar s_height= original_data[i+1];
        uchar lblsz= original_data[i+2];
        i+=3;
        std::vector<uchar> lbls_short(lblsz);
        for (int j=0;j<lblsz;j++) {
            lbls_short[j]=original_data[j+i];
        }
        i+=lblsz;
        int s_size=s_width*s_height;
        std::vector<uchar> dat(s_size);
        for (int j=0;j<s_size;j++) {
            dat[j]=original_data[j+i];
        }
        i+=s_size;

        maindata.push_back({dat,lbls_short,s_width, s_height, (int)translation.size()}); // pushing original data to vector for utility functions and random access

    }
    originals=maindata;
}

void DataBlob::resize(int nsize, bool pad, bool preserve_aspect_ratio, bool random_padding) {

    vector_dimension=nsize*nsize;
    this->width=nsize;
    this->height=nsize;
    if(pad) {
        for (auto &od : maindata) {
            od.Pad(nsize,random_padding);
        }
    }
    else {
        for (auto &od : maindata) {
            od.resize(nsize,preserve_aspect_ratio);
        }
    }
}

int DataBlob::getMemorySize() {
    int memsz=0;
    for (auto &s : originals) {
        memsz+=3;
        memsz+=(int)s.labels_short.size();
        memsz+=s.matrix_cols*s.matrix_rows;
    }
    return memsz;
}

void DataBlob::saveBlob() {

    uchar *data_package=(uchar*)std::malloc(getMemorySize()*sizeof(uchar));
    uchar *original_pointer=data_package;

    size_t i=0;
    for (auto &s : originals) {
        data_package[i]=s.matrix_cols;
        i++;
        data_package[i]=s.matrix_rows;
        i++;
        data_package[i]=s.labels_short.size();
        i++;
        if (s.labels_short.size()) std::memcpy(&data_package[i],&s.labels_short[0],s.labels_short.size()*sizeof(uchar));
        i+=s.labels_short.size();
        std::memcpy(&data_package[i],&s.data_matrix[0],s.matrix_cols*s.matrix_rows*sizeof(uchar));
        i+=s.matrix_cols*s.matrix_rows;
    }

    std::fstream out (file_src, std::ios_base::out | std::ios_base::binary);
    out << int(getMemorySize());
    out.write(reinterpret_cast<char*>(data_package),getMemorySize()*sizeof(uchar));
    std::free(original_pointer);
}

void DataBlob::uploadSample(cv::Mat sample, const int character) {
    if (character==-1) {
        LOG << "ERROR: Class unidentified\n";
        return;
    }
    uchar sample_w=sample.size().width;
    uchar sample_h=sample.size().height;
    uchar *src;
    if (sample.isContinuous()) src=sample.ptr();
    else {
        src=(uchar*)std::malloc(sample_h*sample_w*sizeof(uchar));
        for (int y=0;y<sample_h;y++) {
            for (int x=0;x<sample_w;x++) {
                src[x+y*sample_h]=sample.ptr(y)[x];
            }
        }
    }
    auto src_vec=std::vector<uchar>(&src[0],&src[sample_h*sample_w]);
    auto lbls=std::vector<uchar>(character);
    originals.push_back({src_vec, lbls, sample_w, sample_h, (int)this->translation.size()});

    saveBlob();
}

void DataBlob::addSample(cv::Mat sample, const std::vector<uchar> labels) {

    uchar sample_w=sample.size().width;
    uchar sample_h=sample.size().height;
    uchar *src;
    if (sample.isContinuous()) src=sample.ptr();
    else {
        src=(uchar*)std::malloc(sample_h*sample_w*sizeof(uchar));
        for (int y=0;y<sample_h;y++) {
            for (int x=0;x<sample_w;x++) {
                src[x+y*sample_h]=sample.ptr(y)[x];
            }
        }
    }
    auto src_vec=std::vector<uchar>(&src[0],&src[sample_h*sample_w]);
    originals.push_back({src_vec, labels, sample_w, sample_h, (int)this->translation.size()});
}

void DataBlob::uploadSample(OriginalSample sample) {
    originals.push_back({sample.data_matrix, sample.labels_short, sample.matrix_cols, sample.matrix_rows,(int)this->translation.size()});
    saveBlob();
}

void DataBlob::printSample(int no) {
    if (!this->rangeCheck(no)) return;
    LOG << this->getLabelsString(no) << "\n";
    for (size_t i=0;i<originals[no].data_matrix.size();i++) {
        if (i%originals[no].matrix_cols==0) LOG << "\n";
        LOG << originals[no].data_matrix[i] << " ";
    }
    LOG << "\n";
}


cv::Mat DataBlob::toCvMat(const int sample_no) {
    if (!this->rangeCheck(sample_no)) return {};
    auto smpl=maindata[sample_no];
    cv::Mat pic(smpl.matrix_rows,smpl.matrix_cols,CV_8U);
    int length=smpl.matrix_rows*smpl.matrix_cols;
    for (int i=0;i<length;i++) pic.at<uchar>(i)=(uchar)(smpl.data_matrix[i]);
    return pic;
}

bool DataBlob::rangeCheck(int no) {
    if ((size_t)no>=maindata.size()&&(size_t)no>=originals.size()) {
        LOG << "index of sample exceeds dataset size\n";
        return false;
    }
    return true;
}

void DataBlob::correctLabel(const int no, const int correct_label) {
    maindata[no].labels_short.clear();
    maindata[no].labels_short.push_back(correct_label);
    originals[no].labels_short.clear();
    originals[no].labels_short.push_back(correct_label);
    saveBlob();
}

void DataBlob::print() {
    LOG << "Data vector length: " << vector_dimension << ", ";
    LOG << "Blob size: " << this->size() << "\n";
    LOG << "Classes count: " << translation.size() << "\n";
}

void DataBlob::removeSample(const int no) {
    if (!this->rangeCheck(no)) return;
    maindata.erase(maindata.begin()+no);
    originals.erase(originals.begin()+no);
    this->saveBlob();
}

void DataBlob::removeClass(const std::string class_character) {
    int class_no=translation.fromText(class_character);
    if (class_no==-1) return;

    auto it=maindata.begin();
    while (it!=maindata.end()) {
        if (std::find(it->labels_short.begin(),it->labels_short.end(),class_no)!=it->labels_short.end()) {
           it=maindata.erase(it);
        }
        else
            ++it;
    }

    it=originals.begin();
    while (it!=originals.end()) {
        if (std::find(it->labels_short.begin(),it->labels_short.end(),class_no)!=it->labels_short.end()) {
           it=originals.erase(it);
        }
        else
            ++it;
    }

    for (auto &original : originals ) {
        for (auto &cl : original.labels_short) {
            if (cl>class_no) cl--;
        }
    }

    for (auto &main : maindata ) {
        for (auto &cl : main.labels_short) {
            if (cl>class_no) cl--;
        }
    }
    translation.eraseClass(class_no);
    this->saveBlob();
}

void DataBlob::printDistribution() {
    std::vector<int> res(translation.size());
    for (size_t i=0;i<res.size();i++)
        res[i]=std::count_if(this->maindata.begin(), this->maindata.end(),[i](auto a){return std::find(a.labels_short.begin(), a.labels_short.end(),i)==a.labels_short.end();});

    for (size_t i=0;i<res.size();i++)
        LOG << res[i] << " - " << translation.toText(i) << "\n";

}

void DataBlob::transform(Transformation transformat) {
    transforms.insert(transforms.begin(),transformat);
    for (size_t i=0;i<maindata.size();i++) {
        transformat.apply(&maindata[i].data_matrix_f[0],maindata[i].data_matrix_f.size());
    }
    std::cout << "Data transformation is complete\n";

}

DataBlob::XY DataBlob::at(int no) {
    return {&maindata[no].data_matrix_f[0], &maindata[no].labels[0]};
}

void DataBlob::display(int no, bool original) {
    if (!this->rangeCheck(no)) return;
    OriginalSample smpl;
    if (original)
        smpl=originals[no];
    else  smpl=maindata[no];

    cv::Mat pic(smpl.matrix_rows,smpl.matrix_cols,CV_8U,&smpl.data_matrix[0]);
    cv::namedWindow("Sample");
    cv::imshow("Sample",pic);
    cv::waitKey();
}

void DataBlob::transferSample(DataBlob *where, int no) {
    auto package=originals[no];
    this->removeSample(no);
    where->uploadSample(package);
}

void DataBlob::distributionGraph() {

    std::vector<int> res(translation.size());
    for (size_t i=0;i<res.size();i++)
        res[i]=std::count_if(this->maindata.begin(), this->maindata.end(),[i](auto a){return std::find(a.labels_short.begin(), a.labels_short.end(),i)==a.labels_short.end();});


    std::vector<double> x(res.size()), y1(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = i;
        y1[i] = res[i];
    }
    auto axes = CvPlot::makePlotAxes();
    axes.create<CvPlot::Series>(x, y1, "-r");
    CvPlot::show("mywindow", axes);

}

std::string DataBlob::getLabelsString(int no) {
    auto dat=originals[no];
    std::string out="";
    for (size_t i=0;i<dat.labels_short.size();i++) {
        out+=translation.toText(dat.labels_short[i]);
        if (i+1!=dat.labels_short.size()) out+=", ";
    }
    return out;
}

void DataBlob::loadFromFile() {
        std::string filename=OpenWindowsFileDialog("Blobs", ".blob","K:\\Blobs\\");
        if (!filename.empty()) {
            std::string translation_filename=OpenWindowsFileDialog( "TranslationUnit Files", ".trun", "K:\\TranslationUnits\\");
            if(!translation_filename.empty()) {
                *this=DataBlob(filename,translation_filename);
            }
            else {
                LOG << "ERROR: translation wasn't loaded\n";
                return;
            }
        }
        else {
            LOG << "ERROR: blob wasn't loaded\n";
            return;
        }
}

void DataBlob::loadTranslation() {
    translation.loadUnitFromFile();
}

void DataBlob::splitBlob(std::string new_blob_name, int percentage_of_data_going_to_new_blob) {
    DataBlob newblob(new_blob_name);

    std::random_device dev;
    std::default_random_engine e1(dev());
    std::shuffle(originals.begin(),originals.end(),e1);

    int border=(int)originals.size()*(100-percentage_of_data_going_to_new_blob)/100.0f;
    LOG << border << " - border\n";
    newblob.originals=std::vector<OriginalSample>(this->originals.begin()+border,originals.end());
    this->originals=std::vector<OriginalSample>(this->originals.begin(),this->originals.begin()+border);
    newblob.saveBlob();
    this->saveBlob();
}

}
