#include <Proj/NeuralNetwork/Blobs/TranslationUnit.h>

namespace NN {


TranslationUnit::TranslationUnit(std::string filename_src) {

    filename=filename_src;
    std::fstream in(filename_src,std::ios_base::in);
    if (in.is_open())
    {
       while (!in.eof())
       {
          std::string line;
          std::getline(in,line);
          std::regex re("(\\d+)\\s(.+)");
          auto matches_begin=std::sregex_iterator(line.begin(),line.end(),re);
          auto matches_end=std::sregex_iterator();

          for (auto i=matches_begin;i!=matches_end;i++) {
              std::match_results match=*i;
              int no=std::stoi(match[1]);
              std::string str= match[2];
              auto res=m_translation.insert(std::map<int, std::string>::value_type(no, str));
              if (!res.second) LOG << "ERROR: insertion failed" << "\n";
          }
          m_loaded=true;
       }
       in.close();
    }

    for (auto t : m_translation) {
        auto res=m_reverse_translation.insert(std::map<std::string, int>::value_type(t.second,t.first));
        if (!res.second) LOG << "ERROR: insertion failed" << "\n";
    }

}

std::string TranslationUnit::toText(int classNo) {
    auto f=m_translation.find(classNo);
    if (f!=m_translation.end()) return f->second;
    else {
        LOG << classNo << " isn't found\n";
        return "";
    }
}

bool TranslationUnit::loadUnitFromFile() {
    std::string flname=OpenWindowsFileDialog("TranslationUnit Files", ".trun","K:\\TranslationUnits\\" );
    filename = flname;
    if (!flname.empty()) {
        *this=TranslationUnit(filename);
        m_loaded=true;
        return true;
    }
    else {
        return false;
    }
}

void TranslationUnit::printUnit() {
    LOG << "Translation unit:\n";
    for (auto u : m_translation) {
        LOG << u.first << " - " << u.second << "\n";
    }
    LOG << "\n";
}

int TranslationUnit::fromText(std::string text) {
    if (!m_reverse_translation.size()) {
        LOG << "translation unit is empty\n";
        return -1;
    }

    auto it=m_reverse_translation.find(text);
    if (it==m_reverse_translation.end()) {
        LOG << "Unknown syllable\n";
        return -1;
    }

    return m_reverse_translation.at(text);
}

bool TranslationUnit::update(std::string src) {

    auto search=m_reverse_translation.find(src);
    if (search!=m_reverse_translation.end()) {
        LOG << "Class already exists\n";
        return false;
    }
    int last=m_translation.size();

    auto res=m_translation.insert(std::map<int, std::string>::value_type(last, src));
    if (!res.second) {
        LOG << "ERROR: insertion failed" << "\n";
        return false;
    }
    auto reverse_res=m_reverse_translation.insert(std::map<std::string, int>::value_type(src,last));
    if (!reverse_res.second) {
        LOG << "ERROR: insertion failed" << "\n";
        return false;
    }

    return p_upload();
}

bool TranslationUnit::eraseClass(int class_no) {
    auto search=m_translation.find(class_no);
    if (search==m_translation.end()) {
        LOG << "Class not found\n";
        for (auto u : m_translation) {
            LOG << u.first << " - " << u.second << "\n";
        }
        return false;
    }
    m_translation.erase(search);
    std::map<int, std::string> dir;

    int i=0;
    for (auto p : m_translation) {
        auto res=dir.insert(std::map<int, std::string>::value_type(i, p.second));
        if (!res.second) LOG << "ERROR: insertion failed" << "\n";
        i++;
    }
    m_translation=dir;

    m_reverse_translation.clear();
    for (auto t : m_translation) {
        auto res=m_reverse_translation.insert(std::map<std::string, int>::value_type(t.second,t.first));
        if (!res.second) LOG << "ERROR: insertion failed" << "\n";
    }

    return p_upload();
}

bool TranslationUnit::p_upload() {
    std::fstream out(filename, std::ios::out);
    if (out.is_open())
        for (auto& line : m_translation)
            out << line.first << " " << line.second << "\n";
    else
        return false;
    out.close();
    return true;
}


}
