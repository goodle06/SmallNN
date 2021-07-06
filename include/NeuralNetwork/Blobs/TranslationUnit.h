#ifndef TRANSLATIONUNIT_H
#define TRANSLATIONUNIT_H
#include <Common.h>

namespace NN {


class TranslationUnit {
public:
    TranslationUnit() {}
    TranslationUnit(std::map<int, std::string> translation_map) : m_translation(translation_map) {}
    TranslationUnit(std::string filename);

    bool isLoaded() { return m_loaded;}
    std::string toText(int classNo);
    int fromText(std::string text);
    size_t size() {return m_reverse_translation.size();}
    size_t size2() {return m_translation.size();}
    bool loadUnitFromFile();
    void printUnit();
    bool update(std::string src);
    bool eraseClass(int class_no);

private:
    std::map<int, std::string> m_translation;
    std::map<std::string, int> m_reverse_translation;
    std::string filename;
    bool m_loaded=false;
    bool p_upload();
};


}

#endif // TRANSLATIONUNIT_H
