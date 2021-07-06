#ifndef WRAPPERWINDOWSFILEDIALOG_H
#define WRAPPERWINDOWSFILEDIALOG_H

#include <Windows.h>
#include <string>

#include <shobjidl.h>
#include <atlbase.h>  

struct ComInit
{
    ComInit() { CoInitialize(nullptr); }
    ~ComInit() { CoUninitialize(); }
};

std::string OpenWindowsFileDialog();
std::string OpenWindowsFileDialog(std::string fileType, std::string fileExtension, std::string initial_dir = "");






#endif // !WRAPPERWINDOWSFILEDIALOG_H


