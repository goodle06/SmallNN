#include <WrappersAndServices/WindowsWrappers.h>

std::string OpenWindowsFileDialog(std::string fileType, std::string fileExtension, std::string initial_dir) {

    size_t sz1 = fileType.size();
    size_t sz2 = fileExtension.size();

    char* filter = new char[sz1 + sz2 + 2];
    std::memcpy(filter, &fileType[0], sizeof(char) * (sz1));
    std::memcpy(filter + sz1, &fileExtension[0], sizeof(char) * (sz2));
    filter[sz1 + sz2] = '\0';
    filter[sz1 + sz2 + 1] = '\0';

    char* dir = new char[initial_dir.size() + 1];
    std::memcpy(dir, &initial_dir[0], sizeof(char) * initial_dir.size());
    dir[initial_dir.size()] = '\0';

    OPENFILENAMEA ofn_struct = {};
    std::string filename(MAX_PATH, L'\0');
    ofn_struct.lStructSize = sizeof(ofn_struct);
    ofn_struct.hwndOwner = NULL;
    ofn_struct.lpstrFilter = filter;
    if (dir != "")
        ofn_struct.lpstrInitialDir = dir;
    ofn_struct.nMaxFile = MAX_PATH;
    ofn_struct.lpstrTitle = "Select a file";
    ofn_struct.Flags = OFN_DONTADDTORECENT | OFN_FILEMUSTEXIST;
    ofn_struct.lpstrFile = &filename[0];

    std::string result;
    if (GetOpenFileNameA(&ofn_struct))
        result = filename;

    delete[] filter;
    delete[] dir;
    return result;

}


std::string OpenWindowsFileDialog() {
    ComInit com;

    // Create an instance of IFileOpenDialog.
    CComPtr<IFileOpenDialog> pFolderDlg;
    pFolderDlg.CoCreateInstance(CLSID_FileOpenDialog);

    // Set options for a filesystem folder picker dialog.
    FILEOPENDIALOGOPTIONS opt{};
    pFolderDlg->GetOptions(&opt);
    pFolderDlg->SetOptions(opt | FOS_PICKFOLDERS | FOS_PATHMUSTEXIST | FOS_FORCEFILESYSTEM);

    // Show the dialog modally.
    if (SUCCEEDED(pFolderDlg->Show(nullptr)))
    {
        // Get the path of the selected folder and output it to the console.

        CComPtr<IShellItem> pSelectedItem;
        pFolderDlg->GetResult(&pSelectedItem);

        CComHeapPtr<wchar_t> pPath;
        pSelectedItem->GetDisplayName(SIGDN_FILESYSPATH, &pPath);

        std::wstring wstr(pPath.m_pData);
        std::string str(wstr.begin(), wstr.end());
        return str;
    }
    return "";
    // Else dialog has been canceled. 
}