# SmallNN
Neural network library project


This is neural network implementation built from scratch. Primary use of the library is image classification and recognition.

For now only simplest architectures are available:
1. Fully connected layers;
2. Convolutional layers;
3. Pooling layers;

Single class and multi class classifications are possible, depending on chosen loss function.

Every datablob needs Translation Unit to convert numerical class to respected symbol.
Datablobs organized as follows:
1. Size of entire dataset in bytes;
2. Sample width;
3. Sample height;
4. Number of classes attributed to the sample;
5. Data of size (width x height);
6. Label data of size equal to number of classes attributed to the sample;
This type of dataset storage scheme permits to keep data of different sizes and classes.
All data is stored as uchar, it leads to restrictions on size of input (lower than 255x255) and number of classes (less than 255). This issue will be addressed after development of fully convolutional neural networks.

As dataset are loaded you are free to resize images and/or pad, preserving aspect ratio of not (Library responsible for resizing is avir). You also can make certain transformations of underlying data, for example in my networks I found it useful to transform X to sin(X) or cos(X). You can also display any image by index by running Display function.

Datasets can be found at https://github.com/goodle06/NNDataset, so is translation unit.

External dependencies:
1. OpenCV;
2. cvPlot (https://github.com/Profactor/cv-plot)
3. avir (https://github.com/avaneev/avir)
4. Intel MKL

Installation guide:
1. Clone repository;
2. Run CMake
3. Ready to go.

You can download datasets and translation units with this link (https://github.com/goodle06/NNDataset), put appropriate paths to default.netcf file and run command "load_config" and train your own net. Configuration file syntax described in NetConfigTemplate, so you can configure net with this tool or with standard input (std::cin).
