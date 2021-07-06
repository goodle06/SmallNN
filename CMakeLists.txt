cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

projec(small_nn)

set(SOURCES
	src/DesignPatterns/Factory.cpp

	src/WrappersAndServices/FStreamInterface.cpp
	src/WrappersAndServices/WindowsWrappers.cpp

	src/NeuralNetwork/Blobs/DataBlob.cpp
	src/NeuralNetwork/Blobs/DataTransformtion.cpp
	src/NeuralNetwork/Blobs/OriginalSample.cpp
	src/NeuralNetwork/Blobs/TranslationUnit.cpp

	src/NeuralNetwork/Layers/Layer.cpp
	src/NeuralNetwork/Layers/ConvolutionalLayer.cpp
	src/NeuralNetwork/Layers/PoolingLayer.cpp
	src/NeuralNetwork/Layers/DeconvolutionalLayer.cpp
	src/NeuralNetwork/Layers/LayerFactory.cpp

	src/NeuralNetwork/ActivationFunction.cpp
	src/NeuralNetwork/CommandParser.cpp
	src/NeuralNetwork/Estimate.cpp
	src/NeuralNetwork/LossFunction.cpp
	src/NeuralNetwork/NetTelemetry.cpp
	src/NeuralNetwork/NeuralNetwork.cpp
	src/NeuralNetwork/TrainingTelemtry.cpp
)

option(USE_OPENCV "Using openCV" ON)

add_executables(small_nn ${SOURCES})

target_include_directories(small_nn PRIVATE ${PROJECT_SOURCE_DIR}/include)

