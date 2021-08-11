cmake_minimum_required(VERSION 3.18)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(small_nn)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src/cmake")

set(SOURCES
	SmallNN.cpp

	src/DesignPatterns/Factory.cpp

	src/WrappersAndServices/FStreamInterface.cpp
	src/WrappersAndServices/WindowsWrappers.cpp

	src/NeuralNetwork/Blobs/DataBlob.cpp
	src/NeuralNetwork/Blobs/DataTransformation.cpp
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
	src/NeuralNetwork/TrainingTelemetry.cpp
	src/NeuralNetwork/Registry.cpp
	src/NeuralNetwork/NetState.cpp


	include/Common.h
	
	include/DesignPatterns/Command.h
	include/DesignPatterns/Factory.h

	include/NeuralNetwork/Blobs/Datablob.h
	include/NeuralNetwork/Blobs/DataTransformation.h
	include/NeuralNetwork/Blobs/OriginalSample.h
	include/NeuralNetwork/Blobs/TranslationUnit.h

	include/NeuralNetwork/Layers/ConvolutionalLayer.h
	include/NeuralNetwork/Layers/DeconvolutionalLayer.h
	include/NeuralNetwork/Layers/Layer.h
	include/NeuralNetwork/Layers/LayerFactory.h
	include/NeuralNetwork/Layers/PoolingLayer.h

	include/NeuralNetwork/ActivationFunction.h
	include/NeuralNetwork/ActivationFunctionFactory.h
	include/NeuralNetwork/CommandParser.h
	include/NeuralNetwork/Estimate.h
	include/NeuralNetwork/LossFunction.h
	include/NeuralNetwork/LossFunctionFactory.h
	include/NeuralNetwork/NetTelemetry.h
	include/NeuralNetwork/NeuralNetwork.h
	include/NeuralNetwork/Object.h
	include/NeuralNetwork/TrainingTelemetry.h
	include/NeuralNetwork/Registry.h
	include/NeuralNetwork/NetState.h

	include/WrappersAndServices/FStreamInterface.h
	include/WrappersAndServices/WindowsWrappers.h

)

option(USE_OPENCV "Using openCV" ON)



add_executable(small_nn ${SOURCES})

target_include_directories(small_nn PRIVATE 
	${PROJECT_SOURCE_DIR}/include
	${PROJECT_SOURCE_DIR}/include/3party/CvPlot/CvPlot/inc
	${PROJECT_SOURCE_DIR}/include/3party
)

set(MKL_USE_STATIC_LIBS ON)
find_package(MKL REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE MKL::Libs)

