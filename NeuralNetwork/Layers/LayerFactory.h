#pragma once

#include <Proj/NeuralNetwork/Layer.h>
#include <Proj/NeuralNetwork/Layers/ConvolutionalLayer.h>
#include <Proj/NeuralNetwork/Layers/PoolingLayer.h>
#include <Proj/NeuralNetwork/Layers/DeconvolutionalLayer.h>
#include <Proj/DesignPatterns/Factory.h>


namespace NN {
	template<typename T> NN::Layer* SpawnLayer() { return new T; }
}