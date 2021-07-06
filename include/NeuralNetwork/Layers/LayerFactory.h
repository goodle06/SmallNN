#pragma once

#include <NeuralNetwork/Layers/Layer.h>
#include <NeuralNetwork/Layers/ConvolutionalLayer.h>
#include <NeuralNetwork/Layers/PoolingLayer.h>
#include <NeuralNetwork/Layers/DeconvolutionalLayer.h>
#include <DesignPatterns/Factory.h>


namespace NN {
	template<typename T> NN::Layer* SpawnLayer() { return new T; }
}