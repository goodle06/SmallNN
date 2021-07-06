#pragma once

#include <NeuralNetwork/LossFunction.h>
#include <DesignPatterns/Factory.h>


namespace NN {
	template<typename T> NN::LossFunction* SpawnLossFunction() { return new T; }
}