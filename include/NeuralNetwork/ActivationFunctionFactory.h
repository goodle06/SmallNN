#pragma once

#include <Proj/DesignPatterns/Factory.h>
#include <Proj/NeuralNetwork/ActivationFunction.h>

namespace NN {
	template <typename T> ActivationFunction* SpawnActivationFunction() { return new T; }
}