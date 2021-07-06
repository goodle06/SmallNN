#pragma once

#include <DesignPatterns/Factory.h>
#include <NeuralNetwork/ActivationFunction.h>

namespace NN {
	template <typename T> ActivationFunction* SpawnActivationFunction() { return new T; }
}