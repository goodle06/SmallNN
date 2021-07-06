#pragma once

#include <Proj/NeuralNetwork/LossFunction.h>
#include <Proj/DesignPatterns/Factory.h>


namespace NN {
	template<typename T> NN::LossFunction* SpawnLossFunction() { return new T; }
}