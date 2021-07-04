#pragma once


#include <Proj/NeuralNetwork/CommandParser.h>

namespace NN {

	class Object {
	public:
		virtual void ExecuteCommand(NetCommand& cmd) = 0;
	};

}