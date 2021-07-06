#pragma once


#include <NeuralNetwork/CommandParser.h>

namespace NN {

	class Object {
	public:
		virtual void ExecuteCommand(NetCommand& cmd) = 0;
	};

}