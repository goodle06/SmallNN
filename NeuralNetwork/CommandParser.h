#pragma once
#include <string>
#include <vector>
#include <queue>


namespace NN {


	class NetCommand {
	public:
		NetCommand(const std::string &cmd);
		std::string GetCommand() { return m_cmd; }
		void SetCommand(std::string cmd) { m_cmd = cmd; }
		std::string GetParameter();
	private:
		std::string m_cmd;
		std::queue<std::string> params_queue;
	};
}

