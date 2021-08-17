#pragma once

#define LAYER 1
#define LOSSFUNC 2
#define TRAIN_DATA 4
#define TEST_DATA 8

#define WEIGHTS 16
#define TRAIN_OPTS 32
#define CONNECT 64


#include <NeuralNetwork/NeuralNetwork.h>

namespace NN {

	class NetState {
		std::string m_name;
	public:
		NetState() = default;
		NetState(std::string name) : m_name(name) {};
		virtual ~NetState() = default;

		virtual void Train(NeuralNetwork* net) { default_behavior(); }
		virtual void SetTrainingOptions(NeuralNetwork* net, const int epochs, const int batchSize, const float learing_rate) { default_behavior(); };
		virtual void Connect(NeuralNetwork* net) { default_behavior(); };
		virtual void AddTrainData(NeuralNetwork* net, DataBlob* blob) { default_behavior(); };
		virtual void AddTestData(NeuralNetwork* net, DataBlob* blob) { default_behavior(); };
		virtual void SeedWeights(NeuralNetwork* net, const float lower_bound, const float upper_bound) { default_behavior(); };
		virtual void SetLossFunction(NeuralNetwork* net, LossFunction* function) { default_behavior(); };
		virtual void Run(NeuralNetwork* net) { default_behavior(); };
		virtual void AddLayer(NeuralNetwork* net, Layer* layer) { default_behavior(); };

		virtual std::string GetName() const { return m_name; }
	protected:
		virtual void ChangeState(NeuralNetwork* net, NetState* state) {
			net->ChangeState(state);
			std::cout << "New state: " << state->GetName() << "\n";
		}
	private:
		void default_behavior() { std::cout << "ERROR: Incorrect state, current state: " << GetName() << "\n"; }
	};

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
	class Trained : public NetState {
	public:
		Trained() : NetState("trained") {}
		virtual void Run(NeuralNetwork* net) override {
			net->RunOnce();
		}
		virtual void Train(NeuralNetwork* net) override {
			net->train();
		}
	};
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
	class ReadyToTrain : public NetState {
	public:
		ReadyToTrain() : NetState("ready to train") {}
		virtual void Train(NeuralNetwork* net) override {
			net->train();
			NetState::ChangeState(net, new Trained);
		}
	};

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
	class Connected : public NetState {
	public:
		Connected() : NetState("connected") {}
		virtual void SetTrainingOptions(NeuralNetwork* net, const int epochs, const int batchSize, const float learing_rate) {
			net->SetTrainingOptions(epochs, batchSize, learing_rate);
			NetState::ChangeState(net, new ReadyToTrain);
		}
	};
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
	class ReadyToSeedWeights : public NetState {
	public:
		ReadyToSeedWeights() : NetState("ready to seed weights") {}
		virtual void SeedWeights(NeuralNetwork* net, const float lower_bound, const float upper_bound) override {
			net->connect();
			net->SeedWeights(lower_bound, upper_bound);
			NetState::ChangeState(net, new Connected);
		}
	};
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
	class Uninitialized : public NetState {
	public:
		Uninitialized() : NetState("uninitialized") {}
		virtual void AddLayer(NeuralNetwork* net, Layer* layer) override {
			net->addLayer(layer);
			m_bitmask |= LAYER;
			ChangeState(net, new ReadyToSeedWeights);
		}
		virtual void AddTrainData(NeuralNetwork* net, DataBlob* data) override {
			net->addTrainingData(data);
			m_bitmask |= TRAIN_DATA;
			ChangeState(net, new ReadyToSeedWeights);
		}
		virtual void AddTestData(NeuralNetwork* net, DataBlob* data) override {
			net->addTestingData(data);
			m_bitmask |= TEST_DATA;
			ChangeState(net, new ReadyToSeedWeights);
		}
		virtual void SetLossFunction(NeuralNetwork* net, LossFunction* function) override {
			net->SetLossFunction(function);
			m_bitmask |= LOSSFUNC;
			ChangeState(net, new ReadyToSeedWeights);
		}
	private:
		long m_bitmask = 0;
		bool CheckBitMask() {
			auto mask = std::bitset<4>(m_bitmask);
			return std::bitset<4>(m_bitmask).all();
		}
		void ChangeState(NeuralNetwork* net, NetState* state) override {
			if (CheckBitMask())
				NetState::ChangeState(net, state);
		}
	};





}