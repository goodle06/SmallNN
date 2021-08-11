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
	public:
		NetState() = default;
		virtual ~NetState() = default;

		virtual void Train(NeuralNetwork* net) {};
		virtual void SetTrainingOptions(NeuralNetwork* net, const int epochs, const int batchSize, const float learing_rate) {};
		virtual void Connect(NeuralNetwork* net) {};
		virtual void AddTrainData(NeuralNetwork* net, DataBlob* blob) {};
		virtual void AddTestData(NeuralNetwork* net, DataBlob* blob) {};
		virtual void SeedWeights(NeuralNetwork* net, const float lower_bound = 0.0f, const float upper_bound = 0.01f) {};
		virtual void SetLossFunction(NeuralNetwork* net, LossFunction* function) {};
		virtual void Run(NeuralNetwork* net) {};
		virtual void AddLayer(NeuralNetwork* net, Layer* layer) {};

	protected:
		virtual void ChangeState(NeuralNetwork* net, NetState* state) {
			net->ChangeState(state);
		}
	};

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
	class Trained : public NetState {
	public:
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
		virtual void Train(NeuralNetwork* net) override {
			net->train();
			ChangeState(net, new Trained);
		}
	};

/*-----------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
	class ReadyToSeedWeights : public NetState {
	public:
		virtual void SeedWeights(NeuralNetwork* net, const float lower_bound = 0.0f, const float upper_bound = 0.01f) override {
			net->connect();
			net->SeedWeights(lower_bound, upper_bound);
			ChangeState(net, new ReadyToTrain);
		}
	};
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
/*-----------------------------------------------------------------------------------------------------------------------------------------*/
	class Uninitialized : public NetState {
	public:
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
			return std::bitset<sizeof(long)>(m_bitmask).all();
		}
		void ChangeState(NeuralNetwork* net, NetState* state) override {
			if (CheckBitMask())
				NetState::ChangeState(net, state);
		}
	};





}