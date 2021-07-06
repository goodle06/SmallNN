#ifndef LAYER_H
#define LAYER_H

#include <NeuralNetwork/ActivationFunctionFactory.h>
#include <NeuralNetwork/Object.h>

#include <NeuralNetwork/Blobs/DataBlob.h>
#include <WrappersAndServices/FStreamInterface.h>
#include <NeuralNetwork/CommandParser.h>



namespace NN {

enum class LayerType : int { FC=0, Conv=1, MaxPool=2, Deconv=3, Unpool=4 };

    class Layer : public Object {
    public:
        Layer();
        Layer(Layer& other) = delete;
        Layer& operator=(Layer& other) = delete;
        /*TO DO: move constructor*/ 
        //Layer(Layer&& other);

        Layer(int dim, ActivationFunction* func);
        Layer(std::initializer_list<int> params);

        void RegisterActivationFunctions();

        virtual ~Layer();


        /*forward pass of args through net, for example fully connected layer performs Wx+b, where W - weight matrix, X - input vector, b - offset vector*/
        virtual float* RunWX(float *X, float* dest);
        /*applies activation function f to result of Wx+b operation*/
        virtual void RunActivate(float *WX, float *Y);
        /*calculates derivative of f(Wx+b), where f is activation function*/
        virtual void RunDerivative(float *WX,float *D);

        /*backward pass, propagates net output backwards, pooling map saves indices of activated neurons (only for pooling layer, otherwise not needed)*/
        /*gradient calculation is split into two phases: calculates derivative for futher propagatin backwards and local gradient for current layer, 
        split is needed since it's derivative of complex function*/
        virtual void RunBackwards(float *P, float *Y, float *derivative, float *pooling_map=nullptr);
        virtual void CalculateGradient(float *derivative, float *X);


        virtual void print(bool with_weights=false) const;
        virtual void printNeuron(const int no) const;
        virtual void printGradient() const;

        virtual void setActivationFunction(ActivationFunction* actFunc) final;
        virtual void setActivationFunction(std::string actFunc) final;
        virtual void setActivationFunction(ActivationFunctionType actFunc) final;

        void SetSize(const int dim);
        virtual void SetParams(std::initializer_list<int> params);
        virtual bool SetConnections(Layer* layer);
        virtual bool SetConnections(DataBlob& blob);

        /*initialize memory for weights and gradients*/
        virtual void assignMemory();
        virtual void assignMemory(float *load_weights);

        /*temporarily only uniform distribution is used for weights seeding */
        virtual void SeedWeights(const float lower_bound=-0.2f, const float upper_bound=0.2f);

        /*updates weights, batch size is size of dataset used to calculate accumulated gradient values*/
        virtual void UpdateWeights(const int batchSize, const float learning_rate);

        /*serializes and de- layer weights and parameters*/
        virtual void SaveStd(std::fstream &stream);
        virtual void LoadStd(std::fstream &stream);

        /*logging weights changes*/
        virtual std::vector<float> getlog(int weight_no);

        virtual LayerType Type() const {return LayerType::FC;}

        /*rolling back last weights changes*/
        virtual void Rollback();
        /*creates backup to which to rollback*/
        virtual void CreateBackup();

        LayerType mLayerType=LayerType::FC;
        
        /*setters and getters for weights freezing*/
        void FreezeWeights() { open = false; }
        void UnfreezeWeights() { open = true; }
        bool IsFrozen() { return open; }

        /*enables logging*/
        void EnableLogging() { weights_logging = true; }
        void DisableLogging() { weights_logging = false; }

        /*sets maximum weights change*/
        void SetMaximumWeightChange(float change_limit) { max_change = change_limit; }

        /*getters of internal layer params*/
        int GetRows() const { return rows; }
        int GetCols() const { return cols; }
        int GetTotalCount() const { return total_count; }

        void ExecuteCommand(NetCommand &cmd) override;

    protected:
        /*boolean variable signals if weights can be changed or frozen*/
        bool open = true;
        /*weights logging flag*/
        bool weights_logging = false;
        /*maximum absolute change, prevents exploding gradient*/
        float max_change = 0.0f;

        int rows=0;
        int cols=0;
        int total_count=0;

        float *layer_weights=nullptr;
        float *layer_gradient=nullptr;

        float *offset_vector=nullptr;
        float *offset_gradient=nullptr;

        std::vector<float> weights_log;

        /*rollback weights*/
        float *echo_weights=nullptr;

        ActivationFunction* m_activation_function=nullptr;

        /*mutex protects gradient*/
        std::mutex m_mutex;

        DesignPatterns::Factory<ActivationFunction, std::string, ActivationFunction* (*)()> activationFunctionFactoryString;
        DesignPatterns::Factory<ActivationFunction, int, ActivationFunction* (*)()> activationFunctionFactoryInt;
    private:



    };
}

#endif // LAYER_H
