#include "Layer.hpp"

Layer::Layer(size_t inputSize, size_t output_size, bool hiddenLayer)
    : inputSize(inputSize),
      outputSize(output_size),
      isHiddenLayer(hiddenLayer)
{
    for (size_t i = 0; i < output_size; ++i)
        neurons.emplace_back(inputSize);
}

Layer::~Layer() {}

std::vector<double> Layer::forward(std::vector<double>& input)
{
    std::vector<double> output;
    for (Neuron &neuron : neurons)
        output.push_back(neuron.forward(input));
    activeNeuron(output);
    return output;
}

std::vector<double> Layer::backward(std::vector<double>& deltas, double learningRate)
{
    std::vector<double> newDeltas(neurons[0].getWeights().size(), 0.0);

    for (size_t i = 0; i < neurons.size(); ++i)
    {
        std::vector<double> oldWeights = neurons[i].getWeights();
        double deltaToPropagate = neurons[i].backward(deltas[i], isHiddenLayer, learningRate);
        for (size_t j = 0; j < oldWeights.size(); ++j)
            newDeltas[j] += oldWeights[j] * deltaToPropagate;
    }
    return newDeltas;
}

void Layer::activeNeuron(std::vector<double> &output)
{
    if (isHiddenLayer)
        ReLu(output);
}

void Layer::ReLu(std::vector<double> &output)
{
    for (double &o : output)
        o = o > 0.0 ? o : 0.0;
}

int Layer::getInputSize()
{
    return (this->inputSize);
}

int Layer::getOutputSize()
{
    return (this->outputSize);
}

bool Layer::getIsHiddenLayer()
{
    return (this->isHiddenLayer);
}

std::vector<Neuron> &Layer::getNeurons()
{
    return (this->neurons);
}

void Layer::setNeurons(std::vector<Neuron> neurons)
{
    this->neurons = neurons;
}
