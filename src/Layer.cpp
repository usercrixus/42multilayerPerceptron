#include "Layer.hpp"

Layer::Layer(size_t inputSize, size_t output_size, double learningRate):
inputSize(inputSize),
outputSize(output_size),
learningRate(learningRate)
{
    for (size_t i = 0; i < output_size; ++i)
        neurons.emplace_back(inputSize, learningRate);
}

Layer::~Layer()
{
}
double relu(double x) {
    return x > 0 ? x : 0;
}

std::vector<double> Layer::forward(const std::vector<double>& input)
{
    std::vector<double> output;
    for (auto& neuron : neurons) {
        double sum = neuron.forward(input);  // raw sum
        output.push_back(relu(sum));          // apply ReLU here
    }
    return output;
}

void Layer::backward(const std::vector<double>& input, const std::vector<double>& deltas)
{
    for (size_t i = 0; i < neurons.size(); ++i)
        neurons[i].backward(input, deltas[i]);
}

std::vector<Neuron> Layer::getNeurons() const
{
    return (neurons);
}

size_t Layer::getInputSize() const
{
    return inputSize;
}
