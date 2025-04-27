#include "Neuron.hpp"

Neuron::Neuron(size_t inputSize, double learningRate):
learningRate(learningRate)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dis(0.0, std::sqrt(2.0 / inputSize)); // HE initialization
    for (size_t i = 0; i < inputSize; ++i)
        weights.push_back(dis(gen));
    bias = 0.0; // usually better to start bias at 0
}

Neuron::~Neuron()
{
}

double Neuron::forward(const std::vector<double>& input)
{
    double sum = bias;
    for (size_t i = 0; i < input.size(); ++i)
        sum += input[i] * weights[i];
    return sum;
}
void Neuron::backward(const std::vector<double>& input, double delta)
{
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] -= learningRate * input[i] * delta;
    bias -= learningRate * delta;
}

std::vector<double> Neuron::getWeights() const
{
    return (weights);
}
