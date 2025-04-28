#include "Neuron.hpp"
#include "Layer.hpp"

Neuron::Neuron(size_t inputSize)
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

double Neuron::forward(std::vector<double>& input)
{
    lastInput = input;
    double sum = bias;
    for (size_t i = 0; i < input.size(); ++i)
        sum += input[i] * weights[i];
    lastSum = sum;
    return sum;
}

double Neuron::backward(double delta, bool isHiddenLayer, double learningRate)
{
    double activationDerivative = 0.0;
    if (isHiddenLayer)
    {
        activationDerivative = (lastSum > 0.0) ? 1.0 : 0.01;
        delta = delta * activationDerivative;
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] -= learningRate * lastInput[i] * delta;
        bias -= learningRate * delta;
    }
    else
    {
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] -= learningRate * lastInput[i] * delta;
        bias -= learningRate * delta;
    }
    return delta;
}


std::vector<double> Neuron::getWeights()
{
    return (weights);
}

double Neuron::getLastSum()
{
    return (lastSum);
}

void Neuron::setLastSum(double sum)
{
    lastSum = sum;
}
