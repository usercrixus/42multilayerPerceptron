#include "MultilayerPerceptron.hpp"

MultilayerPerceptron::MultilayerPerceptron(std::vector<size_t> layerSizes)
{
    for (size_t i = 0; i < layerSizes.size() - 1; ++i)
        layers.emplace_back(layerSizes[i], layerSizes[i+1], i != layerSizes.size() - 2);
}

MultilayerPerceptron::~MultilayerPerceptron()
{
}
std::vector<double> MultilayerPerceptron::forward(std::vector<double>& input)
{
    std::vector<double> output = input;
    for (size_t i = 0; i < layers.size(); ++i)
        output = layers[i].forward(output);
    return output;
}



void MultilayerPerceptron::trainStep(std::vector<double>& input, double trueLabel, double learningRate)
{
    std::vector<double> prediction = forward(input);

    std::vector<double> delta(prediction.size(), 0.0);
    for (size_t i = 0; i < prediction.size(); ++i)
        delta[i] = prediction[i] - (i == static_cast<size_t>(trueLabel) ? 1.0 : 0.0);

    for (int i = layers.size() - 1; i >= 0; --i)
        delta = layers[i].backward(delta, learningRate);
}
