#include "MultilayerPerceptron.hpp"

MultilayerPerceptron::MultilayerPerceptron(const std::vector<size_t>& layerSizes, double learningRate):
learningRate(learningRate)
{
    for (size_t i = 0; i < layerSizes.size() - 1; ++i)
        layers.emplace_back(layerSizes[i], layerSizes[i+1], learningRate);
}

MultilayerPerceptron::~MultilayerPerceptron()
{
}
double MultilayerPerceptron::binaryCrossEntropy(double trueLabel, double predictedLabel) {
    const double epsilon = 1e-8; // To avoid log(0)
    predictedLabel = std::min(std::max(predictedLabel, epsilon), 1.0 - epsilon); // Clamp predictions to avoid log(0)

    if (trueLabel == 0)
		return (-1 * std::log(1.0 - predictedLabel));
	return -1 * (trueLabel * std::log(predictedLabel));
}

double MultilayerPerceptron::forward(const std::vector<double>& input)
{
    std::vector<double> output = input;
    for (auto& layer : layers) {
        output = layer.forward(output);
    }
    return sigmoid(output[0]); // <- apply sigmoid at the end
}

void MultilayerPerceptron::trainStep(const std::vector<double>& input, double trueLabel)
{
    std::vector<std::vector<double>> activations;
    std::vector<double> currentInput = input;
    for (auto& layer : layers) {
        currentInput = layer.forward(currentInput);
        activations.push_back(currentInput);
    }

    double predicted = activations.back()[0];
	std::vector<double> delta = { predicted - trueLabel };
    for (int i = layers.size() - 1; i >= 0; --i)
    {
        const std::vector<double>& inputToThisLayer = (i == 0) ? input : activations[i-1];
        layers[i].backward(inputToThisLayer, delta);
        if (i != 0)
            delta = computeNewDelta(layers[i], delta);
    }
}
std::vector<double> MultilayerPerceptron::computeNewDelta(const Layer& layer, const std::vector<double>& nextDelta)
{
    std::vector<double> newDelta(layer.getInputSize(), 0.0);

    for (size_t i = 0; i < layer.getNeurons().size(); ++i) {
        for (size_t j = 0; j < layer.getNeurons()[i].getWeights().size(); ++j) {
            newDelta[j] += layer.getNeurons()[i].getWeights()[j] * nextDelta[i];
        }
    }

    return newDelta;
}
double MultilayerPerceptron::sigmoid(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}
