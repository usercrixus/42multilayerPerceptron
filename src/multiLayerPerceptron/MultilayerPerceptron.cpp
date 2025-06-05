#include "MultilayerPerceptron.hpp"

MultilayerPerceptron::MultilayerPerceptron()
{
}

MultilayerPerceptron::MultilayerPerceptron(std::vector<size_t> layerSizes)
{
    for (size_t i = 0; i < layerSizes.size() - 1; ++i)
        layers.emplace_back(layerSizes[i], layerSizes[i + 1], i != layerSizes.size() - 2);
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

bool MultilayerPerceptron::saveModelObject(const std::string& filename)
{
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open())
        return (false);
    char buffer[1024];
    out.rdbuf()->pubsetbuf(buffer, sizeof(buffer));
    size_t numLayers = layers.size();
    out.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));

    for (auto& layer : layers) {
        size_t inputSize = layer.getInputSize();
        size_t outputSize = layer.getOutputSize();
        bool isHidden = layer.getIsHiddenLayer();

        out.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
        out.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));
        out.write(reinterpret_cast<const char*>(&isHidden), sizeof(isHidden));

        auto neurons = layer.getNeurons();
        size_t numNeurons = neurons.size();
        out.write(reinterpret_cast<const char*>(&numNeurons), sizeof(numNeurons));

        for (auto& neuron : neurons) {
            double bias = neuron.getBias();
            out.write(reinterpret_cast<const char*>(&bias), sizeof(bias));

            auto weights = neuron.getWeights();
            size_t numWeights = weights.size();
            out.write(reinterpret_cast<const char*>(&numWeights), sizeof(numWeights));
            out.write(reinterpret_cast<const char*>(weights.data()), numWeights * sizeof(double));
        }
    }
    out.close();
    return (true);
}

bool MultilayerPerceptron::loadModelObject(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
        return (false);
    size_t numLayers;
    in.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));
    layers.clear();

    for (size_t l = 0; l < numLayers; ++l) {
        size_t inputSize, outputSize;
        bool isHidden;

        in.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
        in.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
        in.read(reinterpret_cast<char*>(&isHidden), sizeof(isHidden));

        Layer layer(inputSize, outputSize, isHidden);

        size_t numNeurons;
        in.read(reinterpret_cast<char*>(&numNeurons), sizeof(numNeurons));
        std::vector<Neuron> neurons;
        for (size_t n = 0; n < numNeurons; ++n) {
            double bias;
            in.read(reinterpret_cast<char*>(&bias), sizeof(bias));

            size_t numWeights;
            in.read(reinterpret_cast<char*>(&numWeights), sizeof(numWeights));
            std::vector<double> weights(numWeights);
            in.read(reinterpret_cast<char*>(weights.data()), numWeights * sizeof(double));

            Neuron neuron(numWeights);
            neuron.setWeights(weights);
            neuron.setBias(bias);
            neurons.push_back(neuron);
        }
        layer.setNeurons(neurons);
        layers.push_back(layer);
    }
    in.close();
    return (true);
}
